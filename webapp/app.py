import os
import sys
import tempfile
import base64
import io
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
import torch
import numpy as np
from PIL import Image
import hashlib
import json
import requests

# Ensure project root is on sys.path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

# Reuse preprocessing from existing inference code
from inference.predict import preprocess_mri
from models.cnn_model import AlzheimerCNN
try:
	from models.true_quantum_model import TrueQuantumHybridModel
except Exception:
	TrueQuantumHybridModel = None

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "secret-key")
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_BYTES', str(2 * 1024 * 1024 * 1024)))

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_alzheimers_cnn.pth"
QUANTUM_MODEL_PATH = Path(os.environ.get("QUANTUM_MODEL_PATH", str(PROJECT_ROOT / "models" / "hospital_1_best_quantum.pth")))
DATA_DIR = PROJECT_ROOT / "data"
HOSPITALS_DIR = DATA_DIR

# Lazy-loaded model
_model = None
_quantum_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple in-memory job status
_jobs = {}

def get_model():
	global _model
	if _model is None:
		if not BEST_MODEL_PATH.exists():
			raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")
		# Allow selecting quantum model via env var
		if os.environ.get("USE_QUANTUM_MODEL", "0") == "1" and TrueQuantumHybridModel is not None:
			model = TrueQuantumHybridModel()
		else:
			model = AlzheimerCNN()
		state = torch.load(BEST_MODEL_PATH, map_location=_device)
		model.load_state_dict(state)
		model.to(_device)
		model.eval()
		_model = model
	return _model

def get_quantum_model_if_available():
	global _quantum_model
	if TrueQuantumHybridModel is None:
		return None
	if _quantum_model is not None:
		return _quantum_model
	try:
		if QUANTUM_MODEL_PATH.exists():
			q = TrueQuantumHybridModel()
			state = torch.load(QUANTUM_MODEL_PATH, map_location=_device)
			q.load_state_dict(state)
			q.to(_device)
			q.eval()
			_quantum_model = q
			return _quantum_model
	except Exception:
		return None
	return None

@app.route("/", methods=["GET", "POST"])
def index():
	prediction = None
	probability = None
	error = None
	image_data_url = None
	probability_pct = None
	filename = None
	if request.method == "POST":
		if "file" not in request.files:
			error = "No file part in the request"
			return render_template("index.html", prediction=prediction, probability=probability, error=error, image_data_url=image_data_url, probability_pct=probability_pct, filename=filename)
		file = request.files["file"]
		if file.filename == "":
			error = "No file selected"
			return render_template("index.html", prediction=prediction, probability=probability, error=error, image_data_url=image_data_url, probability_pct=probability_pct, filename=filename)
		try:
			# Save to a temporary file
			with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
				file.save(tmp.name)
				tmp_path = tmp.name
			# Preprocess and predict
			x = preprocess_mri(tmp_path).to(_device)
			model = get_model()
			qmodel = get_quantum_model_if_available()
			with torch.no_grad():
				logits_c = model(x)
				prob_c = torch.softmax(logits_c, dim=1)[0, 1].item()
				if qmodel is not None:
					logits_q = qmodel(x)
					prob_q = torch.softmax(logits_q, dim=1)[0, 1].item()
					prob = 0.5 * (prob_c + prob_q)
				else:
					prob = prob_c
			prediction = "HIGH RISK of Alzheimer's" if prob > 0.5 else "LOW RISK of Alzheimer's"
			probability = f"{prob:.4f}"
			probability_pct = int(round(prob * 100))
			filename = file.filename

			# Build a preview image from the preprocessed slice (works for .img/.hdr too)
			preview = x.squeeze().detach().cpu().numpy()  # (128,128) in [0,1]
			preview_uint8 = np.clip(preview * 255.0, 0, 255).astype(np.uint8)
			pil_img = Image.fromarray(preview_uint8, mode="L")
			buf = io.BytesIO()
			pil_img.save(buf, format="PNG")
			buf.seek(0)
			image_data_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")
		finally:
			# Cleanup temp file
			try:
				os.remove(tmp_path)
			except Exception:
				pass
	return render_template("index.html", prediction=prediction, probability=probability, error=error, image_data_url=image_data_url, probability_pct=probability_pct, filename=filename)

# --- Minimal dashboard endpoints ---
@app.route("/dashboard")
def dashboard():
	return render_template("dashboard.html")

@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
	hospital_id = request.form.get("hospital_id", "hospital_1").strip() or "hospital_1"
	upload_mode = request.form.get("upload_mode", "folder")
	hospital_root = HOSPITALS_DIR / hospital_id / "raw"
	hospital_root.mkdir(parents=True, exist_ok=True)

	# Folder upload (webkitdirectory)
	if upload_mode == "folder":
		files = request.files.getlist("folder")
		if not files:
			flash("No files received for folder upload", "error")
			return redirect(url_for("dashboard"))
		for f in files:
			# Preserve client-side relative paths when provided
			rel_path = f.filename.replace("\\", "/")
			# Strip any drive letters or absolute prefixes
			while ":/" in rel_path:
				rel_path = rel_path.split(":/", 1)[1]
			target_path = hospital_root / rel_path
			target_path.parent.mkdir(parents=True, exist_ok=True)
			f.save(str(target_path))
		flash(f"Uploaded {len(files)} files to {hospital_root}", "success")
		return redirect(url_for("dashboard"))

	# ZIP upload
	if upload_mode == "zip":
		zip_file = request.files.get("zipfile")
		if not zip_file or zip_file.filename == "":
			flash("No ZIP file provided", "error")
			return redirect(url_for("dashboard"))
		with tempfile.TemporaryDirectory() as td:
			tmp_zip = Path(td) / Path(zip_file.filename).name
			zip_file.save(str(tmp_zip))
			import zipfile
			with zipfile.ZipFile(str(tmp_zip), 'r') as zf:
				zf.extractall(str(hospital_root))
		flash(f"Extracted ZIP into {hospital_root}", "success")
		return redirect(url_for("dashboard"))

	flash("Unsupported upload mode", "error")
	return redirect(url_for("dashboard"))

@app.route("/start_training", methods=["POST"])
def start_training():
	hospital_id = request.form.get("hospital_id", "hospital_1").strip() or "hospital_1"
	epochs = int(request.form.get("epochs", "1") or 1)
	use_quantum = request.form.get("use_quantum", "0") == "1"
	# Launch background job: preprocess then train
	from threading import Thread

	def run_job():
		try:
			_jobs[hospital_id] = "Preprocessing dataset..."
			raw_dir = HOSPITALS_DIR / hospital_id / "raw"
			prep_dir = HOSPITALS_DIR / hospital_id / "preprocessed"
			prep_dir.mkdir(parents=True, exist_ok=True)
			# Batch preprocess: expect CONTROL/ and ALZ/ subfolders
			classes = [("CONTROL", 0), ("ALZ", 1)]
			count = 0
			for class_name, label in classes:
				class_dir = raw_dir / class_name
				if not class_dir.exists():
					continue
				for root, _, files in os.walk(class_dir):
					for fname in files:
						ext = Path(fname).suffix.lower()
						if ext not in ['.img', '.hdr', '.gif', '.jpg', '.jpeg', '.png']:
							continue
						fpath = Path(root) / fname
						try:
							tensor = preprocess_mri(str(fpath))  # (1,1,128,128)
							out = { 'image': tensor.squeeze(0), 'label': int(label) }
							out_name = f"{class_name}_{count:06d}.pt"
							torch.save(out, str(prep_dir / out_name))
							count += 1
						except Exception:
							continue
			_jobs[hospital_id] = f"Preprocessing complete: {count} samples. Starting training..."

			# Build dataloader and train
			from utils.dataset_loader import get_loader
			model_type = "Classical"
			if use_quantum and TrueQuantumHybridModel is not None:
				model = TrueQuantumHybridModel()
				model_type = "Quantum"
			else:
				model = AlzheimerCNN()
			loader = get_loader(prep_dir, batch_size=16)
			optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
			criterion = torch.nn.CrossEntropyLoss()
			model.to(_device)
			best_acc = 0.0
			best_state = None
			best_path = PROJECT_ROOT / 'models' / (f'{hospital_id}_best_quantum.pth' if model_type=="Quantum" else f'{hospital_id}_best.pth')
			param_count = sum(p.numel() for p in model.parameters())
			_jobs[hospital_id] = _jobs.get(hospital_id, "") + f"\nModel: {model_type} | params={param_count} | saving to {best_path.name}"
			import time
			for epoch in range(max(1, epochs)):
				start = time.time()
				model.train()
				total, correct = 0, 0
				for images, labels in loader:
					images, labels = images.to(_device), labels.to(_device)
					optimizer.zero_grad()
					logits = model(images)
					loss = criterion(logits, labels)
					loss.backward()
					optimizer.step()
					preds = logits.argmax(dim=1)
					correct += (preds == labels).sum().item()
					total += labels.size(0)
				acc = correct / max(1, total)
				dur = max(1e-6, time.time() - start)
				sps = total / dur
				_jobs[hospital_id] = f"Epoch {epoch+1}/{epochs} - acc={acc:.4f} | {dur:.2f}s | {sps:.1f} samples/s | {model_type}"
				if acc > best_acc:
					best_acc = acc
					best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
			# Save only once at the end using the best tracked state
			if best_state is not None:
				torch.save(best_state, str(best_path))
			# Compute SHA256 and optionally record to Fabric REST if configured
			try:
				h = hashlib.sha256()
				with open(best_path, 'rb') as f:
					for chunk in iter(lambda: f.read(1024*1024), b''):
						h.update(chunk)
				hash_hex = h.hexdigest()
				fabric_url = (os.environ.get('FABRIC_RECORD_URL') or '').strip()
				storage_uri = os.environ.get('FABRIC_STORAGE_URI', f'file://{best_path.as_posix()}')
				round_id = (os.environ.get('FL_ROUND_ID', 'local') or 'local').strip()
				if fabric_url:
					payload = {
						"updateHash": hash_hex,
						"accuracy": int(round(best_acc * 10000)),
						"hospitalId": hospital_id,
						"storageUri": storage_uri,
						"roundId": str(round_id),
						"modelType": model_type
					}
					requests.post(fabric_url, json=payload, timeout=10)
			except Exception:
				pass
			_jobs[hospital_id] = f"Done. Best acc={best_acc:.4f}. Model saved to {best_path.name} | {model_type}"
		except Exception as e:
			_jobs[hospital_id] = f"Error: {e}"

	Thread(target=run_job, daemon=True).start()
	flash(f"Queued local training for {hospital_id}: epochs={epochs}, quantum={use_quantum}", "success")
	return redirect(url_for("dashboard"))

@app.route("/job_status")
def job_status():
	hospital_id = request.args.get("hospital_id")
	if not hospital_id:
		return {"error": "hospital_id parameter required"}, 400
	return {"hospital_id": hospital_id, "status": _jobs.get(hospital_id, "No jobs running.")}

@app.route("/federated")
def federated():
	return render_template("federated.html")

@app.route("/blockchain")
def blockchain():
	# Prefer Fabric REST (default to local REST mock) and normalize for template
	from pathlib import Path as _Path
	logs = []
	fabric_history = (os.environ.get('FABRIC_HISTORY_URL', 'http://127.0.0.1:3000/history') or '').strip()
	try:
		resp = requests.get(fabric_history, timeout=5)
		if resp.ok:
			data = resp.json()
			raw_logs = data.get('data', data) if isinstance(data, dict) else data
			normalized = []
			if isinstance(raw_logs, list):
				for item in raw_logs:
					if isinstance(item, dict):
						h = item.get('updateHash') or item.get('hash') or ''
						acc_val = item.get('accuracy', 0)
						# Normalize accuracy to basis points so template (acc/100)% is correct
						if isinstance(acc_val, (float, int)):
							if acc_val <= 1:
								acc = int(round(acc_val * 10000))  # 0.958 -> 9580 (95.80%)
							elif acc_val <= 100:
								acc = int(round(acc_val * 100))    # 95.8 -> 9580
							else:
								acc = int(round(acc_val))
						else:
							acc = 0
						hosp = item.get('hospitalId') or item.get('hospital') or ''
						ts = item.get('timestamp') or item.get('time') or ''
						normalized.append((h, acc, hosp, ts))
					else:
						try:
							h, acc_in, hosp, ts = item
							acc = int(round(acc_in))
							normalized.append((h, acc, hosp, ts))
						except Exception:
							pass
			logs = normalized
			try:
				app.logger.info(f"[blockchain] logs_count={len(logs)}")
			except Exception:
				pass
		# Fallback attempt: direct 127.0.0.1 if no logs yet and env was localhost
		if not logs and 'localhost' in fabric_history:
			try:
				resp2 = requests.get('http://127.0.0.1:3000/history', timeout=5)
				if resp2.ok:
					data2 = resp2.json()
					raw2 = data2.get('data', data2) if isinstance(data2, dict) else data2
					if isinstance(raw2, list):
						for item in raw2:
							if isinstance(item, dict):
								h = item.get('updateHash') or item.get('hash') or ''
								acc_val = item.get('accuracy', 0)
								if isinstance(acc_val, (float, int)):
									if acc_val <= 1:
										acc = int(round(acc_val * 10000))
									elif acc_val <= 100:
										acc = int(round(acc_val * 100))
									else:
										acc = int(round(acc_val))
								else:
									acc = 0
								hosp = item.get('hospitalId') or item.get('hospital') or ''
								ts = item.get('timestamp') or item.get('time') or ''
								normalized.append((h, acc, hosp, ts))
						logs = normalized
			except Exception:
				pass
	except Exception:
		# Fallback to Ethereum connector only if explicitly configured
		try:
			from blockchain.blockchain_connector import BlockchainLogger
			rpc = os.environ.get('RPC_URL')
			addr = os.environ.get('CONTRACT_ADDRESS')
			abi = os.environ.get('CONTRACT_ABI_PATH')
			if rpc and addr and abi:
				bl = BlockchainLogger(rpc, addr, _Path(abi))
				eth_logs = bl.get_update_history()
				normalized = []
				for item in eth_logs:
					h = item.get('hash', '')
					acc_val = item.get('accuracy', 0)
					if isinstance(acc_val, (float, int)):
						if acc_val <= 1:
							acc = int(round(acc_val * 10000))
						elif acc_val <= 100:
							acc = int(round(acc_val * 100))
						else:
							acc = int(round(acc_val))
					else:
						acc = 0
					hosp = item.get('hospital', '')
					ts = item.get('timestamp', '')
					normalized.append((h, acc, hosp, ts))
				logs = normalized
		except Exception:
			logs = []
	return render_template("blockchain.html", logs=logs)

@app.route("/_debug_blockchain")
def _debug_blockchain():
	fabric_history = (os.environ.get('FABRIC_HISTORY_URL', 'http://127.0.0.1:3000/history') or '').strip()
	info = { 'fabric_history': fabric_history, 'logs': [] }
	try:
		resp = requests.get(fabric_history, timeout=5)
		info['status_code'] = resp.status_code
		if resp.ok:
			info['raw'] = resp.json()
		else:
			info['text'] = resp.text
	except Exception as e:
		info['error'] = str(e)
	return info

if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=True)
