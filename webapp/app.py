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

# Ensure project root is on sys.path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

# Reuse preprocessing from existing inference code
from inference.predict import preprocess_mri
from models.cnn_model import AlzheimerCNN

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET_KEY", "secret-key")

BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_alzheimers_cnn.pth"

# Lazy-loaded model
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
	global _model
	if _model is None:
		if not BEST_MODEL_PATH.exists():
			raise FileNotFoundError(f"Best model not found at {BEST_MODEL_PATH}")
		model = AlzheimerCNN()
		state = torch.load(BEST_MODEL_PATH, map_location=_device)
		model.load_state_dict(state)
		model.to(_device)
		model.eval()
		_model = model
	return _model

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
			with torch.no_grad():
				logits = model(x)
				prob = torch.softmax(logits, dim=1)[0, 1].item()
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

if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=False)
