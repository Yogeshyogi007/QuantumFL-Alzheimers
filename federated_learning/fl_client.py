import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import os

# Ensure project root on path for `utils` and `models` imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.dataset_loader import get_loader
from models.cnn_model import AlzheimerCNN
try:
    from models.true_quantum_model import TrueQuantumHybridModel
except Exception:
    TrueQuantumHybridModel = None
import hashlib
from typing import Optional

try:
    from blockchain.blockchain_connector import BlockchainLogger
except Exception:
    BlockchainLogger = None

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data_dir, args, blockchain: Optional[BlockchainLogger] = None, hospital_id: str = "HOSP"):
        self.model = model
        self.data_dir = data_dir
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = get_loader(data_dir, batch_size=args.batch_size)
        self.blockchain = blockchain
        self.hospital_id = hospital_id
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.args.epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        # After local training, compute hash of weights and optionally log to blockchain
        flat = []
        for _, val in self.model.state_dict().items():
            flat.append(val.detach().cpu().numpy().tobytes())
        digest = hashlib.sha256(b''.join(flat)).hexdigest()
        metrics = {}
        if self.blockchain is not None:
            try:
                acc_bp = int(10000)  # placeholder; accuracy is provided in evaluate
                tx = self.blockchain.record_update(digest, acc_bp, self.hospital_id)
                metrics["blockchain_tx"] = tx
            except Exception as e:
                metrics["blockchain_error"] = str(e)
        return self.get_parameters(config={}), len(self.train_loader.dataset), metrics
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        return float(accuracy), len(self.train_loader.dataset), {"accuracy": float(accuracy)}

def run_fl_client(model, data_dir, args, blockchain: Optional[BlockchainLogger] = None, hospital_id: str = "HOSP"):
    client = FlowerClient(model, data_dir, args, blockchain=blockchain, hospital_id=hospital_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Federated Learning Client (Flower)')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to preprocessed data')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / 'data' / 'preprocessed'
    model = TrueQuantumHybridModel() if TrueQuantumHybridModel is not None and getattr(args, 'quantum', False) else AlzheimerCNN()
    # Optional blockchain wiring via env vars
    bc = None
    if BlockchainLogger is not None:
        import os
        rpc = os.environ.get('RPC_URL')
        addr = os.environ.get('CONTRACT_ADDRESS')
        abi = os.environ.get('CONTRACT_ABI_PATH')
        pk = os.environ.get('PRIVATE_KEY')
        if rpc and addr and abi and pk:
            try:
                bc = BlockchainLogger(rpc, addr, Path(abi), pk)
            except Exception:
                bc = None
    run_fl_client(model, data_dir, args, blockchain=bc, hospital_id=os.environ.get('HOSPITAL_ID', 'HOSP'))