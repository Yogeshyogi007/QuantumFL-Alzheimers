import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from utils.dataset_loader import get_loader
from models.cnn_model import AlzheimerCNN

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data_dir, args):
        self.model = model
        self.data_dir = data_dir
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = get_loader(data_dir, batch_size=args.batch_size)
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
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}
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

def run_fl_client(model, data_dir, args):
    client = FlowerClient(model, data_dir, args)
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
    model = AlzheimerCNN()
    run_fl_client(model, data_dir, args)