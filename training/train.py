import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.dataset_loader import get_loader
from models.cnn_model import AlzheimerCNN

# Optional imports for federated and quantum
try:
    import flwr as fl
except ImportError:
    fl = None
try:
    from models.quantum_model import QuantumHybridModel
except ImportError:
    QuantumHybridModel = None


def train_local(model, train_loader, device, epochs=10, lr=1e-3):
    """Train model locally."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_path = Path(__file__).resolve().parent.parent / 'models' / 'alzheimers_cnn.pth'
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
    print(f"Best model saved to {best_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Alzheimer CNN or Quantum model.')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to preprocessed data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--federated', action='store_true', help='Enable federated learning')
    parser.add_argument('--quantum', action='store_true', help='Enable quantum hybrid model')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / 'data' / 'preprocessed'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.quantum:
        if QuantumHybridModel is None:
            raise ImportError('PennyLane or Qiskit not installed or quantum_model.py missing.')
        model = QuantumHybridModel()
    else:
        model = AlzheimerCNN()

    train_loader = get_loader(data_dir, batch_size=args.batch_size)
    print(f"Training on {len(train_loader.dataset)} samples")

    if args.federated:
        if fl is None:
            raise ImportError('Flower not installed. Install with `pip install flwr`.')
        from federated_learning.fl_client import run_fl_client
        run_fl_client(model, data_dir, args)
    else:
        train_local(model, train_loader, device, epochs=args.epochs)

if __name__ == "__main__":
    main()