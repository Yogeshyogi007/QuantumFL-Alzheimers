import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import sys

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.dataset_loader import get_loader
from models.cnn_model import AlzheimerCNN

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def plot_and_save_metrics(y_true, y_pred, y_prob, out_dir):
    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_true, y_pred, target_names=['Control', 'Alzheimer'])
    # Save confusion matrix
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig(out_dir / 'confusion_matrix.png')
    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(out_dir / 'roc_curve.png')
    # Save classification report
    with open(out_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Alzheimer model.')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to preprocessed test data')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save metrics')
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / 'data' / 'preprocessed'
    model_path = Path(args.model_path) if args.model_path else base_dir / 'models' / 'alzheimers_cnn.pth'
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / 'training'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlzheimerCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    loader = get_loader(data_dir, batch_size=16, shuffle=False)
    y_true, y_pred, y_prob = evaluate(model, loader, device)
    plot_and_save_metrics(y_true, y_pred, y_prob, out_dir)
    print('Evaluation complete. Metrics saved to', out_dir)

if __name__ == "__main__":
    main()