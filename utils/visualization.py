import matplotlib.pyplot as plt
import numpy as np

def plot_training_curve(train_acc, val_acc, out_path):
    """Plot and save training/validation accuracy curves."""
    plt.figure()
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def plot_mri_slice(slice_, out_path=None):
    """Plot and optionally save a single MRI slice."""
    plt.figure()
    plt.imshow(slice_, cmap='gray')
    plt.axis('off')
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()

def plot_metrics(cm, roc_auc, out_dir):
    """Plot confusion matrix and ROC AUC curve."""
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig(out_dir / 'confusion_matrix.png')
    plt.close()
    # ROC curve plotting would require fpr, tpr arrays

if __name__ == "__main__":
    # Example usage
    plot_training_curve([0.7, 0.8, 0.9], [0.65, 0.75, 0.85], 'train_curve.png')