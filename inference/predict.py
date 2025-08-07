import argparse
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import sys
from PIL import Image
import cv2

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from models.cnn_model import AlzheimerCNN
from models.quantum_model import QuantumHybridModel

def preprocess_mri(mri_path):
    """Preprocess MRI from various file formats."""
    mri_path = Path(mri_path)
    
    if mri_path.suffix.lower() in ['.img', '.hdr']:
        # Handle Analyze format files
        try:
            img = nib.load(str(mri_path)).get_fdata()
            
            # Handle different dimensions
            if len(img.shape) == 3:
                z = img.shape[2] // 2
                slice_ = img[:, :, z]
            elif len(img.shape) == 4:
                z = img.shape[2] // 2
                slice_ = img[:, :, z, 0]
            elif len(img.shape) == 2:
                slice_ = img
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
                
        except Exception as e:
            print(f"Error loading Analyze file {mri_path}: {e}")
            raise
            
    elif mri_path.suffix.lower() in ['.gif', '.jpg', '.jpeg', '.png']:
        # Handle image files
        try:
            img = Image.open(mri_path).convert('L')  # Convert to grayscale
            slice_ = np.array(img)
        except Exception as e:
            print(f"Error loading image file {mri_path}: {e}")
            raise
            
    else:
        raise ValueError(f"Unsupported file format: {mri_path.suffix}")
    
    # Normalize to [0, 1]
    slice_ = (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_) + 1e-8)
    
    # Convert to tensor and resize
    slice_tensor = torch.tensor(slice_, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    slice_tensor = F.interpolate(slice_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    
    return slice_tensor

def load_model(model_path):
    """Load the appropriate model based on the saved state dict."""
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Check if it's a quantum model by looking for quantum layer parameters
    is_quantum = any('quantum_layer' in key for key in state_dict.keys())
    
    if is_quantum:
        print("Loading Quantum-Enhanced Model...")
        model = QuantumHybridModel()
    else:
        print("Loading Classical CNN Model...")
        model = AlzheimerCNN()
    
    model.load_state_dict(state_dict)
    return model

def main():
    parser = argparse.ArgumentParser(description='Predict Alzheimer\'s probability from MRI scan.')
    parser.add_argument('--input', type=str, required=True, help='Path to MRI file (.img, .gif, .jpg, .png)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent
    
    # Use best model by default (99.81% accuracy)
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = base_dir / 'models' / 'best_alzheimers_cnn.pth'
        print("Using Best Model (99.81% accuracy) by default")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    print(f"Processing file: {args.input}")
    print(f"Using device: {device}")
    
    try:
        model = load_model(model_path)
        model.to(device)
        model.eval()
        
        x = preprocess_mri(args.input).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        
        print(f"Alzheimer's probability: {prob:.4f}")
        if prob > 0.5:
            print("Prediction: HIGH RISK of Alzheimer's")
        else:
            print("Prediction: LOW RISK of Alzheimer's")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("\nAvailable file types in the dataset:")
        print("- RAW/.img files: Actual MRI data (recommended)")
        print("- PROCESSED/.img files: Preprocessed MRI data")
        print("- .gif files: Visualizations (not recommended for prediction)")

if __name__ == "__main__":
    main()