import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
import time

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.dataset_loader import get_loader
from models.cnn_model import AlzheimerCNN

def train_best_classical(epochs=50, batch_size=32, learning_rate=0.001, weight_decay=1e-5):
    """Train the original simple CNN to achieve best accuracy on OASIS dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'preprocessed'
    full_dataset = get_loader(data_dir, batch_size=batch_size, shuffle=True).dataset
    
    # Split into train and validation (90-10 split for maximum training data)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validating on {len(val_dataset)} samples")
    
    # Initialize the original simple CNN
    model = AlzheimerCNN()
    model.to(device)
    
    # Simple loss function without label smoothing
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with minimal weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Early stopping
    best_val_accuracy = 0.0
    patience_counter = 0
    max_patience = 10
    
    print("\n" + "="*60)
    print("🚀 BEST CLASSICAL CNN TRAINING (ORIGINAL SIMPLE MODEL)")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            accuracy = 100 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        train_accuracy = correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # Save best model
            model_path = Path(__file__).resolve().parent.parent / 'models' / 'best_alzheimers_cnn.pth'
            torch.save(model.state_dict(), model_path)
            print(f"🏆 New best model saved! Val Accuracy: {best_val_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    print(f"\n🎯 BEST CLASSICAL TRAINING COMPLETED!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: models/best_alzheimers_cnn.pth")
    
    return model, best_val_accuracy

def evaluate_best_model(model_path, batch_size=32):
    """Evaluate the best model on the full dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data_dir = Path(__file__).resolve().parent.parent / 'data' / 'preprocessed'
    test_loader = get_loader(data_dir, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = AlzheimerCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    y_true, y_pred = [], []
    
    print(f"\n🔬 EVALUATING BEST MODEL")
    print("="*50)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    
    print(f"\n=== BEST MODEL EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives: {cm[1,1]}")
    
    return accuracy

def test_best_inference():
    """Test best model inference on different samples."""
    print(f"\n🧠 TESTING BEST MODEL INFERENCE")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test files
    test_files = [
        "archive/Data/Mild Dementia/OAS1_0028_MR1_mpr-1_100.jpg",
        "archive/Data/Non Demented/OAS1_0001_MR1_mpr-1_100.jpg"
    ]
    
    try:
        # Load model
        model = AlzheimerCNN()
        model_path = Path(__file__).resolve().parent.parent / 'models' / 'best_alzheimers_cnn.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        for test_file in test_files:
            try:
                # Preprocess and predict
                from inference.predict import preprocess_mri
                x = preprocess_mri(test_file).to(device)
                
                with torch.no_grad():
                    output = model(x)
                    prob = torch.softmax(output, dim=1)[0, 1].item()
                
                print(f"\n📁 File: {test_file}")
                print(f"🔮 Prediction: {prob:.4f}")
                print(f"🎯 Risk Level: {'HIGH RISK' if prob > 0.5 else 'LOW RISK'}")
                print(f"💪 Confidence: {max(prob, 1-prob):.2%}")
                
            except Exception as e:
                print(f"❌ Error processing {test_file}: {e}")
        
        print(f"\n✅ Best model inference successful!")
        
    except Exception as e:
        print(f"❌ Error in best inference: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train best classical CNN')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--test_inference', action='store_true', help='Test best inference')
    
    args = parser.parse_args()
    
    if args.evaluate:
        model_path = Path(__file__).resolve().parent.parent / 'models' / 'best_alzheimers_cnn.pth'
        if model_path.exists():
            evaluate_best_model(model_path)
        else:
            print("No best model found. Train first with --epochs")
    
    elif args.test_inference:
        test_best_inference()
    
    else:
        print("🚀 Starting Best Classical Training...")
        model, best_acc = train_best_classical(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Evaluate after training
        model_path = Path(__file__).resolve().parent.parent / 'models' / 'best_alzheimers_cnn.pth'
        evaluate_best_model(model_path)
        
        # Test inference
        test_best_inference() 