import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuantumInspiredLayer(nn.Module):
    """Optimized quantum-inspired layer with better regularization and stability."""
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Smaller initialization for better training stability
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Phase parameters with smaller initialization
        self.phases = nn.Parameter(torch.randn(output_dim) * 0.5)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # Apply quantum-inspired transformation with better stability
        # 1. Linear transformation
        linear_out = F.linear(x, self.weights, self.bias)
        
        # 2. Apply phase rotation (quantum-inspired) with clipping
        phase_rotation = torch.cos(torch.clamp(self.phases, -math.pi, math.pi)).unsqueeze(0) * linear_out
        
        # 3. Apply quantum-inspired activation with better stability
        quantum_out = torch.tanh(phase_rotation * 0.5) * torch.sigmoid(linear_out * 0.5)
        
        # 4. Layer normalization and dropout
        quantum_out = self.layer_norm(quantum_out)
        quantum_out = self.dropout(quantum_out)
        
        return quantum_out

class OptimizedQuantumHybridModel(nn.Module):
    """Optimized hybrid CNN + Quantum-inspired classifier with better architecture."""
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Enhanced CNN feature extractor with more layers and better regularization
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Classical fully connected layers with better regularization
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # Quantum-inspired layers with smaller dimensions and better regularization
        self.quantum_layer1 = QuantumInspiredLayer(64, 32, dropout_rate=0.2)
        self.quantum_layer2 = QuantumInspiredLayer(32, 16, dropout_rate=0.2)
        self.quantum_layer3 = QuantumInspiredLayer(16, 2, dropout_rate=0.1)
        
        # Additional regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classical processing with better regularization
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Quantum-inspired processing
        x = self.quantum_layer1(x)
        x = self.quantum_layer2(x)
        x = self.quantum_layer3(x)
        
        return x

class QuantumHybridModel(nn.Module):
    """Backward compatibility wrapper for the optimized quantum model."""
    def __init__(self):
        super().__init__()
        self.model = OptimizedQuantumHybridModel()
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = OptimizedQuantumHybridModel()
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print("Optimized quantum-inspired hybrid output shape:", out.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))