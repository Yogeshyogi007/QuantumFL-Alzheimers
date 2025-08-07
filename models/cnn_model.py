import torch
import torch.nn as nn
import torch.nn.functional as F

class AlzheimerCNN(nn.Module):
    """
    Simple CNN for binary classification of 128x128 MRI slices.
    Input: (batch, 1, 128, 128)
    Output: (batch, 2)
    """
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 32, 32)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Test model instantiation
    model = AlzheimerCNN()
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    print("Output shape:", out.shape)  # Should be (2, 2)