import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class MRIDataset(Dataset):
    """
    PyTorch Dataset for preprocessed MRI .pt files.
    Each .pt file contains a dict with 'image' and 'label'.
    """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob('*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])
        return sample['image'], sample['label']

def get_loader(data_dir, batch_size=16, shuffle=True, num_workers=0):
    """Utility to get DataLoader for MRIDataset."""
    dataset = MRIDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    # Test loading
    loader = get_loader(Path(__file__).resolve().parent.parent / 'data' / 'preprocessed', batch_size=2)
    for images, labels in loader:
        print("Batch images shape:", images.shape)
        print("Batch labels:", labels)
        break