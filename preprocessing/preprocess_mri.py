import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import glob

def get_middle_slice(img):
    """Extract the middle axial slice from a 3D or 4D MRI volume."""
    if len(img.shape) == 3:
        z = img.shape[2] // 2
        return img[:, :, z]
    elif len(img.shape) == 4:
        z = img.shape[2] // 2
        return img[:, :, z, 0]  # Take first channel if 4D
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

def find_mri_files(raw_dir):
    """Find all MRI files in Analyze format (.img/.hdr) across all disc directories."""
    raw_path = Path(raw_dir)
    mri_files = []
    
    # Search in all disc directories
    for disc_dir in raw_path.glob('disc*'):
        for subject_dir in disc_dir.glob('OAS1_*_MR1'):
            raw_subdir = subject_dir / 'RAW'
            if raw_subdir.exists():
                # Find .img files (Analyze format)
                img_files = list(raw_subdir.glob('*.img'))
                mri_files.extend(img_files)
    
    return mri_files

def preprocess_and_save(mri_path, label, out_dir, filename):
    """Preprocess MRI: extract middle slice, resize, normalize, save tensor."""
    try:
        # Load Analyze format file
        img = nib.load(str(mri_path)).get_fdata()
        
        # Ensure we have a 2D slice
        if len(img.shape) == 3:
            slice_ = get_middle_slice(img)
        elif len(img.shape) == 4:
            slice_ = get_middle_slice(img)
        elif len(img.shape) == 2:
            slice_ = img
        else:
            print(f"Skipping {mri_path}: unexpected shape {img.shape}")
            return False
            
        # Normalize to [0, 1]
        slice_ = (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_) + 1e-8)
        
        # Convert to tensor and ensure correct shape for interpolation
        slice_tensor = torch.tensor(slice_, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        
        # Resize to 128x128 - ensure we have the right format
        if slice_tensor.dim() == 3:  # (1, H, W)
            slice_tensor = slice_tensor.unsqueeze(0)  # (1, 1, H, W)
        
        slice_tensor = F.interpolate(slice_tensor, size=(128, 128), mode='bilinear', align_corners=False)
        slice_tensor = slice_tensor.squeeze(0)  # Back to (1, 128, 128)
        
        # Save tensor and label
        out_path = out_dir / (filename + '.pt')
        torch.save({'image': slice_tensor, 'label': label}, out_path)
        return True
    except Exception as e:
        print(f"Error processing {mri_path}: {e}")
        return False

def main():
    """Preprocess all MRI scans and save as tensors with labels."""
    parser = argparse.ArgumentParser(description='Preprocess OASIS-1 MRI scans.')
    parser.add_argument('--raw_dir', type=str, default=None, help='Path to raw MRI directory')
    parser.add_argument('--csv', type=str, default=None, help='Path to OASIS-1 CSV file')
    parser.add_argument('--out_dir', type=str, default=None, help='Path to save preprocessed tensors')
    args = parser.parse_args()

    # Absolute paths
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = Path(args.raw_dir) if args.raw_dir else base_dir / 'data' / 'raw'
    out_dir = Path(args.out_dir) if args.out_dir else base_dir / 'data' / 'preprocessed'
    csv_path = Path(args.csv) if args.csv else raw_dir / 'oasis_cross-sectional-5708aa0a98d82080.csv'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load clinical data
    print("Loading clinical data...")
    df = pd.read_csv(csv_path)
    print(f"CSV columns: {list(df.columns)}")
    
    # Try to find the correct column names for ID and CDR
    id_col = None
    cdr_col = None
    
    # Common variations of column names
    for col in df.columns:
        if 'id' in col.lower() or 'subject' in col.lower():
            id_col = col
        if 'cdr' in col.lower():
            cdr_col = col
    
    if id_col is None or cdr_col is None:
        print("Could not find ID or CDR columns. Available columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        return
    
    print(f"Using ID column: {id_col}")
    print(f"Using CDR column: {cdr_col}")
    
    # Create mapping from subject ID to CDR
    subj_to_cdr = dict(zip(df[id_col], df[cdr_col]))
    print(f"Found {len(subj_to_cdr)} subjects with CDR scores")

    # Find all MRI files
    print("Finding MRI files...")
    mri_files = find_mri_files(raw_dir)
    print(f"Found {len(mri_files)} MRI files")

    # Process each MRI file
    processed_count = 0
    for mri_file in tqdm(mri_files, desc="Processing MRI files"):
        # Extract subject ID from file path
        # Path format: disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.img
        subject_id = mri_file.parent.parent.name  # OAS1_0001_MR1
        
        # Try to match with CSV data
        cdr = subj_to_cdr.get(subject_id, None)
        if cdr is None:
            # Try alternative matching patterns
            for key in subj_to_cdr.keys():
                if str(key) in subject_id or subject_id in str(key):
                    cdr = subj_to_cdr[key]
                    break
        
        if cdr is None:
            continue
            
        # Convert CDR to binary label
        try:
            cdr_float = float(cdr)
            label = 0 if cdr_float == 0.0 else 1
        except:
            continue
        
        # Create filename for saving
        filename = f"{subject_id}_{mri_file.stem}"
        
        # Preprocess and save
        if preprocess_and_save(mri_file, label, out_dir, filename):
            processed_count += 1

    print(f"Preprocessing complete. {processed_count} tensors saved to {out_dir}")

if __name__ == "__main__":
    main()