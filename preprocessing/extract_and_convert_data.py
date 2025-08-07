import os
import tarfile
import pandas as pd
from pathlib import Path
import glob

def extract_tar_files(raw_dir):
    """Extract all .tar.gz files in the raw directory."""
    raw_path = Path(raw_dir)
    tar_files = list(raw_path.glob('*.tar.gz'))
    
    print(f"Found {len(tar_files)} tar.gz files to extract...")
    
    for tar_file in tar_files:
        print(f"Extracting {tar_file.name}...")
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=raw_path)
            print(f"Successfully extracted {tar_file.name}")
        except Exception as e:
            print(f"Error extracting {tar_file.name}: {e}")

def convert_excel_to_csv(raw_dir):
    """Convert Excel files to CSV format."""
    raw_path = Path(raw_dir)
    excel_files = list(raw_path.glob('*.xlsx'))
    
    print(f"Found {len(excel_files)} Excel files to convert...")
    
    for excel_file in excel_files:
        print(f"Converting {excel_file.name} to CSV...")
        try:
            # Read Excel file
            df = pd.read_excel(excel_file)
            
            # Create CSV filename
            csv_filename = excel_file.stem + '.csv'
            csv_path = raw_path / csv_filename
            
            # Save as CSV
            df.to_csv(csv_path, index=False)
            print(f"Successfully converted {excel_file.name} to {csv_filename}")
            
        except Exception as e:
            print(f"Error converting {excel_file.name}: {e}")

def find_mri_files(raw_dir):
    """Find all .nii.gz files after extraction."""
    raw_path = Path(raw_dir)
    mri_files = list(raw_path.rglob('*.nii.gz'))
    
    print(f"Found {len(mri_files)} MRI files:")
    for mri_file in mri_files[:5]:  # Show first 5
        print(f"  {mri_file.relative_to(raw_path)}")
    if len(mri_files) > 5:
        print(f"  ... and {len(mri_files) - 5} more files")
    
    return mri_files

def main():
    """Main function to extract and convert OASIS-1 data."""
    # Get the raw directory path
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    
    print("=== OASIS-1 Data Preparation ===")
    print(f"Working directory: {raw_dir}")
    
    # Step 1: Extract tar.gz files
    print("\n1. Extracting tar.gz files...")
    extract_tar_files(raw_dir)
    
    # Step 2: Convert Excel to CSV
    print("\n2. Converting Excel files to CSV...")
    convert_excel_to_csv(raw_dir)
    
    # Step 3: Find MRI files
    print("\n3. Scanning for MRI files...")
    mri_files = find_mri_files(raw_dir)
    
    print("\n=== Data Preparation Complete ===")
    print("Next steps:")
    print("1. Check the extracted data structure")
    print("2. Run: python preprocessing/preprocess_mri.py")
    print("3. Make sure the CSV file has 'ID' and 'CDR' columns")

if __name__ == "__main__":
    main() 