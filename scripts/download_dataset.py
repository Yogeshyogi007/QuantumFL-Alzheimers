#!/usr/bin/env python3
"""
OASIS Dataset Download Script
=============================

This script helps users download and set up the OASIS dataset for the QuantumFL-Alzheimers project.

The OASIS dataset is not publicly available and requires registration.
This script provides instructions and validation for the dataset setup.
"""

import os
import sys
import argparse
import requests
import zipfile
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

class OASISDatasetDownloader:
    """Helper class for OASIS dataset download and setup."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.oasis_url = "https://www.oasis-brains.org/"
        self.expected_discs = [f"disc{i}" for i in range(1, 13)]
        
    def create_directory_structure(self) -> None:
        """Create the necessary directory structure."""
        print("📁 Creating directory structure...")
        
        # Create main directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "preprocessed").mkdir(exist_ok=True)
        
        # Create disc directories
        for disc in self.expected_discs:
            (self.data_dir / disc).mkdir(exist_ok=True)
            
        print(f"✅ Directory structure created at {self.data_dir}")
        
    def check_dataset_status(self) -> Dict[str, bool]:
        """Check which parts of the dataset are present."""
        status = {}
        
        for disc in self.expected_discs:
            disc_path = self.data_dir / disc
            status[disc] = disc_path.exists() and any(disc_path.iterdir())
            
        return status
        
    def validate_dataset(self) -> bool:
        """Validate that the dataset is properly set up."""
        print("🔍 Validating dataset...")
        
        status = self.check_dataset_status()
        missing_discs = [disc for disc, present in status.items() if not present]
        
        if missing_discs:
            print(f"❌ Missing dataset discs: {', '.join(missing_discs)}")
            return False
            
        print("✅ Dataset validation passed!")
        return True
        
    def print_download_instructions(self) -> None:
        """Print detailed download instructions."""
        print("\n" + "="*80)
        print("📥 OASIS DATASET DOWNLOAD INSTRUCTIONS")
        print("="*80)
        
        print("\n1. REGISTER FOR ACCESS:")
        print(f"   - Visit: {self.oasis_url}")
        print("   - Click 'Request Access'")
        print("   - Fill out the registration form")
        print("   - Wait for approval (24-48 hours)")
        
        print("\n2. DOWNLOAD THE DATASET:")
        print("   - Log in to your OASIS account")
        print("   - Navigate to 'OASIS-1 Cross-Sectional Data'")
        print("   - Download all 12 disc files (disc1.zip through disc12.zip)")
        print("   - Total size: ~80GB")
        
        print("\n3. EXTRACT FILES:")
        print("   - Extract each disc*.zip file")
        print("   - Place contents in the corresponding disc* folders")
        print(f"   - Example: disc1.zip → {self.data_dir}/disc1/")
        
        print("\n4. VERIFY STRUCTURE:")
        print("   Expected structure:")
        for disc in self.expected_discs:
            print(f"   {self.data_dir}/{disc}/")
            print(f"     ├── *.nii.gz (MRI files)")
            print(f"     ├── *.csv (metadata)")
            print(f"     └── ...")
            
        print("\n5. RUN VALIDATION:")
        print("   python scripts/download_dataset.py --validate")
        
        print("\n" + "="*80)
        
    def estimate_download_time(self) -> str:
        """Estimate download time based on file sizes."""
        # OASIS-1 dataset is approximately 80GB
        total_size_gb = 80
        avg_speed_mbps = 50  # Average internet speed
        
        # Calculate time in hours
        time_hours = (total_size_gb * 8 * 1024) / (avg_speed_mbps * 3600)
        
        if time_hours < 1:
            return f"{int(time_hours * 60)} minutes"
        elif time_hours < 24:
            return f"{time_hours:.1f} hours"
        else:
            days = time_hours / 24
            return f"{days:.1f} days"
            
    def create_sample_data(self) -> None:
        """Create sample data structure for testing."""
        print("🧪 Creating sample data structure for testing...")
        
        # Create sample MRI file
        sample_mri = self.data_dir / "disc1" / "sample_mri.nii.gz"
        sample_mri.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy file
        with open(sample_mri, 'w') as f:
            f.write("# Sample MRI file for testing\n")
            
        # Create sample metadata
        sample_csv = self.data_dir / "disc1" / "sample_metadata.csv"
        with open(sample_csv, 'w') as f:
            f.write("SubjectID,Age,CDR,Diagnosis\n")
            f.write("OAS1_0001,25,0,Non-demented\n")
            f.write("OAS1_0002,75,0.5,Very Mild Dementia\n")
            
        print("✅ Sample data created for testing")
        
    def run_validation(self) -> None:
        """Run comprehensive dataset validation."""
        print("🔍 Running dataset validation...")
        
        # Check directory structure
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return
            
        # Check disc directories
        status = self.check_dataset_status()
        present_discs = [disc for disc, present in status.items() if present]
        missing_discs = [disc for disc, present in status.items() if not present]
        
        print(f"\n📊 Dataset Status:")
        print(f"   Present discs: {len(present_discs)}/12")
        print(f"   Missing discs: {len(missing_discs)}/12")
        
        if present_discs:
            print(f"   ✅ Found: {', '.join(present_discs)}")
            
        if missing_discs:
            print(f"   ❌ Missing: {', '.join(missing_discs)}")
            
        # Check file counts
        total_files = 0
        for disc in present_discs:
            disc_path = self.data_dir / disc
            files = list(disc_path.glob("*"))
            total_files += len(files)
            print(f"   📁 {disc}: {len(files)} files")
            
        print(f"\n📈 Total files found: {total_files}")
        
        if len(present_discs) == 12 and total_files > 1000:
            print("✅ Dataset appears to be complete!")
        else:
            print("⚠️  Dataset appears to be incomplete. Please download all discs.")
            
    def create_config_template(self) -> None:
        """Create a configuration template for the dataset."""
        config_content = f"""# Dataset Configuration
# Generated by download_dataset.py

dataset:
  path: "{self.data_dir}"
  name: "OASIS-1"
  description: "Open Access Series of Imaging Studies"
  
  # Dataset statistics
  total_subjects: 416
  age_range: "18-96"
  modalities: ["T1-weighted MRI"]
  
  # File patterns
  mri_pattern: "*.nii.gz"
  metadata_pattern: "*.csv"
  
  # Preprocessing settings
  preprocessing:
    target_size: [256, 256, 256]
    normalization: "z-score"
    augmentation: true
    
  # Validation
  expected_discs: {self.expected_discs}
  min_files_per_disc: 50
"""
        
        config_file = Path("configs/dataset_config.yaml")
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            f.write(config_content)
            
        print(f"✅ Configuration template created: {config_file}")

def main():
    parser = argparse.ArgumentParser(
        description="OASIS Dataset Download Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_dataset.py --setup
  python scripts/download_dataset.py --validate
  python scripts/download_dataset.py --create-sample
        """
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up directory structure and show download instructions"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample data for testing"
    )
    
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory for dataset storage (default: data/raw)"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = OASISDatasetDownloader(args.data_dir)
    
    if args.setup:
        print("🚀 Setting up OASIS dataset...")
        downloader.create_directory_structure()
        downloader.print_download_instructions()
        downloader.create_config_template()
        
        print(f"\n⏱️  Estimated download time: {downloader.estimate_download_time()}")
        print("\n📋 Next steps:")
        print("1. Follow the download instructions above")
        print("2. Extract files to the created directories")
        print("3. Run: python scripts/download_dataset.py --validate")
        
    elif args.validate:
        downloader.run_validation()
        
    elif args.create_sample:
        downloader.create_sample_data()
        print("\n🧪 Sample data created for testing purposes.")
        print("Note: This is not the actual OASIS dataset.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
