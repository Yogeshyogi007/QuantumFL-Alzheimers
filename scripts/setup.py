#!/usr/bin/env python3
"""
QuantumFL-Alzheimers Setup Script
==================================

This script helps users set up the QuantumFL-Alzheimers project environment,
including dependency installation, directory creation, and initial configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict

class ProjectSetup:
    """Setup helper for QuantumFL-Alzheimers project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.required_dirs = [
            "data/raw",
            "data/preprocessed",
            "models/saved_models",
            "logs",
            "checkpoints",
            "results",
            "configs",
            "scripts",
            "tests"
        ]
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"❌ Python {version.major}.{version.minor} detected.")
            print("   This project requires Python 3.8 or higher.")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected.")
        return True
        
    def create_directories(self) -> None:
        """Create necessary project directories."""
        print("📁 Creating project directories...")
        
        for dir_path in self.required_dirs:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
            
    def install_dependencies(self, use_gpu: bool = True) -> bool:
        """Install project dependencies."""
        print("📦 Installing dependencies...")
        
        try:
            # Upgrade pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                             check=True, capture_output=True)
                print("✅ Dependencies installed successfully!")
                return True
            else:
                print("❌ requirements.txt not found!")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
            
    def create_config_files(self) -> None:
        """Create default configuration files."""
        print("⚙️  Creating configuration files...")
        
        # Training config
        training_config = """# Training Configuration
model:
  architecture: "quantum_cnn"
  num_qubits: 8
  depth: 3
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  patience: 10
  device: "cuda"  # or "cpu"

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  augmentation: true

federated:
  num_clients: 5
  rounds: 100
  aggregation: "fedavg"
  communication_rounds: 10

quantum:
  backend: "default.qubit"
  shots: 1000
  optimization_level: 2
"""
        
        config_dir = self.project_root / "configs"
        config_dir.mkdir(exist_ok=True)
        
        with open(config_dir / "training_config.yaml", "w") as f:
            f.write(training_config)
            
        # Environment file
        env_content = """# Environment Configuration
DATASET_PATH=data/raw
MODEL_PATH=models/saved_models
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0
BLOCKCHAIN_NETWORK=testnet
FEDERATED_SERVER_URL=http://localhost:8000
"""
        
        with open(self.project_root / ".env.example", "w") as f:
            f.write(env_content)
            
        print("✅ Configuration files created!")
        
    def create_sample_scripts(self) -> None:
        """Create sample scripts for common tasks."""
        print("📝 Creating sample scripts...")
        
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Quick start script
        quick_start = """#!/usr/bin/env python3
\"\"\"
Quick Start Script for QuantumFL-Alzheimers
===========================================

This script provides a quick way to test the project setup.
\"\"\"

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def main():
    print("🚀 QuantumFL-Alzheimers Quick Start")
    print("=" * 40)
    
    # Check if dataset is available
    data_dir = project_root / "data" / "raw"
    if data_dir.exists() and any(data_dir.iterdir()):
        print("✅ Dataset found!")
    else:
        print("⚠️  Dataset not found. Please run:")
        print("   python scripts/download_dataset.py --setup")
        return
    
    # Test imports
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("❌ PyTorch not available")
        return
        
    try:
        import pennylane as qml
        print("✅ PennyLane available")
    except ImportError:
        print("❌ PennyLane not available")
        return
        
    print("\\n🎉 Setup complete! You can now:")
    print("1. Run training: python training/train_local.py")
    print("2. Start federated learning: python federated_learning/server.py")
    print("3. Make predictions: python inference/predict.py")

if __name__ == "__main__":
    main()
"""
        
        with open(scripts_dir / "quick_start.py", "w") as f:
            f.write(quick_start)
            
        # Make executable
        os.chmod(scripts_dir / "quick_start.py", 0o755)
        
        print("✅ Sample scripts created!")
        
    def run_tests(self) -> bool:
        """Run basic tests to verify setup."""
        print("🧪 Running basic tests...")
        
        try:
            # Test imports
            import torch
            import numpy as np
            import pandas as pd
            
            print("✅ Core dependencies imported successfully")
            
            # Test GPU availability
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA not available - will use CPU")
                
            return True
            
        except ImportError as e:
            print(f"❌ Import test failed: {e}")
            return False
            
    def print_next_steps(self) -> None:
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        
        print("\n📋 Next Steps:")
        print("1. Download the OASIS dataset:")
        print("   python scripts/download_dataset.py --setup")
        
        print("\n2. Test the setup:")
        print("   python scripts/quick_start.py")
        
        print("\n3. Start training:")
        print("   python training/train_local.py")
        
        print("\n4. Run federated learning:")
        print("   python federated_learning/server.py")
        
        print("\n📚 Documentation:")
        print("   - README.md: Complete project documentation")
        print("   - configs/: Configuration files")
        print("   - scripts/: Utility scripts")
        
        print("\n🔧 Configuration:")
        print("   - Edit configs/training_config.yaml for training settings")
        print("   - Copy .env.example to .env and configure environment")
        
        print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description="QuantumFL-Alzheimers Setup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU-specific installations"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests after setup"
    )
    
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize setup
    project_root = Path(args.project_dir).resolve()
    setup = ProjectSetup(project_root)
    
    print("🚀 QuantumFL-Alzheimers Setup")
    print("=" * 40)
    
    # Check Python version
    if not setup.check_python_version():
        sys.exit(1)
        
    # Create directories
    setup.create_directories()
    
    # Install dependencies
    if not setup.install_dependencies(use_gpu=not args.no_gpu):
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
        
    # Create config files
    setup.create_config_files()
    
    # Create sample scripts
    setup.create_sample_scripts()
    
    # Run tests
    if not args.skip_tests:
        if not setup.run_tests():
            print("⚠️  Some tests failed, but setup completed")
            
    # Print next steps
    setup.print_next_steps()

if __name__ == "__main__":
    main()
