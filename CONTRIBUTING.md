# Contributing to QuantumFL-Alzheimers

Thank you for your interest in contributing to the QuantumFL-Alzheimers project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/QuantumFL-Alzheimers.git
   cd QuantumFL-Alzheimers
   ```

### 2. Set Up Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions small and focused

### Example Code Structure

```python
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class QuantumCNN(nn.Module):
    """Quantum-inspired Convolutional Neural Network for MRI analysis.
    
    This model combines classical CNN layers with quantum-inspired
    feature extraction for improved Alzheimer's disease detection.
    
    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        quantum_layers: Number of quantum layers
    """
    
    def __init__(self, input_channels: int, num_classes: int, quantum_layers: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.quantum_layers = quantum_layers
        
        # Initialize layers
        self._build_layers()
        
    def _build_layers(self) -> None:
        """Build the neural network layers."""
        # Implementation here
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Implementation here
        pass
```

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use pytest for testing
- Run tests before submitting PR

### Example Test

```python
import pytest
import torch
from models.quantum_cnn import QuantumCNN


class TestQuantumCNN:
    """Test cases for QuantumCNN model."""
    
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        model = QuantumCNN(input_channels=1, num_classes=2, quantum_layers=3)
        assert model.input_channels == 1
        assert model.num_classes == 2
        assert model.quantum_layers == 3
        
    def test_forward_pass(self):
        """Test forward pass with sample input."""
        model = QuantumCNN(input_channels=1, num_classes=2)
        x = torch.randn(2, 1, 256, 256)  # Batch of 2, 1 channel, 256x256
        output = model(x)
        
        assert output.shape == (2, 2)
        assert torch.allclose(output.sum(dim=1), torch.ones(2), atol=1e-6)
```

## 🏗️ Project Structure

### Adding New Models

1. Create model file in `models/`:
   ```python
   # models/new_model.py
   import torch.nn as nn
   
   class NewModel(nn.Module):
       def __init__(self, **kwargs):
           super().__init__()
           # Implementation
           
       def forward(self, x):
           # Implementation
           pass
   ```

2. Add to model registry in `models/__init__.py`:
   ```python
   from .new_model import NewModel
   
   __all__ = ['NewModel']
   ```

3. Add configuration in `configs/`:
   ```yaml
   # configs/new_model_config.yaml
   model:
     name: "new_model"
     parameters:
       # Model-specific parameters
   ```

### Adding New Preprocessing Steps

1. Create preprocessing script in `preprocessing/`:
   ```python
   # preprocessing/new_preprocessing.py
   import numpy as np
   
   def new_preprocessing_function(data: np.ndarray) -> np.ndarray:
       """Apply new preprocessing step.
       
       Args:
           data: Input data
           
       Returns:
           Preprocessed data
       """
       # Implementation
       return processed_data
   ```

2. Add to preprocessing pipeline in `preprocessing/pipeline.py`

### Adding New Federated Learning Algorithms

1. Create algorithm file in `federated_learning/algorithms/`:
   ```python
   # federated_learning/algorithms/new_algorithm.py
   class NewFederatedAlgorithm:
       def __init__(self, **kwargs):
           # Implementation
           pass
           
       def aggregate(self, client_models):
           # Implementation
           pass
   ```

2. Register in `federated_learning/__init__.py`

## 🔧 Development Tools

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install
```

### Code Formatting

Use black for code formatting:

```bash
black .
```

### Linting

Use flake8 for linting:

```bash
flake8 .
```

### Type Checking

Use mypy for type checking:

```bash
mypy .
```

## 📝 Documentation

### Docstring Standards

Use Google-style docstrings:

```python
def process_mri_data(mri_file: str, output_dir: str) -> bool:
    """Process MRI data and save results.
    
    Args:
        mri_file: Path to input MRI file
        output_dir: Directory to save processed results
        
    Returns:
        True if processing successful, False otherwise
        
    Raises:
        FileNotFoundError: If mri_file doesn't exist
        ValueError: If output_dir is invalid
    """
    # Implementation
    pass
```

### README Updates

When adding new features:

1. Update the main README.md
2. Add usage examples
3. Update configuration documentation
4. Add any new dependencies to requirements.txt

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=.

# Run with verbose output
pytest -v
```

### Writing Tests

- Test both success and failure cases
- Use fixtures for common test data
- Mock external dependencies
- Test edge cases and boundary conditions

## 🚀 Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Checklist

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Update documentation

## 📞 Getting Help

### Communication Channels

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Email**: Contact maintainers directly for sensitive issues

### Issue Templates

When creating issues, use the appropriate template:

- Bug report
- Feature request
- Documentation request
- Question

## 🙏 Recognition

Contributors will be recognized in:

- README.md contributors section
- Release notes
- Project documentation

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to QuantumFL-Alzheimers! Your contributions help advance research in quantum machine learning and medical AI.
