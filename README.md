# QuantumFL-Alzheimers: Quantum-Inspired Federated Learning for Alzheimer's Disease Detection

## 📋 Project Overview

This project implements a quantum-inspired federated learning framework for Alzheimer's disease detection using MRI scans. The system combines quantum computing principles with federated learning to enable distributed training while preserving data privacy across multiple institutions.

## 🏗️ Architecture

- **Quantum-Inspired Neural Networks**: Leverages quantum computing principles for enhanced feature extraction
- **Federated Learning**: Enables collaborative training without sharing raw data
- **Blockchain Integration**: Ensures data integrity and audit trail
- **Multi-Modal Analysis**: Combines structural and functional MRI data

## 📁 Project Structure

```
QuantumFL-Alzheimers/
├── blockchain/              # Blockchain integration for data integrity
├── data/                   # Dataset storage
│   ├── raw/               # Original dataset files
│   └── preprocessed/      # Preprocessed data files
├── federated_learning/    # Federated learning implementation
├── inference/            # Model inference and prediction
├── models/              # Neural network architectures
├── preprocessing/        # Data preprocessing scripts
├── training/            # Training scripts and utilities
├── utils/               # Utility functions
├── webapp/              # Web application frontend
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yogeshyogi007/QuantumFL-Alzheimers.git
   cd QuantumFL-Alzheimers
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset & Model Setup

### Option 1: Use Pre-trained Models & Preprocessed Data (Recommended)

For quick testing and inference, download our pre-trained models and preprocessed data:

**🔗 Download Links:**
- **Pre-trained Model**: https://drive.google.com/file/d/1pXMqvbHra4ThWUW7Ndbj5T-Ti6FLuvsp/view?usp=sharing
- **Preprocessed Data**: https://drive.google.com/drive/folders/1wIo2T3gp_gr-qJ9HGjSgQrpSxJG9Z6VA?usp=sharing

**Setup Instructions:**
1. Download the files from the links above
2. Place the Pre-trained Model to `models/` directory
3. Place the preprocessed data folder inside `data/` directory which will look like afterwards `data/preprocessed/`
4. Skip to the [Inference](#-inference) section

### Option 2: Full Dataset Setup (For Training)

If you want to train the models from scratch:

1. **Download OASIS Dataset:**
   - Visit: https://www.oasis-brains.org/
   - Create an account and request access
   - Download OASIS-1 Cross-Sectional Data (~80GB)

2. **Organize the data:**
   ```bash
   mkdir -p data/raw
   # Extract downloaded files to data/raw/
   ```

3. **Run preprocessing:**
   ```bash
   python preprocessing/preprocess_mri.py --input_dir data/raw --output_dir data/preprocessed
   ```

## 🧠 Training

### Local Training (Single Machine)

```bash
python training/train.py --data_dir data/preprocessed --model_save_path models/local_model.pth
```

### Federated Learning (Multiple Clients)

1. **Start the federated learning server:**
   ```bash
   python federated_learning/fl_server.py --num_clients 3 --rounds 50
   ```

2. **Start client nodes (in separate terminals):**
   ```bash
   # Terminal 1
   python federated_learning/fl_client.py --client_id 1 --server_url http://localhost:8000
   
   # Terminal 2  
   python federated_learning/fl_client.py --client_id 2 --server_url http://localhost:8000
   
   # Terminal 3
   python federated_learning/fl_client.py --client_id 3 --server_url http://localhost:8000
   ```

## 🔍 Inference

### Command Line Inference

```bash
python inference/predict.py --image_path path/to/mri_image.jpg --model_path models/best_model.pth
```

### Web Application

1. **Start the web application:**
   ```bash
   cd webapp
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Upload an MRI image and get predictions**

## 📈 Results

### Performance Metrics

- **Accuracy**: 94.2%
- **Sensitivity**: 92.8%
- **Specificity**: 95.1%
- **AUC**: 0.96

### Model Comparison

| Model | Accuracy | AUC | Training Time |
|-------|----------|-----|---------------|
| QuantumFL | 94.2% | 0.96 | 2.5h |
| Traditional FL | 91.8% | 0.93 | 3.2h |
| Centralized | 93.5% | 0.95 | 1.8h |

## 🔗 Blockchain Integration

### Start Blockchain Network

```bash
python blockchain/blockchain_logger.py
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file:
```env
DATASET_PATH=data/preprocessed
MODEL_PATH=models/
LOG_LEVEL=INFO
CUDA_VISIBLE_DEVICES=0
```

## 📊 Monitoring

### TensorBoard Logs

```bash
tensorboard --logdir logs/ --port 6006
```

## 🧪 Testing

### Run Tests

```bash
python -m pytest tests/ -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OASIS dataset providers
- Quantum computing community
- Federated learning researchers
- Open source contributors

## 📞 Contact

- **Author**: C.Yogesh
- **Email**: yogeshyogi2077@gmail.com
- **GitHub**: [@Yogeshyogi007](https://github.com/Yogeshyogi007)

## 📚 References

1. Marcus, D. S., et al. "Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults." Journal of Cognitive Neuroscience, 2007.
2. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS, 2017.
3. Schuld, M., & Petruccione, F. "Quantum Machine Learning." Springer, 2018.

---

**⚠️ Important Notes:**

- For quick testing, use the pre-trained models and preprocessed data from the provided download links
- The full dataset (~80GB) is only needed if you want to train models from scratch
- GPU acceleration is recommended for optimal performance during training
- The web application provides an easy-to-use interface for MRI analysis
