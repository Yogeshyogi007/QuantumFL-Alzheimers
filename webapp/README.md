# Alzheimer's MRI Detection Web App

A simple Flask-based web interface to upload MRI images and get Alzheimer's prediction using the best classical CNN model.

## Prerequisites
- Python 3.9+
- All project dependencies installed

## Install
From the project root:
```bash
pip install -r requirements.txt
```

## Run the App
From the project root:
```bash
export FLASK_APP=webapp/app.py  # on Windows: set FLASK_APP=webapp/app.py
python webapp/app.py
```
Open your browser at http://localhost:5000

## Usage
- Upload an MRI file (.img/.hdr) or static image (.gif/.jpg/.png)
- The app reuses the same preprocessing pipeline as CLI inference
- It loads `models/best_alzheimers_cnn.pth` by default

## Notes
- If you want to point to a different model, adjust BEST_MODEL_PATH in `webapp/app.py`
- The app auto-selects CPU/GPU
