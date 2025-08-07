# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/preprocessed models/saved_models logs checkpoints results

# Set up pre-commit hooks (optional, for development)
RUN pip install pre-commit && \
    pre-commit install

# Create a non-root user
RUN useradd -m -u 1000 quantumfl && \
    chown -R quantumfl:quantumfl /app
USER quantumfl

# Expose ports for federated learning server
EXPOSE 8000 8001

# Set default command
CMD ["python", "scripts/quick_start.py"]
