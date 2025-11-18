# C2PA Robustness Research Pipeline - Docker Image
# Base image: NVIDIA CUDA 12.1 with cuDNN 8 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && apt-get install -y \
    # Python 3.12
    software-properties-common \
    # Build tools
    build-essential \
    git \
    wget \
    curl \
    # Media processing
    ffmpeg \
    libvmaf-dev \
    # Image libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-dev python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Create python3 symlink
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --set python3 /usr/bin/python3.12

# --------------------------
# Working directory
# --------------------------
WORKDIR /workspace

# --------------------------
# Python dependencies
# --------------------------
# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python packages with CUDA 12.1 support
RUN pip3 install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# --------------------------
# C2PA Tool
# --------------------------
# Note: We copy from the Windows binary location
# In production, you might want to download the Linux version
COPY tools/c2patool/c2patool/c2patool.exe /tmp/c2patool_windows.exe

# Download Linux version of c2patool (if Windows binary doesn't work)
RUN wget -q https://github.com/contentauth/c2pa-rs/releases/download/v0.24.0/c2patool-v0.24.0-x86_64-unknown-linux-gnu.tar.gz -O /tmp/c2patool.tar.gz && \
    tar -xzf /tmp/c2patool.tar.gz -C /usr/local/bin && \
    chmod +x /usr/local/bin/c2patool && \
    rm /tmp/c2patool.tar.gz

# --------------------------
# Copy project files
# --------------------------
# Copy scripts directory
COPY scripts/ scripts/

# Copy project configuration files
COPY FLOW_DIAGRAM.md .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p data/assets/raw_images \
             data/assets/raw_videos \
             data/assets/raw_images_for_videos \
             data/assets/raw_out_videos \
             data/prepared_assets/signed_assets/images \
             data/prepared_assets/signed_assets/videos/internal \
             data/prepared_assets/signed_assets/videos/external \
             data/prepared_assets/c2pa_manifests \
             data/prepared_assets/transformed \
             data/prepared_assets/platform_tests \
             data/results/csv \
             data/results/logs \
             data/results/analysis_results \
             /workspace/.cache/huggingface \
             /workspace/.cache/torch

# --------------------------
# Environment Variables
# --------------------------
# GPU memory optimization for 8GB VRAM (RTX 4060)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV CUDA_MODULE_LOADING=LAZY

# Model cache directories
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Data paths
ENV PROJECT_ROOT=/workspace
ENV DATA_DIR=/workspace/data
ENV C2PATOOL_PATH=/usr/local/bin/c2patool

# Disable Python buffering for better logging
ENV PYTHONUNBUFFERED=1

# --------------------------
# Entrypoint script
# --------------------------
COPY scripts/docker_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (run pipeline help)
CMD ["--help"]