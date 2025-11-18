# C2PA Robustness Research Pipeline - Docker Image
# Base image: NVIDIA CUDA 12.6 with cuDNN 9 runtime on Ubuntu 24.04
# Note: CUDA 12.6 is forward-compatible with CUDA 12.1 dependencies
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Disable PEP 668 (externally-managed-environment) for Docker
# This allows pip install without --break-system-packages in containers
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    git \
    wget \
    curl \
    # Python 3.12 (default in Ubuntu 24.04)
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    # Media processing (FFmpeg with VMAF support)
    ffmpeg \
    # Image libraries (Ubuntu 24.04 package names)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Note: Ubuntu 24.04 ships with Python 3.12.3+ by default (matches local 3.12.6)
# FFmpeg includes libvmaf support, accessed through ffmpeg-python wrapper

# Update pip and setuptools
# Note: Use --ignore-installed to avoid Ubuntu 24.04 packaging conflicts
RUN python3 -m pip install --upgrade --ignore-installed pip setuptools wheel

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
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# --------------------------
# C2PA Tool
# --------------------------
# Note: c2patool installation skipped - should be installed on host system
# The pipeline uses c2patool via utils.py which looks for it in:
# 1. Environment variable: C2PATOOL_PATH
# 2. Local path: tools/c2patool/c2patool/c2patool.exe (Windows)
# 3. System PATH: c2patool
#
# For Docker usage, install c2patool manually or mount it as a volume
# See README_DOCKER.md for installation instructions

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
ENV PYTHONPATH=/workspace

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