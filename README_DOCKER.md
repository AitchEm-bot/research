# 🐳 Docker Deployment Guide for C2PA Robustness Research Pipeline

**Complete guide for deploying and using the containerized C2PA robustness testing pipeline**

---

## 📖 Overview

This guide helps you deploy and run the C2PA robustness research pipeline using Docker. Whether you're reproducing results from a published paper or running your own experiments, this guide will walk you through every step.

### What is Docker and Why Use It?

**Docker** packages the entire research environment (Python, PyTorch, CUDA, dependencies) into a single image that runs identically on any machine. This ensures:
- ✅ **Reproducibility**: Same results on different machines
- ✅ **No dependency hell**: All packages pre-installed and tested
- ✅ **Isolation**: Won't conflict with your system Python
- ✅ **Portability**: Works on Windows, Linux, macOS

### Understanding Docker Concepts

Before starting, understand these key concepts:

```
┌─────────────────────────────────────────────────────────┐
│  DOCKER IMAGE (c2pa-research)                          │
│  ─────────────────────────────────────────────────     │
│  Like a "template" or "recipe" containing:             │
│  • Ubuntu 24.04 + Python 3.12                          │
│  • PyTorch, CUDA, FFmpeg                               │
│  • Your research scripts                               │
│  • Empty folder structure                              │
│                                                         │
│  Size: ~16 GB                                          │
│  Storage: Docker's internal database                   │
│  NOT directly browseable as regular files              │
└─────────────────────────────────────────────────────────┘
                         ↓ docker run
┌─────────────────────────────────────────────────────────┐
│  CONTAINER (running instance)                          │
│  ─────────────────────────────────────────────────     │
│  Like a "cookie" made from the template:               │
│  • Isolated environment                                │
│  • Can read/write files inside                         │
│  • Changes disappear when stopped (unless mounted)     │
│                                                         │
│  Each container is completely independent              │
└─────────────────────────────────────────────────────────┘
                         ↕ Volume Mount (-v)
┌─────────────────────────────────────────────────────────┐
│  YOUR LOCAL MACHINE                                    │
│  ─────────────────────────────────────────────────     │
│  C:\Users\you\research\data\  (Windows)                │
│  /home/you/research/data/      (Linux)                 │
│                                                         │
│  • Directly accessible files                           │
│  • Visible in File Explorer / Finder                   │
│  • Persist forever after container stops               │
│  • Can commit to git, share with others                │
└─────────────────────────────────────────────────────────┘
```

**Critical Concept: Volume Mounting**

Without `-v` volume mounting:
- ❌ Files created inside container are **trapped** inside
- ❌ **Disappear** when container stops
- ❌ Cannot access results with Excel, pandas, or File Explorer

With `-v` volume mounting:
- ✅ Files created inside container appear on **your local machine**
- ✅ **Persist forever** after container stops
- ✅ Accessible with any local tool
- ✅ **Required for reproducible research**

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Deployment Options](#-deployment-options)
  - [Option 0: Quick Install (Recommended)](#option-0-quick-install-recommended-for-most-users)
  - [Option 1: Pull Public Image (Manual)](#option-1-pull-public-image-manual-docker-commands)
  - [Option 2: Build from Source](#option-2-build-from-source)
- [Understanding File Storage](#-understanding-file-storage-critical)
- [Running Your First Test](#-running-your-first-test)
- [Complete Pipeline Execution](#-complete-pipeline-execution)
- [Accessing Results](#-accessing-results)
- [Volume Management](#-volume-management)
- [GPU Configuration](#%EF%B8%8F-gpu-configuration)
- [Troubleshooting](#-troubleshooting)
- [Advanced Usage](#-advanced-usage)
- [Best Practices for Reproducibility](#-best-practices-for-reproducibility)

---

## 🔧 Prerequisites

### Required Software

#### 1. Docker Desktop (Required for all users)

**Windows:**
```powershell
# Download from: https://www.docker.com/products/docker-desktop/
# Install Docker Desktop with WSL2 backend
# After installation, open PowerShell and verify:
docker --version
# Expected output: Docker version 20.10+
```

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
docker --version
```

**macOS:**
```bash
# Download from: https://www.docker.com/products/docker-desktop/
# Or use Homebrew:
brew install --cask docker
docker --version
```

#### 2. NVIDIA GPU Driver (Required for GPU acceleration)

**Check your current driver:**
```bash
nvidia-smi
```

**Required:** NVIDIA Driver version 525.60.13 or newer (supports CUDA 12.1+)

**Windows:** Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y nvidia-driver-535  # or latest
sudo reboot
```

#### 3. NVIDIA Container Toolkit (Linux only - Windows users skip this)

**Windows users:** Docker Desktop includes GPU support automatically when WSL2 backend is enabled.

**Linux users:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU with 6GB VRAM | RTX 4060 (8GB) or better |
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB free | 100GB free |
| **OS** | Windows 10/11, Ubuntu 20.04+ | Windows 11, Ubuntu 24.04 |
| **CPU** | 4 cores | 8+ cores |

---

## 🚀 Deployment Options

### Option 0: Quick Install (Recommended for Most Users)

**Best for:** Fastest setup, easiest to use, recommended for peer reviewers and researchers

The quick-install script automatically pulls the Docker image and installs the `c2pa` command-line wrapper:

**Linux/macOS:**
```bash
curl -sSL https://raw.githubusercontent.com/AitchEm-bot/research/master/quick-install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/AitchEm-bot/research/master/quick-install.ps1 | iex
```

**What this does:**
1. ✅ Checks Docker installation
2. ✅ Pulls the pre-built Docker image (~16 GB)
3. ✅ Downloads and installs `c2pa` wrapper scripts
4. ✅ Configures PATH automatically
5. ✅ Verifies installation

**After installation (Windows users: restart PowerShell):**

```bash
# Quick test run with preset assets
c2pa test

# Full pipeline with preset assets
c2pa run

# Phase-by-phase execution
c2pa phase 0             # Asset generation/loading
c2pa phase 1             # C2PA embedding
c2pa phase 2             # Transformations
c2pa phase 2.5           # Platform testing setup (optional)
c2pa phase 3             # Verification & metrics
c2pa phase 4             # Analysis & visualization

# Custom generation
c2pa phase 0 --images 50 --videos 10    # Generate custom counts

# Check status
c2pa status

# Interactive shell
c2pa shell
```

**Configuration (optional):**
```bash
# Set custom Docker image
export C2PA_IMAGE=aitchem037/c2pa-research:v1.0

# Set custom data directory
export C2PA_DATA_DIR=/path/to/results

# Disable GPU (use CPU only)
export C2PA_GPU=false
```

**What you get:**
- ✅ One-command installation
- ✅ Simple `c2pa` command instead of long docker commands
- ✅ Automatic volume mounting to `./c2pa-results/`
- ✅ GPU support configured automatically
- ✅ Model cache persistence
- ✅ Cross-platform compatibility
- ✅ Preset assets included (10 images + 2 videos for quick testing)

**Skip ahead to:** [Running Your First Test](#-running-your-first-test)

---

### Option 1: Pull Public Image (Manual Docker Commands)

**Best for:** Reproducing published results, quick testing

```bash
# Step 1: Pull the pre-built image (~16 GB download)
docker pull aitchem037/c2pa-research:latest

# Step 2: Verify image
docker images | grep c2pa-research

# Step 3: Create local data directory
mkdir my-research-data
cd my-research-data

# Step 4: Run test to verify everything works
docker run --rm --gpus all \
  -v $(pwd):/workspace/data \
  aitchem037/c2pa-research:latest run-all --test
```

**What you get:**
- ✅ Pre-built environment (no build time)
- ✅ Verified dependencies
- ✅ Ready to run immediately
- ❌ Cannot modify Python scripts easily
- ❌ No access to source code for browsing

---

### Option 2: Build from Source

**Best for:** Development, customization, contributing

#### Step 1: Clone the Repository

```bash
git clone https://github.com/AitchEm-bot/research.git
cd research
```

#### Step 2: Configure Environment (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit settings (optional)
# nano .env  (Linux/Mac)
# notepad .env  (Windows)

# Key settings:
# CUDA_VISIBLE_DEVICES=0              # Which GPU to use
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
```

#### Step 3: Build the Docker Image

```bash
# Standard build (takes 10-15 minutes)
docker build -t c2pa-research .

# Monitor build progress
docker build --progress=plain -t c2pa-research .

# Verify build success
docker images | grep c2pa-research
# Should show: c2pa-research latest [IMAGE_ID] [SIZE]
```

**What you get:**
- ✅ Full source code access
- ✅ Can modify scripts and rebuild
- ✅ Development environment
- ✅ Can contribute changes back
- ❌ Longer initial setup time

---

## 💾 Understanding File Storage (CRITICAL!)

### Where Are Files Stored?

This is the **most important concept** for reproducibility:

#### Scenario A: WITHOUT Volume Mounting ❌

```bash
# BAD - Files trapped inside container
docker run --rm --gpus all c2pa-research run-all --test
```

**What happens:**
```
Container creates:           Your local machine:
/workspace/data/results/  →  (nothing)
  └─ final_metrics.csv       ❌ File not accessible
                             ❌ Disappears when container stops
```

**Result:** You cannot access your results! ❌

---

#### Scenario B: WITH Volume Mounting ✅

```bash
# GOOD - Files appear on your machine
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test
```

**What happens:**
```
Container creates:           Your local machine:
/workspace/data/results/  ⟺  C:\Users\you\research\data\results\
  └─ final_metrics.csv       ✅ File appears instantly
                             ✅ Persists after container stops
                             ✅ Open in Excel, VS Code, etc.
```

**Result:** Full access to results! ✅

---

### Volume Mounting Syntax

**Linux/macOS:**
```bash
-v $(pwd)/data:/workspace/data
# $(pwd) = current directory
# Example: /home/alice/research/data → /workspace/data
```

**Windows PowerShell:**
```powershell
-v ${PWD}/data:/workspace/data
# ${PWD} = current directory
# Example: C:\Users\alice\research\data → /workspace/data
```

**Windows Command Prompt:**
```cmd
-v %cd%/data:/workspace/data
# %cd% = current directory
```

**Absolute paths (works everywhere):**
```bash
# Linux/macOS
-v /home/alice/research/data:/workspace/data

# Windows (note forward slashes!)
-v C:/Users/alice/research/data:/workspace/data
```

---

## ✅ Running Your First Test

Let's run a complete test to verify everything works:

### Step 1: Create a Test Directory

```bash
# Create a clean directory for testing
mkdir c2pa-test-run
cd c2pa-test-run
```

### Step 2: Run Test Pipeline

**If you installed using Option 0 (Quick Install):**
```bash
c2pa test
```

**If you're using manual Docker commands (Option 1 or 2):**

**Linux/macOS:**
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test
```

**Windows PowerShell:**
```powershell
docker run --rm --gpus all `
  -v ${PWD}/data:/workspace/data `
  c2pa-research run-all --test
```

### Step 3: Monitor Progress

You'll see output like:
```
============================================================
    C2PA Robustness Research Pipeline - Docker Container
============================================================
[✓] Python installed: Python 3.12.3
[✓] GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU, 8192 MiB
[✓] CUDA version: 12.6

╭────────────────────────────────────────────────────╮
│ C2PA Robustness Research Pipeline                  │
│ Is C2PA's Metadata Robust in AI-Generated Content? │
╰────────────────────────────────────────────────────╯

🎯 Starting Phase 1: Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generating 10 test images...
[████████████████████████████████] 10/10 complete
```

### Step 4: Verify Results

Check that files appeared on your local machine:

```bash
# List generated files
ls -lh data/assets/raw_images/
# Should show: img_000_seed42_*.png, img_001_seed43_*.png, ...

# Check results CSV
cat data/results/csv/final_metrics.csv | head -20
# Should show: metric data with PSNR, SSIM, VMAF values

# Count total results
wc -l data/results/csv/final_metrics.csv
# Test mode: ~50-100 rows
# Full mode: ~3,620 rows
```

**Windows:**
```powershell
# List generated files
Get-ChildItem data\assets\raw_images\

# Check results CSV
Get-Content data\results\csv\final_metrics.csv | Select-Object -First 20

# Open in Excel
start excel data\results\csv\final_metrics.csv
```

### Step 5: Success Indicators

✅ **Test passed if you see:**
- Files in `data/assets/raw_images/` (generated images)
- Files in `data/prepared_assets/manifests/images/` (C2PA signed images)
- `data/results/csv/final_metrics.csv` with metrics
- `data/results/logs/pipeline_orchestrator.log` with execution log

❌ **Test failed if:**
- No files in `data/` directory → Forgot volume mounting!
- CUDA out of memory errors → Reduce GPU memory (see Troubleshooting)
- Permission denied errors → Check file ownership (Linux)

---

## 🎯 Complete Pipeline Execution

### Full Production Run

**Estimated time:**
- Phase 1 (Generation): 2-4 hours (100 images + 30 videos)
- Phase 1.5 (C2PA Embedding): 10-15 minutes
- Phase 2 (Transformations): 1-2 hours
- Phase 3 (Verification & Metrics): 30-45 minutes
- Phase 4 (Analysis): 5-10 minutes
- **Total: 4-8 hours** (depending on GPU)

```bash
# Run complete pipeline
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  -v torch-cache:/workspace/.cache/torch \
  --name c2pa-production-run \
  c2pa-research run-all

# Monitor progress in another terminal
docker logs -f c2pa-production-run
```

### Phase-by-Phase Execution

```bash
# Phase 0: Generate AI assets (images and videos) - Optional, separate step
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research phase0 --images 100 --videos 30

# Phase 1: Embed C2PA manifests (auto-copies presets if no assets exist)
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  c2pa-research phase1

# Phase 2: Apply transformations (compression and editing)
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  c2pa-research phase2

# Phase 3: Verify C2PA and calculate quality metrics
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research phase3

# Phase 4: Data analysis and visualization
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  c2pa-research phase4
```

### Resume from Checkpoint

If pipeline was interrupted:

```bash
# Check current status
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  c2pa-research status

# Resume from specific phase
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --resume-from 2.5

# Available resume points: 1, 1.5, 2, 3, 4
```

### Using Docker Compose (Recommended for Production)

```bash
# Start pipeline in background
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop pipeline
docker-compose stop

# Resume pipeline
docker-compose start

# Cleanup
docker-compose down
```

---

## 📊 Accessing Results

### Method 1: Direct File Access (Recommended)

Because you used volume mounting, results are already on your local machine!

**Open in Excel:**
```bash
# Windows
start excel data/results/csv/final_metrics.csv

# macOS
open data/results/csv/final_metrics.csv

# Linux
libreoffice data/results/csv/final_metrics.csv
```

**Analyze with Python/pandas:**
```python
import pandas as pd

# Load results
df = pd.read_csv('data/results/csv/final_metrics.csv')

# Basic statistics
print(df.describe())

# Filter to specific transform
jpeg_results = df[df['transform_type'] == 'jpeg_compression']
print(jpeg_results[['transform_level', 'psnr', 'ssim', 'manifest_present']])
```

**View visualizations:**
```bash
# Linux/macOS
open data/results/analysis_results/plots/vsr_by_transform.png

# Windows
start data\results\analysis_results\plots\vsr_by_transform.png
```

---

### Method 2: Copy from Container (If You Forgot Volume Mounting)

If you ran without `-v` and need to extract results:

```bash
# Find your container
docker ps -a

# Copy results out
docker cp [CONTAINER_ID]:/workspace/data/results ./recovered_results

# Example
docker cp a52c0cb6c042:/workspace/data/results ./recovered_results
```

---

### Method 3: Interactive Exploration

```bash
# Start interactive shell
docker run --rm -it --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research bash

# Inside container, explore results
cd /workspace/data/results
ls -lh csv/
head -20 csv/final_metrics.csv
python3 -c "import pandas as pd; print(pd.read_csv('csv/final_metrics.csv').describe())"
exit
```

---

### Expected Output Structure

After successful execution:

```
data/
├── assets/
│   ├── raw_images/              # 100 generated images (1024x1024 PNG)
│   │   ├── img_000_seed42_20251118_*.png
│   │   └── ... (100 images)
│   ├── raw_videos/              # 30 generated videos (512x512 MP4)
│   │   ├── video_0_seed100_*.mp4
│   │   └── ... (30 videos)
│   └── raw_out_videos/          # External videos (if provided)
├── prepared_assets/
│   ├── manifests/
│   │   ├── images/              # 100 C2PA-signed images
│   │   └── videos/
│   │       ├── internal/        # 30 C2PA-signed internal videos
│   │       └── external/        # C2PA-signed external videos
│   ├── transformed/
│   │   ├── compression/         # Compressed versions (JPEG/PNG/H264/H265)
│   │   └── editing/             # Edited versions (resize/crop/rotate)
│   └── platform_tests/          # Social media test results (optional)
├── results/
│   ├── csv/
│   │   ├── final_metrics.csv    # 📊 ~3,620 rows (full run)
│   │   ├── c2pa_validation.csv  # C2PA verification details
│   │   ├── quality_metrics.csv  # PSNR/SSIM/VMAF metrics
│   │   └── platform_results.csv # Platform test results (if run)
│   ├── analysis_results/
│   │   ├── plots/               # 📈 Visualization figures
│   │   │   ├── vsr_by_transform.png
│   │   │   ├── quality_distribution.png
│   │   │   └── ... (10+ plots)
│   │   └── csv/                 # Analysis summaries
│   └── logs/
│       ├── pipeline_orchestrator.log  # Master log
│       ├── phase1_generation.log
│       └── ... (detailed logs per phase)
```

---

## 📁 Volume Management

### Understanding Docker Volumes

Docker uses two types of volumes:

1. **Bind mounts** (what we use for `data/`): Maps local directory to container
2. **Named volumes** (what we use for model caches): Managed by Docker

### Persistent Model Cache

```bash
# Create named volumes for model caches (one-time setup)
docker volume create huggingface-cache
docker volume create torch-cache

# Run with persistent caches (models won't re-download)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  -v torch-cache:/workspace/.cache/torch \
  c2pa-research run-all
```

**Benefits:**
- ✅ Models download once, reused across runs
- ✅ Faster startup for subsequent runs
- ✅ Saves bandwidth (Stable Diffusion v1.4 is 4+ GB)

### Managing Volumes

```bash
# List all volumes
docker volume ls

# Inspect volume
docker volume inspect huggingface-cache

# Check volume size
docker system df -v

# Remove volume (WARNING: deletes cached models)
docker volume rm huggingface-cache

# Remove all unused volumes
docker volume prune
```

### Backup and Restore

**Backup results:**
```bash
# Create timestamped backup
tar -czf results-backup-$(date +%Y%m%d).tar.gz data/results/

# Or backup entire data directory
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/
```

**Backup model cache (saves re-downloading):**
```bash
# Export model cache to tar file
docker run --rm \
  -v huggingface-cache:/cache \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/models-cache.tar.gz -C /cache .

# Restore model cache
docker run --rm \
  -v huggingface-cache:/cache \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/models-cache.tar.gz -C /cache
```

---

## 🖥️ GPU Configuration

### Verify GPU Access

```bash
# Test GPU from Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.1     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | ...
```

### Memory Optimization for 8GB GPUs

If you get "CUDA out of memory" errors:

```bash
# Method 1: Environment variables
docker run --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
  -e GENERATION_BATCH_SIZE=1 \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test

# Method 2: Edit .env file before building
echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256" >> .env
docker build -t c2pa-research .
```

### Multi-GPU Setup

```bash
# Use specific GPU (GPU 0)
CUDA_VISIBLE_DEVICES=0 docker run --gpus all ...

# Use multiple GPUs (GPU 0 and 1)
CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all ...

# Use all GPUs
docker run --gpus all ...

# Limit to 2 GPUs (any 2)
docker run --gpus 2 ...

# Use specific devices
docker run --gpus '"device=0,2"' ...
```

### CPU-Only Mode (No GPU)

```bash
# Run without GPU (much slower - 10-20x)
docker run \
  -e CUDA_VISIBLE_DEVICES=-1 \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test

# Expect: Phase 1 takes 2-4 hours instead of 15-30 minutes
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or inside container
docker exec [CONTAINER_NAME] nvidia-smi

# Monitor memory usage
docker exec [CONTAINER_NAME] python3 -c "
import torch
print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB')
"
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "No files in data/ directory after running"

**Problem:** Forgot volume mounting

**Solution:**
```bash
# BAD ❌
docker run --rm c2pa-research run-all --test

# GOOD ✅
docker run --rm -v $(pwd)/data:/workspace/data c2pa-research run-all --test
```

---

#### 2. "CUDA out of memory"

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 1.95 GiB (GPU 0; 7.79 GiB total capacity)
```

**Solutions:**

```bash
# Solution A: Reduce batch size
docker run --gpus all \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test

# Solution B: Clear GPU memory between phases
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  c2pa-research bash

# Inside container
python3 -c "import torch; torch.cuda.empty_cache()"
python3 scripts/run_pipeline.py phase1 --test
python3 -c "import torch; torch.cuda.empty_cache()"
python3 scripts/run_pipeline.py phase2 --test

# Solution C: Use CPU for some phases
docker run --gpus all -v $(pwd)/data:/workspace/data c2pa-research phase1 --test
docker run -v $(pwd)/data:/workspace/data c2pa-research phase2 --test  # CPU only
docker run --gpus all -v $(pwd)/data:/workspace/data c2pa-research phase3 --test
```

---

#### 3. "docker: Error response from daemon: could not select device driver"

**Problem:** Docker cannot access GPU

**Solutions:**

```bash
# Windows: Ensure WSL2 backend enabled
# Check Docker Desktop → Settings → General → Use WSL 2 based engine

# Linux: Install/restart NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all ubuntu nvidia-smi
```

---

#### 4. "Permission denied" (Linux)

**Error:**
```
PermissionError: [Errno 13] Permission denied: '/workspace/data/results/final_metrics.csv'
```

**Solutions:**

```bash
# Solution A: Fix ownership
sudo chown -R $USER:$USER data/

# Solution B: Run container with your user ID
docker run --user $(id -u):$(id -g) \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test

# Solution C: Use chmod (less secure)
chmod -R 777 data/
```

---

#### 5. "Slow model downloads"

**Problem:** HuggingFace models downloading slowly (Stable Diffusion is 4+ GB)

**Solutions:**

```bash
# Solution A: Use named volumes (models persist across runs)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  c2pa-research run-all

# Solution B: Pre-download models on host, then copy to volume
# On host:
pip install diffusers transformers
python3 -c "from diffusers import StableDiffusionPipeline; \
  StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')"

# Copy cache to Docker volume
docker volume create huggingface-cache
docker run --rm \
  -v ~/.cache/huggingface:/host-cache \
  -v huggingface-cache:/container-cache \
  ubuntu cp -r /host-cache/. /container-cache/

# Solution C: Use HuggingFace mirror (if in regions with slow access)
docker run -e HF_ENDPOINT=https://hf-mirror.com \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test
```

---

#### 6. "Container exits immediately"

**Check container logs:**

```bash
# View logs of stopped container
docker ps -a  # Find container ID
docker logs [CONTAINER_ID]

# Common causes:
# - Typo in command
# - Missing required files
# - Environment variable error
```

---

### Debugging Commands

```bash
# View container logs (running container)
docker logs -f [CONTAINER_NAME]

# View logs of stopped container
docker logs [CONTAINER_ID]

# Check container status
docker ps -a

# Inspect container details
docker inspect [CONTAINER_NAME]

# Execute command in running container
docker exec [CONTAINER_NAME] ls -la /workspace/data

# Check GPU usage inside container
docker exec [CONTAINER_NAME] nvidia-smi

# Check Python packages
docker exec [CONTAINER_NAME] pip3 list | grep torch

# Interactive debugging session
docker run --rm -it --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research bash

# Inside container:
python3 scripts/run_pipeline.py phase1 --test --debug
```

---

## 🚀 Advanced Usage

### Development Mode (Live Script Editing)

```bash
# Mount scripts as read-only volume for live editing
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/scripts:/workspace/scripts:ro \
  c2pa-research bash

# Now edit scripts on your machine, changes reflect immediately
# Useful for debugging and development
```

### Custom Pipeline Configuration

```bash
# Create custom run script
cat > run_custom.sh << 'EOF'
#!/bin/bash
set -e
echo "Running custom pipeline..."
python3 scripts/run_pipeline.py phase1 --test
python3 scripts/run_pipeline.py phase2 --skip-videos
python3 scripts/run_pipeline.py phase3 --force
python3 scripts/run_pipeline.py phase4 --publication
EOF
chmod +x run_custom.sh

# Run custom script
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/run_custom.sh:/workspace/run_custom.sh \
  c2pa-research bash /workspace/run_custom.sh
```

### Performance Monitoring

```bash
# Terminal 1: Run pipeline
docker run --name c2pa-monitor --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all

# Terminal 2: Monitor GPU
watch -n 1 docker exec c2pa-monitor nvidia-smi

# Terminal 3: Monitor resources
docker stats c2pa-monitor

# Check disk usage
docker exec c2pa-monitor df -h /workspace/data
```

### CI/CD Integration

**GitHub Actions example:**

```yaml
# .github/workflows/docker-pipeline.yml
name: C2PA Pipeline Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build Docker image
        run: docker build -t c2pa-research .

      - name: Run pipeline test
        run: |
          docker run --rm \
            -v ${{ github.workspace }}/data:/workspace/data \
            c2pa-research run-all --test

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: data/results/
```

---

## 🎓 Best Practices for Reproducibility

### 1. Always Use Volume Mounting

```bash
# ✅ GOOD - Results persist
docker run -v $(pwd)/data:/workspace/data c2pa-research run-all

# ❌ BAD - Results lost
docker run c2pa-research run-all
```

### 2. Use Named Volumes for Model Caches

```bash
# First run: Downloads models
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  c2pa-research run-all

# Subsequent runs: Reuses cached models (much faster)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  c2pa-research run-all
```

### 3. Document Your Exact Docker Command

```bash
# Create reproducibility script
cat > reproduce.sh << 'EOF'
#!/bin/bash
# Reproducibility script for "C2PA Robustness Study 2025"
# Docker Image: c2pa-research:v1.0
# GPU: NVIDIA RTX 4060 8GB
# Date: 2025-01-18

docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  -v torch-cache:/workspace/.cache/torch \
  --name c2pa-reproduction-2025 \
  c2pa-research:v1.0 run-all

echo "Results saved to: $(pwd)/data/results/"
EOF
chmod +x reproduce.sh
```

### 4. Tag Your Docker Images

```bash
# Tag image with version
docker tag c2pa-research c2pa-research:v1.0
docker tag c2pa-research c2pa-research:2025-01-18

# Future users can pull exact version
docker pull aitchem037/c2pa-research:v1.0
```

### 5. Share Results and Environment Together

```bash
# Package everything for sharing
tar -czf c2pa-reproduction-package.tar.gz \
  data/results/ \
  reproduce.sh \
  README.md \
  requirements.txt \
  Dockerfile

# Recipients can:
# 1. Extract package
# 2. Run ./reproduce.sh
# 3. Compare their results with your data/results/
```

### 6. Commit Docker Command to Git

Add to `README.md` or `REPRODUCING.md`:

```markdown
## Reproducibility

This study was conducted using Docker version 24.0.7 with NVIDIA Container Toolkit.

**Exact command used:**
```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  c2pa-research:v1.0 run-all
```

**Hardware:**
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- RAM: 32GB DDR5
- Storage: 1TB NVMe SSD

**Software versions:**
- Docker: 24.0.7
- NVIDIA Driver: 535.54.03
- CUDA: 12.1
```

---

## 🧹 Cleanup

### Remove Containers

```bash
# List all containers
docker ps -a

# Remove specific container
docker rm [CONTAINER_NAME]

# Remove all stopped containers
docker container prune

# Force remove running container
docker rm -f [CONTAINER_NAME]
```

### Remove Images

```bash
# List images
docker images

# Remove specific image
docker rmi c2pa-research

# Remove dangling images
docker image prune

# Remove all unused images
docker image prune -a
```

### Remove Volumes

```bash
# List volumes
docker volume ls

# Remove specific volume (WARNING: deletes cached models!)
docker volume rm huggingface-cache

# Remove all unused volumes
docker volume prune
```

### Complete Cleanup

```bash
# WARNING: This removes EVERYTHING
docker system prune -a --volumes

# More controlled cleanup
docker container prune  # Remove stopped containers
docker image prune -a   # Remove unused images
docker volume prune     # Remove unused volumes
```

### Remove Generated Data (Keep Source Code)

```bash
# Remove generated assets and results
rm -rf data/prepared_assets/
rm -rf data/results/

# Keep raw inputs for re-running
# data/assets/raw_images/
# data/assets/raw_videos/
# data/assets/raw_out_videos/
```

---

## 📚 Additional Resources

- **C2PA Specification:** https://c2pa.org/specifications/
- **Docker Documentation:** https://docs.docker.com/
- **NVIDIA Container Toolkit:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
- **PyTorch Docker Images:** https://hub.docker.com/r/pytorch/pytorch
- **Stable Diffusion:** https://huggingface.co/CompVis/stable-diffusion-v1-4
- **Paper Repository:** [Link to your paper/preprint]

---

## 🤝 Support and Contributing

### Getting Help

1. **Check troubleshooting section** above
2. **Review container logs:** `docker logs [CONTAINER_NAME]`
3. **Open an issue** with:
   ```bash
   # Include these in your issue:
   docker --version
   docker images | grep c2pa-research
   nvidia-smi
   docker logs [CONTAINER_NAME]
   ```

### Contributing

```bash
# Fork and clone repository
git clone https://github.com/AitchEm-bot/research.git
cd research

# Make changes to scripts/Dockerfile
nano scripts/processing/generation/generate_images.py

# Rebuild image
docker build -t c2pa-research-dev .

# Test changes
docker run --rm -v $(pwd)/data:/workspace/data c2pa-research-dev run-all --test

# Submit pull request
git add .
git commit -m "Improve memory efficiency in image generation"
git push origin feature/memory-optimization
```

---

## 📊 Performance Benchmarks

Expected execution times (RTX 4060 8GB):

| Phase | Test Mode | Full Mode |
|-------|-----------|-----------|
| Phase 1: Generation | 5-10 min | 2-4 hours |
| Phase 1.5: C2PA Embedding | 30 sec | 10-15 min |
| Phase 2: Transformations | 2-5 min | 1-2 hours |
| Phase 3: Verification & Metrics | 1-2 min | 30-45 min |
| Phase 4: Analysis | 10 sec | 5-10 min |
| **Total** | **10-20 min** | **4-8 hours** |

**Disk space usage:**
- Docker image: ~16 GB
- Model cache (first run): ~5 GB
- Test mode data: ~500 MB
- Full mode data: ~15-20 GB

---

## ⚠️ Important Notes

1. **Volume mounting is mandatory for reproducibility** - Without `-v`, results are lost
2. **Named volumes save time** - Model caches persist across runs
3. **Containers are isolated** - Changes in one container don't affect others
4. **GPU is optional but highly recommended** - CPU-only mode is 10-20x slower
5. **Results are immediately accessible** - No need to extract from container

---

**Docker Image Version:** 1.0
**Last Updated:** 2025-01-18
**Maintainer:** [Your Name/Organization]
**License:** [Your License]

**For the main project README and source code documentation, see [README.md](README.md)**
