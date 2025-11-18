# 🐳 Docker Setup for C2PA Robustness Research Pipeline

This guide provides comprehensive instructions for running the C2PA robustness research pipeline in a containerized environment using Docker.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Volume Management](#volume-management)
- [GPU Configuration](#gpu-configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## 🔧 Prerequisites

### Required Software

1. **Docker Desktop** (version 20.10+)
   - Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: Install Docker Engine and Docker Compose

2. **NVIDIA Container Toolkit** (for GPU support)
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # Windows (WSL2)
   # NVIDIA Container Toolkit is included with Docker Desktop when WSL2 backend is enabled
   ```

3. **NVIDIA GPU Driver** (version 525.60.13+)
   - Must support CUDA 12.1
   - Verify with: `nvidia-smi`

### System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space (for Docker images and data)
- **OS**: Windows 10/11 with WSL2, Ubuntu 20.04+, or compatible Linux

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/c2pa-robustness.git
cd c2pa-robustness
```

### 2. Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings (optional)
# nano .env
```

### 3. Build the Docker Image

```bash
# Build the image (this may take 10-15 minutes)
docker build -t c2pa-research .

# Or use docker-compose
docker-compose build
```

### 4. Run the Complete Pipeline

```bash
# Run all phases with docker-compose
docker-compose up

# Or run directly with docker
docker run --gpus all -v $(pwd)/data:/workspace/data c2pa-research run-all
```

## 📘 Detailed Setup

### Step 1: Verify GPU Access

```bash
# Check if Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Expected output: Your GPU information
```

### Step 2: Configure Environment

Edit `.env` file to customize settings:

```bash
# Key settings to adjust
CUDA_VISIBLE_DEVICES=0                    # GPU index
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory optimization
HF_HOME=/workspace/.cache/huggingface     # Model cache location
```

### Step 3: Build the Image

```bash
# Standard build
docker build -t c2pa-research .

# Build with specific CUDA version
docker build --build-arg CUDA_VERSION=12.1.0 -t c2pa-research .

# Build with progress output
docker build --progress=plain -t c2pa-research .
```

### Step 4: Prepare Data Directories

```bash
# Create necessary directories
mkdir -p data/assets/raw_images
mkdir -p data/assets/raw_videos
mkdir -p data/assets/raw_out_videos
mkdir -p data/results/logs

# Optional: Place external videos in raw_out_videos/
cp /path/to/external/videos/*.mp4 data/assets/raw_out_videos/
```

## 🎯 Running the Pipeline

### Using Docker Compose (Recommended)

```bash
# Run complete pipeline
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop pipeline
docker-compose down
```

### Using Docker Run

```bash
# Full pipeline
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v huggingface-cache:/workspace/.cache/huggingface \
  c2pa-research run-all

# Test mode (faster, fewer assets)
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --test

# Specific phase only
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research phase2

# Resume from phase
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  c2pa-research run-all --resume-from 2
```

### Interactive Mode

```bash
# Start interactive shell
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  c2pa-research /bin/bash

# Inside container, run commands manually
python3 scripts/run_pipeline.py phase1 --test
python3 scripts/processing/generation/generate_images.py --count 5
```

## 📁 Volume Management

### Data Volumes

The pipeline uses several volume mappings:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/workspace/data` | Input/output data |
| Named volume | `/workspace/.cache/huggingface` | Model cache |
| Named volume | `/workspace/.cache/torch` | PyTorch cache |

### Persistent Model Cache

```bash
# Create named volumes for model caches
docker volume create huggingface-cache
docker volume create torch-cache

# List volumes
docker volume ls

# Inspect volume
docker volume inspect huggingface-cache

# Clean up volumes (WARNING: deletes cached models)
docker volume rm huggingface-cache torch-cache
```

### Backup and Restore

```bash
# Backup results
tar -czf results-backup.tar.gz data/results/

# Backup model cache
docker run --rm -v huggingface-cache:/cache \
  -v $(pwd):/backup ubuntu \
  tar czf /backup/models-backup.tar.gz -C /cache .

# Restore model cache
docker run --rm -v huggingface-cache:/cache \
  -v $(pwd):/backup ubuntu \
  tar xzf /backup/models-backup.tar.gz -C /cache
```

## 🖥️ GPU Configuration

### Memory Optimization

For GPUs with limited VRAM (8GB):

```bash
# Set in .env or docker-compose.yml
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
GENERATION_BATCH_SIZE=1
```

### Multi-GPU Setup

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1  # Use GPU 0 and 1

# Distribute across GPUs
docker run --gpus '"device=0,1"' ...
```

### CPU-Only Mode

```bash
# Run without GPU (much slower)
docker run -v $(pwd)/data:/workspace/data \
  -e CUDA_VISIBLE_DEVICES=-1 \
  c2pa-research run-all --test
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export GENERATION_BATCH_SIZE=1

# Clear cache between phases
docker exec c2pa-research-pipeline python3 -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Docker Can't Find GPU

**Error**: `docker: Error response from daemon: could not select device driver`

**Solution**:
```bash
# Verify NVIDIA Container Toolkit installation
nvidia-ctk --version

# Restart Docker daemon
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all ubuntu nvidia-smi
```

#### 3. Permission Denied Errors

**Error**: `Permission denied` when accessing files

**Solution**:
```bash
# Fix ownership (Linux)
sudo chown -R $USER:$USER data/

# Or run container with user ID
docker run --user $(id -u):$(id -g) ...
```

#### 4. Slow Model Downloads

**Issue**: HuggingFace models downloading slowly

**Solution**:
```bash
# Pre-download models outside Docker
python3 -c "from diffusers import StableDiffusionPipeline; \
  StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')"

# Then copy cache to Docker volume
docker cp ~/.cache/huggingface c2pa-research-pipeline:/workspace/.cache/
```

### Debugging Commands

```bash
# View container logs
docker logs c2pa-research-pipeline

# Check container status
docker ps -a

# Inspect container
docker inspect c2pa-research-pipeline

# Execute commands in running container
docker exec c2pa-research-pipeline ls -la /workspace/data

# Check GPU usage inside container
docker exec c2pa-research-pipeline nvidia-smi

# View Python package versions
docker exec c2pa-research-pipeline pip3 list | grep torch
```

## 🚀 Advanced Usage

### Custom Pipeline Configuration

```bash
# Create custom entrypoint script
cat > run_custom.sh << 'EOF'
#!/bin/bash
python3 scripts/run_pipeline.py phase1 --test
python3 scripts/run_pipeline.py phase2 --force
python3 scripts/run_pipeline.py phase3
python3 scripts/run_pipeline.py phase4 --publication
EOF

# Run with custom script
docker run --gpus all -v $(pwd)/data:/workspace/data \
  -v $(pwd)/run_custom.sh:/workspace/run_custom.sh \
  c2pa-research bash /workspace/run_custom.sh
```

### Development Mode

```bash
# Mount scripts for live editing
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/scripts:/workspace/scripts:ro \
  c2pa-research /bin/bash

# Install additional tools for debugging
docker exec c2pa-research-pipeline apt-get update && \
  apt-get install -y vim htop ncdu
```

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: Run Pipeline
on: [push]
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t c2pa-research .
      - name: Run pipeline test
        run: docker run -v $(pwd)/data:/workspace/data c2pa-research run-all --test
```

### Performance Monitoring

```bash
# Monitor GPU usage
watch -n 1 docker exec c2pa-research-pipeline nvidia-smi

# Monitor container resources
docker stats c2pa-research-pipeline

# Profile memory usage
docker exec c2pa-research-pipeline python3 -c "
import torch
print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB')
print(f'Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB')
"
```

## 📊 Expected Outputs

After successful pipeline execution:

```
data/
├── results/
│   ├── csv/
│   │   ├── final_metrics.csv         # ~3,620 rows of results
│   │   ├── c2pa_validation.csv       # C2PA verification results
│   │   ├── quality_metrics.csv       # PSNR/SSIM/VMAF metrics
│   │   └── platform_results.csv      # Platform test results
│   ├── analysis_results/
│   │   ├── plots/                    # Visualization figures
│   │   └── csv/                      # Analysis summaries
│   └── logs/
│       └── pipeline_orchestrator.log  # Execution logs
```

## 🔄 Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove Docker image
docker rmi c2pa-research

# Clean up volumes (WARNING: deletes cached models)
docker volume rm huggingface-cache torch-cache

# Remove all generated data
rm -rf data/prepared_assets/
rm -rf data/results/
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [C2PA Specification](https://c2pa.org/specifications/)

## 🤝 Support

For issues specific to Docker setup:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review container logs: `docker logs c2pa-research-pipeline`
3. Open an issue with:
   - Docker version: `docker --version`
   - GPU info: `nvidia-smi`
   - Error messages and logs

---

**Note**: This Docker setup is optimized for NVIDIA GPUs with CUDA 12.1 support. For CPU-only execution, expect significantly longer processing times (10-20x slower for generation phases).