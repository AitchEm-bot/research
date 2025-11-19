#!/bin/bash
# Docker Entrypoint Script for C2PA Robustness Research Pipeline
# This script performs environment checks and launches the pipeline

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}    C2PA Robustness Research Pipeline - Docker Container    ${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python installation
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_status "Python installed: $PYTHON_VERSION"
else
    print_error "Python not found!"
    exit 1
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
        print_status "GPU detected: $GPU_INFO"

        # Show CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_status "CUDA version: $CUDA_VERSION"
    else
        print_warning "nvidia-smi failed - GPU may not be accessible"
        print_warning "Running in CPU-only mode (will be slower)"
    fi
else
    print_warning "No GPU detected - running in CPU-only mode"
    export CUDA_VISIBLE_DEVICES=-1
fi

# Check c2patool installation (optional - can be installed on host)
if command -v c2patool &> /dev/null; then
    C2PA_VERSION=$(c2patool --version 2>&1 | head -1)
    print_status "c2patool installed: $C2PA_VERSION"
elif [ -f "/usr/local/bin/c2patool" ]; then
    C2PA_VERSION=$(/usr/local/bin/c2patool --version 2>&1 | head -1)
    print_status "c2patool installed: $C2PA_VERSION"
    export PATH="/usr/local/bin:$PATH"
else
    print_warning "c2patool not found in container"
    print_warning "C2PA operations will require c2patool on host system"
    print_warning "Install from: https://github.com/contentauth/c2pa-rs/releases"
fi

# Check FFmpeg installation
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f3)
    print_status "FFmpeg installed: version $FFMPEG_VERSION"
else
    print_error "FFmpeg not found!"
    exit 1
fi

# Check if data directory is mounted
if [ -d "/workspace/data" ]; then
    print_status "Data directory mounted at /workspace/data"

    # Check available space
    AVAILABLE_SPACE=$(df -h /workspace/data | tail -1 | awk '{print $4}')
    print_status "Available space: $AVAILABLE_SPACE"
else
    print_warning "Data directory not found - creating it"
    mkdir -p /workspace/data
fi

# Create necessary directories if they don't exist
REQUIRED_DIRS=(
    "/workspace/data/assets/raw_images"
    "/workspace/data/assets/raw_videos"
    "/workspace/data/assets/raw_images_for_videos"
    "/workspace/data/assets/raw_out_videos"
    "/workspace/data/prepared_assets/signed_assets/images"
    "/workspace/data/prepared_assets/signed_assets/videos/internal"
    "/workspace/data/prepared_assets/signed_assets/videos/external"
    "/workspace/data/results/csv"
    "/workspace/data/results/logs"
    "/workspace/data/results/analysis_results"
    "/workspace/.cache/huggingface"
    "/workspace/.cache/torch"
)

echo -e "\n${BLUE}Creating required directories...${NC}"
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  Created: $dir"
    fi
done

# Set proper permissions for cache directories
chmod -R 777 /workspace/.cache 2>/dev/null || true

# Display environment variables
echo -e "\n${BLUE}Environment Configuration:${NC}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "  PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF:-default}"
echo "  HF_HOME: ${HF_HOME:-/workspace/.cache/huggingface}"
echo "  PROJECT_ROOT: ${PROJECT_ROOT:-/workspace}"
echo "  DATA_DIR: ${DATA_DIR:-/workspace/data}"

# Check if pipeline script exists
PIPELINE_SCRIPT="/workspace/scripts/run_pipeline.py"
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    print_error "Pipeline script not found at $PIPELINE_SCRIPT"
    exit 1
fi

# Print memory information
echo -e "\n${BLUE}System Resources:${NC}"
free -h | grep -E "^Mem|^Swap" | while read line; do
    echo "  $line"
done

# If no arguments provided, show help
if [ $# -eq 0 ]; then
    echo -e "\n${YELLOW}No command specified. Showing pipeline help:${NC}\n"
    python3 "$PIPELINE_SCRIPT" --help
    echo -e "\n${BLUE}Example commands:${NC}"
    echo "  docker run c2pa-research run-all                    # Run phases 1-4 (starts from embedding)"
    echo "  docker run c2pa-research run-all --test             # Run in test mode"
    echo "  docker run c2pa-research phase0 --images 50         # Asset generation (phase 0)"
    echo "  docker run c2pa-research phase1                     # C2PA embedding (phase 1)"
    echo "  docker run c2pa-research phase2                     # Transformations (phase 2)"
    echo "  docker run c2pa-research phase2_5                   # Platform testing setup (phase 2.5)"
    echo "  docker run c2pa-research phase3                     # Verification & metrics (phase 3)"
    echo "  docker run c2pa-research phase4                     # Analysis (phase 4)"
    echo "  docker run c2pa-research status                     # Check pipeline status"
    echo ""
    echo "  # With wrapper script:"
    echo "  c2pa phase 0 --images 50 --videos 10                # Generate 50 images + 10 videos"
    echo "  c2pa run                                            # Full pipeline (phases 1-4, uses presets)"
    exit 0
fi

# Handle special commands
case "$1" in
    "bash"|"/bin/bash")
        echo -e "\n${BLUE}Starting interactive shell...${NC}"
        exec /bin/bash
        ;;
    "python"|"python3")
        echo -e "\n${BLUE}Starting Python interpreter...${NC}"
        shift
        exec python3 "$@"
        ;;
    "nvidia-smi")
        exec nvidia-smi
        ;;
    "check")
        echo -e "\n${GREEN}All checks passed! Container is ready.${NC}"
        exit 0
        ;;
    *)
        # Run the pipeline with provided arguments
        echo -e "\n${BLUE}Starting pipeline with command: $@${NC}"
        echo "============================================================"
        echo ""

        # Execute the pipeline script
        exec python3 "$PIPELINE_SCRIPT" "$@"
        ;;
esac