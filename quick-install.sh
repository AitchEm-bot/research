#!/bin/bash
# C2PA Research Pipeline - Quick Install Script
# This script pulls the Docker image and installs the c2pa wrapper command

set -e  # Exit on error

# Configuration
DOCKER_IMAGE="aitchem037/c2pa-research:latest"
GITHUB_REPO="AitchEm-bot/research"
GITHUB_BRANCH="master"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}C2PA Research Pipeline - Quick Install${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)
        PLATFORM="Linux"
        INSTALL_DIR="/usr/local/bin"
        WRAPPER_NAME="c2pa"
        ;;
    Darwin*)
        PLATFORM="macOS"
        INSTALL_DIR="/usr/local/bin"
        WRAPPER_NAME="c2pa"
        ;;
    *)
        echo -e "${RED}[ERROR] Unsupported OS: $OS${NC}"
        echo "This script is for Linux/macOS only."
        echo "For Windows, use: irm https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/quick-install.ps1 | iex"
        exit 1
        ;;
esac

echo -e "${GREEN}[OK]${NC} Platform detected: $PLATFORM"
echo ""

# Check if running as root for system-wide install
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
        echo -e "${YELLOW}[INFO]${NC} Will use 'sudo' for system-wide installation"
    else
        echo -e "${RED}[ERROR] This script requires sudo privileges to install to $INSTALL_DIR${NC}"
        echo "Please run with: sudo bash quick-install.sh"
        exit 1
    fi
fi
echo ""

# Step 1: Check Docker
echo -e "${YELLOW}[1/3] Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker not found!${NC}"
    echo ""
    echo "Please install Docker first:"
    case "$PLATFORM" in
        Linux)
            echo "  Ubuntu/Debian: https://docs.docker.com/engine/install/ubuntu/"
            echo "  Other Linux: https://docs.docker.com/engine/install/"
            ;;
        macOS)
            echo "  Docker Desktop for Mac: https://docs.docker.com/desktop/mac/install/"
            ;;
    esac
    exit 1
fi

DOCKER_VERSION=$(docker --version)
echo -e "${GREEN}[OK]${NC} Docker installed: $DOCKER_VERSION"
echo ""

# Step 2: Pull Docker image
echo -e "${YELLOW}[2/3] Pulling Docker image: $DOCKER_IMAGE${NC}"
if docker pull "$DOCKER_IMAGE"; then
    echo -e "${GREEN}[OK]${NC} Docker image pulled successfully"
else
    echo -e "${RED}[ERROR] Failed to pull Docker image${NC}"
    echo ""
    echo "Please check:"
    echo "  1. Image name is correct: $DOCKER_IMAGE"
    echo "  2. You have internet connection"
    echo "  3. Image exists on Docker Hub/GitHub Container Registry"
    exit 1
fi
echo ""

# Step 3: Install wrapper script
echo -e "${YELLOW}[3/3] Installing c2pa wrapper command...${NC}"

WRAPPER_URL="https://raw.githubusercontent.com/$GITHUB_REPO/$GITHUB_BRANCH/c2pa"
TEMP_FILE="/tmp/c2pa_wrapper"

# Download wrapper script
if curl -fsSL "$WRAPPER_URL" -o "$TEMP_FILE"; then
    echo -e "${GREEN}[OK]${NC} Wrapper script downloaded"
else
    echo -e "${RED}[ERROR] Failed to download wrapper script from:${NC}"
    echo "  $WRAPPER_URL"
    exit 1
fi

# Update Docker image name in wrapper script
sed -i.bak "s|DEFAULT_IMAGE=\"c2pa-research\"|DEFAULT_IMAGE=\"$DOCKER_IMAGE\"|g" "$TEMP_FILE"
rm -f "$TEMP_FILE.bak"

# Install to system PATH
if $SUDO cp "$TEMP_FILE" "$INSTALL_DIR/$WRAPPER_NAME"; then
    $SUDO chmod +x "$INSTALL_DIR/$WRAPPER_NAME"
    rm -f "$TEMP_FILE"
    echo -e "${GREEN}[OK]${NC} Installed to: $INSTALL_DIR/$WRAPPER_NAME"
else
    echo -e "${RED}[ERROR] Failed to install wrapper script${NC}"
    rm -f "$TEMP_FILE"
    exit 1
fi
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
if command -v c2pa &> /dev/null; then
    echo -e "${GREEN}[OK]${NC} 'c2pa' command is available"
else
    echo -e "${YELLOW}[WARNING] 'c2pa' not found in PATH${NC}"
    echo "You may need to restart your terminal or add $INSTALL_DIR to PATH"
fi
echo ""

# Print success message
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${GREEN}Quick Start:${NC}"
echo "  c2pa --help              Show all commands"
echo "  c2pa test                Quick test run (uses preset assets)"
echo "  c2pa run                 Full pipeline with presets"
echo "  c2pa phase 0 --test      Generate test assets"
echo "  c2pa status              Check pipeline status"
echo ""
echo -e "${BLUE}Phase-by-Phase Execution:${NC}"
echo "  c2pa phase 0             Asset generation/loading"
echo "  c2pa phase 1             C2PA embedding"
echo "  c2pa phase 2             Transformations"
echo "  c2pa phase 2.5           Platform testing setup (optional)"
echo "  c2pa phase 3             Verification & metrics"
echo "  c2pa phase 4             Analysis & visualization"
echo ""
echo -e "${BLUE}Results Location:${NC}"
echo "  All outputs: ./c2pa-results/"
echo ""
echo -e "${YELLOW}Documentation:${NC}"
echo "  https://github.com/$GITHUB_REPO/blob/$GITHUB_BRANCH/README_DOCKER.md"
echo ""
