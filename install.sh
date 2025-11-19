#!/bin/bash
# C2PA Docker Wrapper Installation Script
# For Linux and macOS

set -e

echo "========================================="
echo "C2PA Docker Wrapper Installation"
echo "========================================="
echo ""

# Check for Docker
echo "[1/4] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not found!"
    echo ""
    echo "Please install Docker first:"
    echo "  Linux:  https://docs.docker.com/engine/install/"
    echo "  macOS:  https://docs.docker.com/desktop/mac/install/"
    exit 1
fi

echo "[OK] Docker found: $(docker --version)"
echo ""

# Detect OS
echo "[2/4] Detecting operating system..."
OS="$(uname -s)"
case "$OS" in
    Linux*)
        echo "[OK] Linux detected"
        INSTALL_DIR="/usr/local/bin"
        ;;
    Darwin*)
        echo "[OK] macOS detected"
        INSTALL_DIR="/usr/local/bin"
        ;;
    *)
        echo "[ERROR] Unsupported operating system: $OS"
        exit 1
        ;;
esac
echo ""

# Check for sudo access
echo "[3/4] Checking permissions..."
if [ ! -w "$INSTALL_DIR" ]; then
    echo "[INFO] Installation requires sudo access"
    SUDO="sudo"
else
    SUDO=""
fi
echo ""

# Install wrapper
echo "[4/4] Installing c2pa wrapper..."

# Copy wrapper script
$SUDO cp c2pa "$INSTALL_DIR/c2pa"
$SUDO chmod +x "$INSTALL_DIR/c2pa"

# Verify installation
if command -v c2pa &> /dev/null; then
    echo "[OK] Installation successful!"
    echo ""
    echo "========================================="
    echo "Installation Complete!"
    echo "========================================="
    echo ""
    echo "The 'c2pa' command is now available."
    echo ""
    echo "Quick Start:"
    echo "  c2pa --help              Show help"
    echo "  c2pa test                Quick test run"
    echo "  c2pa phase 0 --test      Generate test assets"
    echo "  c2pa run                 Full pipeline"
    echo ""
    echo "Documentation:"
    echo "  See README_DOCKER.md for detailed usage"
    echo ""
else
    echo "[ERROR] Installation failed"
    echo "Wrapper not found in PATH"
    exit 1
fi
