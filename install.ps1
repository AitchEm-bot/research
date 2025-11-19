# C2PA Docker Wrapper Installation Script
# For Windows PowerShell

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "C2PA Docker Wrapper Installation" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check for Docker
Write-Host "[1/4] Checking Docker installation..." -ForegroundColor Yellow

try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop first:"
    Write-Host "  https://docs.docker.com/desktop/windows/install/"
    exit 1
}

Write-Host ""

# Detect system
Write-Host "[2/4] Detecting system..." -ForegroundColor Yellow
Write-Host "[OK] Windows detected" -ForegroundColor Green
Write-Host ""

# Create installation directory
Write-Host "[3/4] Setting up installation directory..." -ForegroundColor Yellow

$BinDir = "$env:USERPROFILE\bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
    Write-Host "[OK] Created directory: $BinDir" -ForegroundColor Green
} else {
    Write-Host "[OK] Directory exists: $BinDir" -ForegroundColor Green
}

Write-Host ""

# Install wrapper scripts
Write-Host "[4/4] Installing c2pa wrapper..." -ForegroundColor Yellow

try {
    Copy-Item "c2pa.ps1" "$BinDir\" -Force
    Copy-Item "c2pa.bat" "$BinDir\" -Force
    Write-Host "[OK] Wrapper scripts copied" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to copy wrapper scripts" -ForegroundColor Red
    Write-Host $_.Exception.Message
    exit 1
}

# Update PATH
Write-Host ""
Write-Host "Updating PATH environment variable..." -ForegroundColor Yellow

$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$BinDir", "User")
    Write-Host "[OK] Added $BinDir to PATH" -ForegroundColor Green
    Write-Host ""
    Write-Host "[IMPORTANT] Restart PowerShell for PATH changes to take effect" -ForegroundColor Yellow
} else {
    Write-Host "[OK] $BinDir already in PATH" -ForegroundColor Green
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The 'c2pa' command will be available after restarting PowerShell." -ForegroundColor White
Write-Host ""
Write-Host "Quick Start:" -ForegroundColor Cyan
Write-Host "  c2pa --help              Show help"
Write-Host "  c2pa test                Quick test run"
Write-Host "  c2pa phase 0 --test      Generate test assets"
Write-Host "  c2pa run                 Full pipeline"
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  See README_DOCKER.md for detailed usage"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Close this PowerShell window"
Write-Host "  2. Open a new PowerShell window"
Write-Host "  3. Run: c2pa --help"
Write-Host ""
