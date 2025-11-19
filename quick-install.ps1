# C2PA Research Pipeline - Quick Install Script (Windows)
# This script pulls the Docker image and installs the c2pa wrapper command

param(
    [string]$DockerImage = "aitchem037/c2pa-research:latest",
    [string]$GithubRepo = "AitchEm-bot/research",
    [string]$GithubBranch = "master"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "C2PA Research Pipeline - Quick Install" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Detect platform
Write-Host "[OK] Platform detected: Windows" -ForegroundColor Green
Write-Host ""

# Step 1: Check Docker
Write-Host "[1/3] Checking Docker installation..." -ForegroundColor Yellow

try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop for Windows first:"
    Write-Host "  https://docs.docker.com/desktop/windows/install/"
    Write-Host ""
    exit 1
}

Write-Host ""

# Step 2: Pull Docker image
Write-Host "[2/3] Pulling Docker image: $DockerImage" -ForegroundColor Yellow

try {
    docker pull $DockerImage
    Write-Host "[OK] Docker image pulled successfully" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to pull Docker image" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. Image name is correct: $DockerImage"
    Write-Host "  2. You have internet connection"
    Write-Host "  3. Image exists on Docker Hub/GitHub Container Registry"
    Write-Host ""
    exit 1
}

Write-Host ""

# Step 3: Install wrapper scripts
Write-Host "[3/3] Installing c2pa wrapper command..." -ForegroundColor Yellow

# Create installation directory
$BinDir = "$env:USERPROFILE\bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
    Write-Host "[OK] Created directory: $BinDir" -ForegroundColor Green
} else {
    Write-Host "[OK] Directory exists: $BinDir" -ForegroundColor Green
}

# Download wrapper scripts
$WrapperPs1Url = "https://raw.githubusercontent.com/$GithubRepo/$GithubBranch/c2pa.ps1"
$WrapperBatUrl = "https://raw.githubusercontent.com/$GithubRepo/$GithubBranch/c2pa.bat"
$TempPs1 = "$env:TEMP\c2pa.ps1"
$TempBat = "$env:TEMP\c2pa.bat"

try {
    # Download PowerShell wrapper
    Invoke-WebRequest -Uri $WrapperPs1Url -OutFile $TempPs1 -UseBasicParsing
    Write-Host "[OK] Downloaded c2pa.ps1" -ForegroundColor Green

    # Download batch wrapper
    Invoke-WebRequest -Uri $WrapperBatUrl -OutFile $TempBat -UseBasicParsing
    Write-Host "[OK] Downloaded c2pa.bat" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to download wrapper scripts" -ForegroundColor Red
    Write-Host "URLs attempted:" -ForegroundColor Yellow
    Write-Host "  $WrapperPs1Url"
    Write-Host "  $WrapperBatUrl"
    Write-Host ""
    exit 1
}

# Update Docker image name in wrapper scripts
try {
    # Update PowerShell script
    $ps1Content = Get-Content $TempPs1 -Raw
    $ps1Content = $ps1Content -replace 'C2PA_IMAGE = if \(\$env:C2PA_IMAGE\) \{ \$env:C2PA_IMAGE \} else \{ "c2pa-research" \}', "`$C2PA_IMAGE = if (`$env:C2PA_IMAGE) { `$env:C2PA_IMAGE } else { `"$DockerImage`" }"
    $ps1Content | Set-Content $TempPs1

    # Update batch script
    $batContent = Get-Content $TempBat -Raw
    $batContent = $batContent -replace 'if not defined C2PA_IMAGE set C2PA_IMAGE=c2pa-research', "if not defined C2PA_IMAGE set C2PA_IMAGE=$DockerImage"
    $batContent | Set-Content $TempBat

    Write-Host "[OK] Updated image references to: $DockerImage" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Could not update image name in wrappers" -ForegroundColor Yellow
    Write-Host "You can manually set C2PA_IMAGE environment variable later" -ForegroundColor Yellow
}

# Install to bin directory
try {
    Copy-Item $TempPs1 "$BinDir\c2pa.ps1" -Force
    Copy-Item $TempBat "$BinDir\c2pa.bat" -Force
    Remove-Item $TempPs1 -ErrorAction SilentlyContinue
    Remove-Item $TempBat -ErrorAction SilentlyContinue
    Write-Host "[OK] Installed to: $BinDir" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to install wrapper scripts" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Update PATH if needed
Write-Host "Updating PATH environment variable..." -ForegroundColor Yellow

$CurrentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($CurrentPath -notlike "*$BinDir*") {
    try {
        [Environment]::SetEnvironmentVariable("Path", "$CurrentPath;$BinDir", "User")
        Write-Host "[OK] Added $BinDir to PATH" -ForegroundColor Green
        Write-Host ""
        Write-Host "[IMPORTANT] Restart PowerShell for PATH changes to take effect" -ForegroundColor Yellow
        $NeedRestart = $true
    } catch {
        Write-Host "[WARNING] Could not update PATH automatically" -ForegroundColor Yellow
        Write-Host "Please add manually: $BinDir" -ForegroundColor Yellow
        $NeedRestart = $true
    }
} else {
    Write-Host "[OK] $BinDir already in PATH" -ForegroundColor Green
    $NeedRestart = $false
}

Write-Host ""

# Print success message
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

if ($NeedRestart) {
    Write-Host "Next Steps:" -ForegroundColor Yellow
    Write-Host "  1. Close this PowerShell window" -ForegroundColor White
    Write-Host "  2. Open a new PowerShell window" -ForegroundColor White
    Write-Host "  3. Run: c2pa --help" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "The 'c2pa' command is now available!" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Quick Start:" -ForegroundColor Cyan
Write-Host "  c2pa --help              Show all commands"
Write-Host "  c2pa test                Quick test run (uses preset assets)"
Write-Host "  c2pa run                 Full pipeline with presets"
Write-Host "  c2pa phase 0 --test      Generate test assets"
Write-Host "  c2pa status              Check pipeline status"
Write-Host ""
Write-Host "Phase-by-Phase Execution:" -ForegroundColor Cyan
Write-Host "  c2pa phase 0             Asset generation/loading"
Write-Host "  c2pa phase 1             C2PA embedding"
Write-Host "  c2pa phase 2             Transformations"
Write-Host "  c2pa phase 2.5           Platform testing setup (optional)"
Write-Host "  c2pa phase 3             Verification & metrics"
Write-Host "  c2pa phase 4             Analysis & visualization"
Write-Host ""
Write-Host "Results Location:" -ForegroundColor Cyan
Write-Host "  All outputs: .\c2pa-results\"
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Cyan
Write-Host "  https://github.com/$GithubRepo/blob/$GithubBranch/README_DOCKER.md"
Write-Host ""
