# C2PA Robustness Research Pipeline - Docker Wrapper (PowerShell)
# Phase-based execution wrapper for Windows PowerShell

param(
    [string]$Command,
    [string]$Phase,
    [Parameter(ValueFromRemainingArguments=$true)]
    $Arguments
)

# Configuration
$Image = if ($env:C2PA_IMAGE) { $env:C2PA_IMAGE } else { "aitchem037/c2pa-research:latest" }
$OutputDir = if ($env:C2PA_DATA_DIR) { $env:C2PA_DATA_DIR } else { ".\c2pa-results" }
$ToolsDir = if ($env:C2PA_TOOLS_DIR) { $env:C2PA_TOOLS_DIR } else { ".\tools" }

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Create model cache volumes
docker volume create huggingface-cache 2>$null | Out-Null
docker volume create torch-cache 2>$null | Out-Null

# Check if GPU is available
function Test-GpuAvailable {
    try {
        $dockerInfo = docker info 2>&1
        return $dockerInfo -match "nvidia"
    } catch {
        return $false
    }
}

# Helper function to run Docker
function Invoke-DockerCommand {
    param(
        [bool]$UseGpu,
        [string[]]$CommandArgs
    )

    $dockerArgs = @("run", "--rm")

    # Check GPU availability
    if ($UseGpu) {
        if (Test-GpuAvailable) {
            $dockerArgs += @("--gpus", "all")
        } else {
            Write-Host "[!] GPU not available, running in CPU mode"
        }
    }

    $dockerArgs += @(
        "-v", "$($OutputDir):/workspace/data"
    )

    # Use custom HF cache directory if set, otherwise use named volume
    if ($env:HF_CACHE_DIR) {
        $dockerArgs += @("-v", "$($env:HF_CACHE_DIR):/workspace/.cache/huggingface")
    } else {
        $dockerArgs += @("-v", "huggingface-cache:/workspace/.cache/huggingface")
    }

    $dockerArgs += @("-v", "torch-cache:/workspace/.cache/torch")

    $dockerArgs += $Image
    $dockerArgs += $CommandArgs

    & docker @dockerArgs
}

# Helper function to create platform directories
function New-PlatformDirectories {
    $platforms = @("instagram", "twitter", "facebook", "youtube", "tiktok", "whatsapp")

    foreach ($platform in $platforms) {
        New-Item -ItemType Directory -Force -Path "$OutputDir\prepared_assets\platform_tests\$platform\uploads" | Out-Null
        New-Item -ItemType Directory -Force -Path "$OutputDir\prepared_assets\platform_tests\$platform\returned" | Out-Null
    }
}

# Helper function to generate platform instructions
function New-PlatformInstructions {
    $instructionsContent = @"
========================================
PLATFORM TESTING INSTRUCTIONS
========================================

Assets have been distributed to platform folders.

WORKFLOW:
1. Navigate to: c2pa-results\prepared_assets\platform_tests\
2. For each platform folder:
   a. Find assets in the 'uploads\' subfolder
   b. Upload these assets to the respective platform
   c. Download them back from the platform
   d. Place downloads in the 'returned\' subfolder

3. After completing uploads/downloads for all platforms:
   a. Run: c2pa rename-returns    (standardizes filenames)
   b. Run: c2pa phase 3           (verifies C2PA and calculates metrics)

PLATFORMS TO TEST:
- instagram\    (images + videos)
- twitter\      (images + videos)
- facebook\     (images + videos)
- youtube\      (videos only)
- tiktok\       (videos only)
- whatsapp\     (images + videos)

NOTES:
- Maintain original filenames when possible
- If platform renames files, use descriptive names
- Document any upload/download issues in platform_notes.txt

For detailed instructions, see README_DOCKER.md section on Platform Testing.
"@

    $instructionsContent | Out-File -FilePath "$OutputDir\PLATFORM_UPLOAD_INSTRUCTIONS.txt" -Encoding UTF8

    Write-Host ""
    Write-Host "+================================================================+" -ForegroundColor Cyan
    Write-Host "|              PLATFORM TESTING SETUP COMPLETE                   |" -ForegroundColor Cyan
    Write-Host "+================================================================+" -ForegroundColor Cyan
    Write-Host ""
    Write-Host $instructionsContent
    Write-Host ""
    Write-Host "Instructions saved to: $OutputDir\PLATFORM_UPLOAD_INSTRUCTIONS.txt" -ForegroundColor Green
    Write-Host ""
}

# Main command router
switch ($Command) {
    "phase" {
        switch ($Phase) {
            "0" {
                Write-Host "[Phase 0: Asset Generation]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $true -CommandArgs (@("phase0") + $Arguments)
            }

            "1" {
                Write-Host "[Phase 1: C2PA Manifest Embedding]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $false -CommandArgs (@("phase1") + $Arguments)
            }

            "2" {
                Write-Host "[Phase 2: Transformation Pipeline]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $false -CommandArgs (@("phase2") + $Arguments)
            }

            "2.5" {
                Write-Host "[Phase 2.5: Platform Testing Setup]" -ForegroundColor Yellow
                Write-Host "Creating platform test directories..." -ForegroundColor Cyan
                New-PlatformDirectories

                # Run platform upload preparation
                Invoke-DockerCommand -UseGpu $false -CommandArgs (@("phase2-5") + $Arguments)

                # Generate instructions
                New-PlatformInstructions
            }

            "3" {
                Write-Host "[Phase 3: Verification & Quality Metrics]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $true -CommandArgs (@("phase3") + $Arguments)
            }

            "4" {
                Write-Host "[Phase 4: Analysis & Visualization]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $false -CommandArgs (@("phase4") + $Arguments)
            }

            default {
                Write-Host "Error: Invalid phase number: $Phase" -ForegroundColor Red
                Write-Host ""
                Write-Host "Valid phases:"
                Write-Host "  0    - Asset generation (separate step, run independently)"
                Write-Host "  1    - C2PA manifest embedding"
                Write-Host "  2    - Transformation pipeline"
                Write-Host "  2.5  - Platform testing setup (optional)"
                Write-Host "  3    - Verification & quality metrics"
                Write-Host "  4    - Analysis & visualization"
                Write-Host ""
                Write-Host "Usage: c2pa phase <number> [options]"
                Write-Host "Example: c2pa phase 0 --images 50 --videos 10"
                exit 1
            }
        }
    }

    "run" {
        Write-Host "[Full Pipeline: Phases 1-4, starts from embedding]" -ForegroundColor Yellow
        Write-Host "Note: Phase 0 (generation) is separate. Preset assets will be used." -ForegroundColor Cyan
        Invoke-DockerCommand -UseGpu $true -CommandArgs (@("run-all") + $Arguments)
    }

    "test" {
        Write-Host "[Quick Test Mode: Phases 1-4 with preset assets]" -ForegroundColor Yellow
        Invoke-DockerCommand -UseGpu $true -CommandArgs @("run-all", "--test")
    }

    "status" {
        Write-Host "[Pipeline Status Check]" -ForegroundColor Yellow
        Invoke-DockerCommand -UseGpu $false -CommandArgs @("status")
    }

    "rename-returns" {
        Write-Host "[Rename Platform Returns]" -ForegroundColor Yellow
        Write-Host "Standardizing filenames for returned platform assets..." -ForegroundColor Cyan
        Invoke-DockerCommand -UseGpu $false -CommandArgs @("python", "scripts/processing/preprocessing/platform/rename_platform_returns.py")
        Write-Host ""
        Write-Host "For WhatsApp files, also run:" -ForegroundColor Yellow
        Write-Host "  c2pa shell"
        Write-Host "  python scripts/processing/preprocessing/platform/rename_whatsapp_returns.py"
    }

    "shell" {
        Write-Host "[Interactive Shell]" -ForegroundColor Yellow
        $shellArgs = @("run", "--rm", "-it")

        # Add GPU support if available
        if (Test-GpuAvailable) {
            $shellArgs += @("--gpus", "all")
        }

        $shellArgs += @(
            "-v", "$($OutputDir):/workspace/data",
            "-v", "huggingface-cache:/workspace/.cache/huggingface",
            "-v", "torch-cache:/workspace/.cache/torch",
            $Image,
            "bash"
        )
        & docker @shellArgs
    }

    { $_ -in @("--help", "-h", "help") } {
        Write-Host @"
C2PA Robustness Research Pipeline - Docker Wrapper
===================================================

USAGE:
  c2pa <command> [options]

COMMANDS:
  phase 0         Generate AI assets (images with SD1.4, videos with SVD) - requires GPU
  phase 1         Embed C2PA manifests into assets using c2patool
  phase 2         Apply transformations (JPEG/PNG compression, H.264/H.265, resize, crop, rotate, trim)
  phase 2-5       Prepare assets for platform testing (optional, for social media round-trip tests)
  phase 3         Verify C2PA manifests and calculate quality metrics (PSNR, SSIM, VMAF)
  phase 4         Data analysis and visualization (exploratory plots or publication figures)
  run             Run complete pipeline (phases 1-4 sequentially)
  test            Quick test run with minimal assets
  status          Check pipeline execution status
  setup           Extract c2patool to local tools directory
  rename-returns  Rename downloaded platform files to standard format (run after phase 2.5 downloads)
  shell           Open interactive shell in container

OPTIONS:
  --test              Run in test mode (fewer assets, faster execution)
  --force             Force re-run even if checkpoint exists
  --publication       Generate F1-F7 publication figures (phase 4 and run)
  --skip-viz          Skip visualization generation (phase 4 only)
  --images N          Number of images to generate (phase 0 only)
  --videos N          Number of videos to generate (phase 0 only)
  --resume-from N     Resume from phase N, where N is 1-4 (run only)

EXAMPLES:
  c2pa phase 0 --images 50 --videos 10    # Generate 50 images + 10 videos
  c2pa phase 1                            # Sign assets with C2PA manifests
  c2pa phase 4 --publication              # Generate F1-F7 publication figures
  c2pa run --test                         # Quick test of full pipeline
  c2pa run --publication                  # Full run with publication figures
  c2pa run --resume-from 3                # Resume from phase 3

ENVIRONMENT VARIABLES:
  C2PA_IMAGE      Docker image (default: aitchem037/c2pa-research:latest)
  C2PA_DATA_DIR   Output directory (default: ./c2pa-results)
  C2PA_TOOLS_DIR  Tools directory (default: ./tools)

OUTPUT STRUCTURE:
  ./c2pa-results/
  +-- results/csv/                 CSV metrics files
      +-- final_metrics.csv        Main results
      +-- c2pa_validation.csv      C2PA verification
      +-- quality_metrics.csv      PSNR/SSIM/VMAF
  +-- results/analysis_results/
      +-- plots/                   Visualizations
      +-- csv/                     Analysis summaries
  +-- results/logs/                Execution logs

For detailed documentation, see README_DOCKER.md
"@
    }

    default {
        if ([string]::IsNullOrEmpty($Command)) {
            Write-Host "Error: No command specified" -ForegroundColor Red
            Write-Host "Run 'c2pa --help' for usage information"
            exit 1
        }

        # Forward unknown commands directly to Docker
        Invoke-DockerCommand -UseGpu $true -CommandArgs (@($Command) + $Arguments)
    }
}

# Show results summary if CSV exists
if (Test-Path "$OutputDir\csv\final_metrics.csv") {
    Write-Host ""
    Write-Host "+================================================================+" -ForegroundColor Green
    Write-Host "| Results available in: $OutputDir" -ForegroundColor Green
    Write-Host "| Metrics CSV: $OutputDir\csv\final_metrics.csv" -ForegroundColor Green
    Write-Host "| Plots: $OutputDir\plots\" -ForegroundColor Green
    Write-Host "+================================================================+" -ForegroundColor Green
}
