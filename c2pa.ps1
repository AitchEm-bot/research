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
        $result = docker run --rm --gpus all hello-world 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

# Helper function to run Docker
function Invoke-DockerCommand {
    param(
        [bool]$UseGpu,
        [string[]]$Args
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
        "-v", "$($OutputDir):/workspace/data",
        "-v", "huggingface-cache:/workspace/.cache/huggingface",
        "-v", "torch-cache:/workspace/.cache/torch"
    )

    # Mount tools directory if it exists
    if (Test-Path $ToolsDir) {
        $dockerArgs += @("-v", "$($ToolsDir):/workspace/tools")
    }

    $dockerArgs += $Image
    $dockerArgs += $Args

    & docker @dockerArgs
}

# Helper function to create platform directories
function New-PlatformDirectories {
    $platforms = @("instagram", "twitter", "facebook", "youtube", "tiktok", "whatsapp")

    foreach ($platform in $platforms) {
        New-Item -ItemType Directory -Force -Path "$OutputDir\platform_tests\$platform\uploads" | Out-Null
        New-Item -ItemType Directory -Force -Path "$OutputDir\platform_tests\$platform\returned" | Out-Null
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
1. Navigate to: c2pa-results\platform_tests\
2. For each platform folder:
   a. Find assets in the 'uploads\' subfolder
   b. Upload these assets to the respective platform
   c. Download them back from the platform
   d. Place downloads in the 'returned\' subfolder

3. After completing uploads/downloads for all platforms:
   Run: c2pa phase 3

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
                Invoke-DockerCommand -UseGpu $true -Args (@("phase0") + $Arguments)
            }

            "1" {
                Write-Host "[Phase 1: C2PA Manifest Embedding]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $false -Args (@("phase1") + $Arguments)
            }

            "2" {
                Write-Host "[Phase 2: Transformation Pipeline]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $false -Args (@("phase2") + $Arguments)
            }

            "2.5" {
                Write-Host "[Phase 2.5: Platform Testing Setup]" -ForegroundColor Yellow
                Write-Host "Creating platform test directories..." -ForegroundColor Cyan
                New-PlatformDirectories

                # Run platform upload preparation
                Invoke-DockerCommand -UseGpu $false -Args (@("phase2_5") + $Arguments)

                # Generate instructions
                New-PlatformInstructions
            }

            "3" {
                Write-Host "[Phase 3: Verification & Quality Metrics]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $true -Args (@("phase3") + $Arguments)
            }

            "4" {
                Write-Host "[Phase 4: Analysis & Visualization]" -ForegroundColor Yellow
                Invoke-DockerCommand -UseGpu $false -Args (@("phase4") + $Arguments)
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
        Invoke-DockerCommand -UseGpu $true -Args (@("run-all") + $Arguments)
    }

    "test" {
        Write-Host "[Quick Test Mode: Phases 1-4 with preset assets]" -ForegroundColor Yellow
        Invoke-DockerCommand -UseGpu $true -Args @("run-all", "--test")
    }

    "status" {
        Write-Host "[Pipeline Status Check]" -ForegroundColor Yellow
        Invoke-DockerCommand -UseGpu $false -Args @("status")
    }

    "shell" {
        Write-Host "[Interactive Shell]" -ForegroundColor Yellow
        $shellArgs = @("run", "--rm", "-it") + $GpuFlag.Split() + @(
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

PHASE-BY-PHASE EXECUTION:
  c2pa phase 0 [options]           Generate AI assets
    --images N                     Number of images to generate
    --videos N                     Number of videos to generate
    --test                         Generate test counts (10 images, 2 videos)

  c2pa phase 1 [options]           Embed C2PA manifests
  c2pa phase 2 [options]           Apply transformations
  c2pa phase 2.5 [options]         Setup platform testing (optional)
  c2pa phase 3 [options]           Verify C2PA & calculate metrics
  c2pa phase 4 [options]           Generate analysis & visualizations

FULL PIPELINE:
  c2pa run [options]               Run complete pipeline (skips 2.5)
  c2pa test                        Quick test with minimal assets

UTILITIES:
  c2pa status                      Check pipeline status
  c2pa shell                       Open interactive shell in container
  c2pa --help                      Show this help message

EXAMPLES:
  c2pa phase 0 --images 50 --videos 10    # Generate 50 images + 10 videos
  c2pa phase 0 --test                     # Generate test set
  c2pa run                                # Full pipeline with presets
  c2pa phase 2 --test                     # Test transformations only

ENVIRONMENT VARIABLES:
  C2PA_IMAGE                       Docker image name (default: c2pa-research:latest)
  C2PA_DATA_DIR                    Output directory (default: .\c2pa-results)
  C2PA_GPU                         GPU flags (default: --gpus all)

OUTPUT:
  All results saved to: .\c2pa-results\
  - csv\final_metrics.csv          Main results
  - plots\                         Visualizations
  - logs\                          Execution logs

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
        Invoke-DockerCommand -UseGpu $true -Args (@($Command) + $Arguments)
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
