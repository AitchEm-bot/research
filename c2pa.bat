@echo off
REM C2PA Robustness Research Pipeline - Docker Wrapper (Windows Batch)
REM Phase-based execution wrapper for Windows CMD

setlocal enabledelayedexpansion

REM Configuration
if "%C2PA_IMAGE%"=="" (
    set IMAGE=aitchem037/c2pa-research:latest
) else (
    set IMAGE=%C2PA_IMAGE%
)

if "%C2PA_DATA_DIR%"=="" (
    set OUTPUT_DIR=%cd%\c2pa-results
) else (
    set OUTPUT_DIR=%C2PA_DATA_DIR%
)

if "%C2PA_GPU%"=="" (
    set GPU_FLAG=--gpus all
) else (
    set GPU_FLAG=%C2PA_GPU%
)

if "%C2PA_TOOLS_DIR%"=="" (
    set TOOLS_DIR=%cd%\tools
) else (
    set TOOLS_DIR=%C2PA_TOOLS_DIR%
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Create model cache volumes
docker volume create huggingface-cache >nul 2>&1
docker volume create torch-cache >nul 2>&1

REM Main command router
if "%1"=="" (
    echo Error: No command specified
    echo Run 'c2pa --help' for usage information
    exit /b 1
)

if "%1"=="phase" goto PHASE
if "%1"=="run" goto RUN
if "%1"=="test" goto TEST
if "%1"=="status" goto STATUS
if "%1"=="shell" goto SHELL
if "%1"=="--help" goto HELP
if "%1"=="-h" goto HELP
if "%1"=="help" goto HELP
goto FORWARD

:PHASE
if "%2"=="" (
    echo Error: Phase number not specified
    exit /b 1
)

if "%2"=="0" goto PHASE0
if "%2"=="1" goto PHASE1
if "%2"=="2" goto PHASE2
if "%2"=="2.5" goto PHASE25
if "%2"=="3" goto PHASE3
if "%2"=="4" goto PHASE4

echo Error: Invalid phase number: %2
echo.
echo Valid phases:
echo   0    - Asset generation
echo   1    - C2PA manifest embedding
echo   2    - Transformation pipeline
echo   2.5  - Platform testing setup (optional)
echo   3    - Verification ^& quality metrics
echo   4    - Analysis ^& visualization
echo.
echo Usage: c2pa phase ^<number^> [options]
exit /b 1

:PHASE0
echo [Phase 0: Asset Generation]
shift
shift
docker run --rm %GPU_FLAG% ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% phase0 %1 %2 %3 %4 %5 %6 %7 %8 %9
goto END

:PHASE1
echo [Phase 1: C2PA Manifest Embedding]
shift
shift
docker run --rm ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% phase1 %1 %2 %3 %4 %5 %6 %7 %8 %9
goto END

:PHASE2
echo [Phase 2: Transformation Pipeline]
shift
shift
docker run --rm ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% phase2 %1 %2 %3 %4 %5 %6 %7 %8 %9
goto END

:PHASE25
echo [Phase 2.5: Platform Testing Setup]
echo Creating platform test directories...

REM Create platform directories
for %%p in (instagram twitter facebook youtube tiktok whatsapp) do (
    if not exist "%OUTPUT_DIR%\platform_tests\%%p\uploads" mkdir "%OUTPUT_DIR%\platform_tests\%%p\uploads"
    if not exist "%OUTPUT_DIR%\platform_tests\%%p\returned" mkdir "%OUTPUT_DIR%\platform_tests\%%p\returned"
)

REM Run platform upload preparation
shift
shift
docker run --rm ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% phase2-5 %1 %2 %3 %4 %5 %6 %7 %8 %9

REM Generate instructions
echo. > "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo ========================================>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo PLATFORM TESTING INSTRUCTIONS>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo ========================================>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo.>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo Assets have been distributed to platform folders.>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo.>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo WORKFLOW:>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo 1. Navigate to: c2pa-results\platform_tests\>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo 2. For each platform folder:>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo    a. Find assets in the 'uploads\' subfolder>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo    b. Upload these assets to the respective platform>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo    c. Download them back from the platform>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo    d. Place downloads in the 'returned\' subfolder>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo.>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo 3. After completing uploads/downloads for all platforms:>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo    Run: c2pa phase 3>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo.>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo PLATFORMS TO TEST:>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo - instagram\    (images + videos)>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo - twitter\      (images + videos)>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo - facebook\     (images + videos)>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo - youtube\      (videos only)>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo - tiktok\       (videos only)>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo - whatsapp\     (images + videos)>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo.>> "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"

echo.
echo  +================================================================+
echo ^|              PLATFORM TESTING SETUP COMPLETE                  ^|
echo  +================================================================+
echo.
type "%OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt"
echo.
echo Instructions saved to: %OUTPUT_DIR%\PLATFORM_UPLOAD_INSTRUCTIONS.txt
echo.
goto END

:PHASE3
echo [Phase 3: Verification ^& Quality Metrics]
shift
shift
docker run --rm %GPU_FLAG% ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% phase3 %1 %2 %3 %4 %5 %6 %7 %8 %9
goto END

:PHASE4
echo [Phase 4: Analysis ^& Visualization]
shift
shift
docker run --rm ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% phase4 %1 %2 %3 %4 %5 %6 %7 %8 %9
goto END

:RUN
echo [Full Pipeline: Phases 1-4, starts from embedding]
echo Note: Phase 0 (generation) is separate. Preset assets will be used.
shift
docker run --rm %GPU_FLAG% ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% run-all %1 %2 %3 %4 %5 %6 %7 %8 %9
goto END

:TEST
echo [Quick Test Mode: Phases 1-4 with preset assets]
docker run --rm %GPU_FLAG% ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% run-all --test
goto END

:STATUS
echo [Pipeline Status Check]
docker run --rm ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% status
goto END

:SHELL
echo [Interactive Shell]
docker run --rm -it %GPU_FLAG% ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% bash
goto END

:HELP
echo C2PA Robustness Research Pipeline - Docker Wrapper
echo ===================================================
echo.
echo USAGE:
echo   c2pa ^<command^> [options]
echo.
echo COMMANDS:
echo   phase 0         Generate AI assets (images with SD1.4, videos with SVD) - requires GPU
echo   phase 1         Embed C2PA manifests into assets using c2patool
echo   phase 2         Apply transformations (JPEG/PNG compression, H.264/H.265, resize, crop, rotate, trim)
echo   phase 2.5       Prepare assets for platform testing (optional, for social media round-trip tests)
echo   phase 3         Verify C2PA manifests and calculate quality metrics (PSNR, SSIM, VMAF)
echo   phase 4         Data analysis and visualization (exploratory plots or publication figures)
echo   run             Run complete pipeline (phases 1-4 sequentially)
echo   test            Quick test run with minimal assets
echo   status          Check pipeline execution status
echo   shell           Open interactive shell in container
echo.
echo OPTIONS:
echo   --test              Run in test mode (fewer assets, faster execution)
echo   --force             Force re-run even if checkpoint exists
echo   --publication       Generate F1-F7 publication figures (phase 4 and run)
echo   --skip-viz          Skip visualization generation (phase 4 only)
echo   --images N          Number of images to generate (phase 0 only)
echo   --videos N          Number of videos to generate (phase 0 only)
echo   --resume-from N     Resume from phase N, where N is 1-4 (run only)
echo.
echo EXAMPLES:
echo   c2pa phase 0 --images 50 --videos 10    # Generate 50 images + 10 videos
echo   c2pa phase 1                            # Sign assets with C2PA manifests
echo   c2pa phase 4 --publication              # Generate F1-F7 publication figures
echo   c2pa run --test                         # Quick test of full pipeline
echo   c2pa run --publication                  # Full run with publication figures
echo   c2pa run --resume-from 3                # Resume from phase 3
echo.
echo ENVIRONMENT VARIABLES:
echo   C2PA_IMAGE      Docker image (default: aitchem037/c2pa-research:latest)
echo   C2PA_DATA_DIR   Output directory (default: .\c2pa-results)
echo   C2PA_TOOLS_DIR  Tools directory (default: .\tools)
echo.
echo OUTPUT STRUCTURE:
echo   .\c2pa-results\
echo   +-- results\csv\                 CSV metrics files
echo       +-- final_metrics.csv        Main results
echo       +-- c2pa_validation.csv      C2PA verification
echo       +-- quality_metrics.csv      PSNR/SSIM/VMAF
echo   +-- results\analysis_results\
echo       +-- plots\                   Visualizations
echo       +-- csv\                     Analysis summaries
echo   +-- results\logs\                Execution logs
echo.
echo For detailed documentation, see README_DOCKER.md
goto END

:FORWARD
REM Forward unknown commands directly to Docker
docker run --rm %GPU_FLAG% ^
    -v "%OUTPUT_DIR%:/workspace/data" ^
    -v huggingface-cache:/workspace/.cache/huggingface ^
    -v torch-cache:/workspace/.cache/torch ^
    -v "%TOOLS_DIR%:/workspace/tools" ^
    %IMAGE% %*
goto END

:END
REM Show results summary if CSV exists
if exist "%OUTPUT_DIR%\csv\final_metrics.csv" (
    echo.
    echo  +================================================================+
    echo ^| Results available in: %OUTPUT_DIR%                            ^|
    echo ^| Metrics CSV: %OUTPUT_DIR%\csv\final_metrics.csv               ^|
    echo ^| Plots: %OUTPUT_DIR%\plots\                                    ^|
    echo  +================================================================+
)

endlocal
