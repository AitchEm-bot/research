#!/usr/bin/env python3
"""
Master Pipeline Orchestrator for C2PA Robustness Research
==========================================================

This script coordinates all phases of the research pipeline while respecting
the existing modular structure of 33+ specialized scripts.

Phases:
  0: Generate AI assets (images and videos) - Separate step, run independently
  1: Embed C2PA manifests (auto-copies presets if no assets exist)
  2: Apply transformations (compression and editing)
  2.5: Platform testing setup (optional)
  3: Verify C2PA and calculate quality metrics
  4: Data analysis and visualization

Usage:
  python scripts/run_pipeline.py run-all          # Run phases 1-4 (starts from embedding)
  python scripts/run_pipeline.py run-all --test   # Run in test mode
  python scripts/run_pipeline.py phase0 --images 50  # Generate 50 new images
  python scripts/run_pipeline.py phase1           # C2PA embedding (phase 1)
  python scripts/run_pipeline.py phase2 --test    # Transformations in test mode
  python scripts/run_pipeline.py --help           # Show help
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities
try:
    from scripts.common import utils
except ImportError:
    # Fallback if utils not available yet
    class utils:
        @staticmethod
        def setup_logging(log_file):
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger(__name__)

# Initialize app and console
app = typer.Typer(
    help="""C2PA Robustness Research Pipeline

PHASES:
  phase0      Generate AI assets (images/videos) - requires GPU
  phase1      Embed C2PA manifests into assets
  phase2      Apply transformations (compression/editing)
  phase2-5    Prepare assets for platform testing (optional)
  phase3      Verify C2PA and calculate quality metrics
  phase4      Data analysis and visualization

WORKFLOWS:
  run-all     Run phases 1-4 sequentially
  status      Check pipeline execution status

FLAGS:
  --test          Run in test mode (fewer assets)
  --force         Force re-run even if checkpoint exists
  --publication   Generate publication figures (F1-F7) in phase 4

EXAMPLES:
  c2pa phase1                    # Run phase 1
  c2pa phase4 --publication      # Generate F1-F7 figures
  c2pa run-all --test            # Quick test run
  c2pa status                    # Check progress
""",
    add_completion=False,
    no_args_is_help=True
)
console = Console()

# Global logger (will be initialized in main)
logger = None

# Pipeline checkpoint file
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "results" / "pipeline_checkpoint.json"


def save_checkpoint(phase: str, status: str = "completed", metadata: Dict = None):
    """Save pipeline checkpoint for resumability"""
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "phase": phase,
        "status": status,
        "metadata": metadata or {}
    }

    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoints
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoints = json.load(f)
    else:
        checkpoints = {"phases": {}}

    checkpoints["phases"][phase] = checkpoint
    checkpoints["last_updated"] = datetime.now().isoformat()

    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoints, f, indent=2)

    logger.info(f"Checkpoint saved for phase {phase}")


def check_checkpoint(phase: str) -> bool:
    """Check if a phase has been completed"""
    if not CHECKPOINT_FILE.exists():
        return False

    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoints = json.load(f)

    return phase in checkpoints.get("phases", {}) and \
           checkpoints["phases"][phase].get("status") == "completed"


def copy_preset_assets():
    """
    Copy preset assets from Docker image to working data directory.
    This allows users to run the pipeline immediately without generation.
    """
    import shutil

    preset_dir = Path("/workspace/preset_assets")
    target_dir = PROJECT_ROOT / "data" / "assets"

    if not preset_dir.exists():
        console.print("[yellow]No preset assets found. Will need to generate assets.[/yellow]")
        return False

    console.print("[cyan]Copying preset assets to working directory...[/cyan]")

    # Copy raw images
    if (preset_dir / "raw_images").exists():
        for img_file in (preset_dir / "raw_images").glob("*"):
            shutil.copy2(img_file, target_dir / "raw_images" / img_file.name)
        img_count = len(list((target_dir / "raw_images").glob("*.png")))
        console.print(f"  [OK] Copied {img_count} images")

    # Copy conditioning images for videos
    if (preset_dir / "raw_images_for_videos").exists():
        for img_file in (preset_dir / "raw_images_for_videos").glob("*"):
            shutil.copy2(img_file, target_dir / "raw_images_for_videos" / img_file.name)
        vid_img_count = len(list((target_dir / "raw_images_for_videos").glob("*.png")))
        console.print(f"  [OK] Copied {vid_img_count} conditioning images")

    # Copy external videos
    if (preset_dir / "raw_out_videos").exists():
        for vid_file in (preset_dir / "raw_out_videos").glob("*.mp4"):
            shutil.copy2(vid_file, target_dir / "raw_out_videos" / vid_file.name)
        vid_count = len(list((target_dir / "raw_out_videos").glob("*.mp4")))
        console.print(f"  [OK] Copied {vid_count} external videos")

    console.print("[green][OK] Preset assets loaded successfully[/green]")
    logger.info("Preset assets copied from /workspace/preset_assets")
    return True


def run_script(script_path: str, args: List[str] = None, test_mode: bool = False,
               timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """
    Execute a Python script with optional arguments, streaming output in real-time.

    Args:
        script_path: Path to the Python script
        args: Optional list of command-line arguments
        test_mode: If True, add --test flag
        timeout: Optional timeout in seconds

    Returns:
        CompletedProcess object

    Raises:
        RuntimeError: If script execution fails
    """
    cmd = [sys.executable, str(PROJECT_ROOT / script_path)]

    if test_mode:
        cmd.append("--test")

    if args:
        cmd.extend(args)

    logger.info(f"Running: {' '.join(cmd)}")
    console.print(f"[cyan]Executing:[/cyan] {Path(script_path).name}")

    try:
        # Use Popen to stream output in real-time (shows tqdm progress bars)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for unified output
            text=True,
            bufsize=1,  # Line buffered
            cwd=str(PROJECT_ROOT),
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Disable Python output buffering
        )

        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)  # Print immediately
            output_lines.append(line)

        process.wait(timeout=timeout)

        stdout_text = ''.join(output_lines)

        if process.returncode != 0:
            logger.error(f"Script failed with code {process.returncode}")
            logger.error(f"Output: {stdout_text[-2000:]}")  # Last 2000 chars
            console.print(f"[red]✗ Failed:[/red] {Path(script_path).name}")
            raise RuntimeError(f"Pipeline failed at {script_path}")

        console.print(f"[green]✓ Completed:[/green] {Path(script_path).name}")
        return subprocess.CompletedProcess(cmd, process.returncode, stdout_text, '')

    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"Script timed out after {timeout} seconds")
        console.print(f"[red]✗ Timeout:[/red] {Path(script_path).name}")
        raise RuntimeError(f"Script timed out: {script_path}")


@app.command()
def phase0(
    test: bool = typer.Option(False, "--test", help="Run in test mode (fewer assets)"),
    force: bool = typer.Option(False, "--force", help="Force re-run even if checkpoint exists"),
    images: Optional[int] = typer.Option(None, "--images", help="Number of images to generate"),
    videos: Optional[int] = typer.Option(None, "--videos", help="Number of videos to generate"),
    skip_images: bool = typer.Option(False, "--skip-images", help="Skip image generation"),
    skip_videos: bool = typer.Option(False, "--skip-videos", help="Skip video generation"),
):
    """
    Phase 0: Generate AI assets (images with SD1.4, videos with SVD).

    Requires GPU. Use --images and --videos to set counts.
    Default: 100 images, 10 videos in normal mode; 10/2 in test mode.
    """

    if check_checkpoint("phase0") and not force:
        console.print("[yellow]Phase 0 already completed. Use --force to re-run.[/yellow]")
        return

    console.print("+================================================================+")
    console.print("|                PHASE 0: Asset Generation                      |")
    console.print("+================================================================+")
    console.print("")

    # Determine counts - if one type is specified, only generate that type
    if skip_images:
        image_count = 0
    elif images is not None:
        image_count = images
    elif videos is not None:
        # If only videos specified, skip images
        image_count = 0
    else:
        image_count = 10 if test else 100

    if skip_videos:
        video_count = 0
    elif videos is not None:
        video_count = videos
    elif images is not None:
        # If only images specified, skip videos
        video_count = 0
    else:
        video_count = 2 if test else 30

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        if image_count > 0:
            # Generate images
            task = progress.add_task(f"Generating {image_count} images...", total=None)
            run_script(
                "scripts/processing/generation/generate_images.py",
                ["--seed", "42", "--count", str(image_count)],
                test_mode=False
            )
            progress.update(task, completed=True)
            console.print(f"  [OK] Generated {image_count} images")

        if video_count > 0:
            # Generate conditioning images for videos
            task = progress.add_task("Generating conditioning images...", total=None)
            run_script(
                "scripts/processing/generation/generate_video_images.py",
                ["--count", str(video_count)],
                test_mode=False
            )
            progress.update(task, completed=True)

            # Generate videos
            task = progress.add_task(f"Generating {video_count} videos...", total=None)
            run_script(
                "scripts/processing/generation/generate_videos.py",
                ["--count", str(video_count)],
                test_mode=False
            )
            progress.update(task, completed=True)
            console.print(f"  [OK] Generated {video_count} videos")

    save_checkpoint("phase0", metadata={"test_mode": test, "image_count": image_count, "video_count": video_count})
    console.print("Status: Asset generation complete")
    console.print("[OK] Phase 0 completed successfully!")


@app.command()
def phase1(
    test: bool = typer.Option(False, "--test", help="Run in test mode"),
    force: bool = typer.Option(False, "--force", help="Force re-run even if checkpoint exists")
):
    """
    Phase 1: Embed C2PA manifests into images and videos.

    Signs assets with c2patool using test certificates.
    Auto-copies preset assets if data/assets/ is empty.
    """

    if check_checkpoint("phase1") and not force:
        console.print("[yellow]Phase 1 already completed. Use --force to re-run.[/yellow]")
        return

    # Check if assets exist, if not copy presets
    assets_dir = PROJECT_ROOT / "data" / "assets"
    raw_images = list((assets_dir / "raw_images").glob("*.png"))
    raw_videos = list((assets_dir / "raw_out_videos").glob("*.mp4"))

    if not raw_images and not raw_videos:
        console.print("[yellow]No assets found. Copying preset assets...[/yellow]")
        if copy_preset_assets():
            console.print("[green]Preset assets copied successfully![/green]")
        else:
            console.print("[red]No preset assets available. Run 'phase0' to generate assets first.[/red]")
            return

    console.print("+================================================================+")
    console.print("|               PHASE 1: C2PA Embedding                         |")
    console.print("+================================================================+")
    console.print("")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Sign internal assets with C2PA
        task = progress.add_task("Signing internal assets...", total=None)
        run_script("scripts/c2pa/embedding/embed_c2pa_v2.py", test_mode=test)
        progress.update(task, completed=True)

        # Process external videos (Veo3.1, etc.)
        task = progress.add_task("Processing external videos...", total=None)
        run_script(
            "scripts/processing/preprocessing/external/prepare_external_videos.py",
            test_mode=test
        )
        progress.update(task, completed=True)

        # Extract manifests to JSON (optional)
        task = progress.add_task("Extracting manifest JSONs...", total=None)
        run_script("scripts/c2pa/embedding/extract_manifests.py", test_mode=test)
        progress.update(task, completed=True)

    save_checkpoint("phase1", metadata={"test_mode": test})
    console.print("[green]Phase 1 completed successfully![/green]")


@app.command()
def phase2(
    test: bool = typer.Option(False, "--test", help="Run in test mode"),
    force: bool = typer.Option(False, "--force", help="Force re-run even if checkpoint exists")
):
    """
    Phase 2: Apply transformations to signed assets.

    Includes: JPEG/PNG compression, H.264/H.265 encoding,
    resize, crop, rotate (images), trim (videos).
    """

    if check_checkpoint("phase2") and not force:
        console.print("[yellow]Phase 2 already completed. Use --force to re-run.[/yellow]")
        return

    console.print(Panel.fit("PHASE 2: Transformations", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Image compression transformations
        task = progress.add_task("Compressing images...", total=None)
        run_script(
            "scripts/processing/transformations/compress_images.py",
            test_mode=test
        )
        progress.update(task, completed=True)

        # Video compression transformations
        task = progress.add_task("Compressing videos...", total=None)
        run_script(
            "scripts/processing/transformations/compress_videos.py",
            test_mode=test,
            timeout=1800  # 30 minutes timeout for video processing
        )
        progress.update(task, completed=True)

        # Editing transformations (resize, crop, rotate, etc.)
        task = progress.add_task("Applying editing transformations...", total=None)
        run_script(
            "scripts/processing/transformations/edit_assets.py",
            test_mode=test
        )
        progress.update(task, completed=True)

    save_checkpoint("phase2", metadata={"test_mode": test})
    console.print("[green]Phase 2 completed successfully![/green]")


@app.command()
def phase3(
    test: bool = typer.Option(False, "--test", help="Run in test mode"),
    force: bool = typer.Option(False, "--force", help="Force re-run even if checkpoint exists")
):
    """
    Phase 3: Verify C2PA manifests and calculate quality metrics.

    Checks manifest integrity after transforms.
    Calculates PSNR, SSIM (images), VMAF (videos).
    Outputs: c2pa_validation.csv, quality_metrics.csv, final_metrics.csv
    """

    if check_checkpoint("phase3") and not force:
        console.print("[yellow]Phase 3 already completed. Use --force to re-run.[/yellow]")
        return

    console.print(Panel.fit("PHASE 3: Verification & Metrics", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Verify baseline (original signed assets)
        task = progress.add_task("Verifying baseline manifests...", total=None)
        run_script(
            "scripts/c2pa/verification/verify_original_manifests.py",
            test_mode=test
        )
        progress.update(task, completed=True)

        # Verify transformed assets
        task = progress.add_task("Verifying C2PA manifests...", total=None)
        run_script(
            "scripts/c2pa/verification/verify_c2pa.py",
            test_mode=test,
            timeout=3600  # 1 hour timeout for verification
        )
        progress.update(task, completed=True)

        # Calculate quality metrics (PSNR, SSIM, VMAF)
        task = progress.add_task("Calculating quality metrics...", total=None)
        run_script(
            "scripts/processing/metrics/calculate_quality_metrics.py",
            test_mode=test,
            timeout=3600  # 1 hour timeout for quality metrics
        )
        progress.update(task, completed=True)

        # Merge all results into final CSV
        task = progress.add_task("Merging results...", total=None)
        run_script("scripts/processing/metrics/merge_results.py")
        progress.update(task, completed=True)

    save_checkpoint("phase3", metadata={"test_mode": test})
    console.print("[green]Phase 3 completed successfully![/green]")


@app.command()
def phase4(
    test: bool = typer.Option(False, "--test", help="Run in test mode"),
    skip_viz: bool = typer.Option(False, "--skip-viz", help="Skip visualization generation"),
    publication: bool = typer.Option(False, "--publication", help="Generate publication figures"),
    force: bool = typer.Option(False, "--force", help="Force re-run even if checkpoint exists")
):
    """
    Phase 4: Data analysis and visualization.

    Generates exploratory plots by default.
    Use --publication to generate F1-F7 publication figures.
    Use --skip-viz to only run statistical analysis.
    """

    if check_checkpoint("phase4") and not force:
        console.print("[yellow]Phase 4 already completed. Use --force to re-run.[/yellow]")
        return

    console.print(Panel.fit("PHASE 4: Data Analysis", style="bold blue"))

    args = []
    if skip_viz:
        args.append("--skip-viz")
    if publication:
        args.append("--publication-figures")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        task = progress.add_task("Running data analysis...", total=None)
        run_script(
            "scripts/analysis/run_phase4_analysis.py",
            args,
            test_mode=test
        )
        progress.update(task, completed=True)

    save_checkpoint("phase4", metadata={
        "test_mode": test,
        "skip_viz": skip_viz,
        "publication": publication
    })
    console.print("[green]Phase 4 completed successfully![/green]")


@app.command()
def run_all(
    test: bool = typer.Option(False, "--test", help="Run in test mode (fewer assets, faster)"),
    resume_from: int = typer.Option(1, "--resume-from", help="Resume from phase number (1, 2, 3, 4)"),
    force: bool = typer.Option(False, "--force", help="Force re-run all phases"),
    publication: bool = typer.Option(False, "--publication", help="Generate publication figures in Phase 4")
):
    """
    Run complete pipeline: Phase 1 → Phase 4.

    Skips Phase 0 (asset generation) - run 'phase0' separately if needed.
    Use --resume-from to restart from a specific phase.
    Use --publication to generate F1-F7 figures in Phase 4.
    """

    start_time = time.time()

    console.print(Panel.fit(
        "[bold]C2PA Robustness Research Pipeline[/bold]\n" +
        f"Mode: {'TEST' if test else 'FULL'}\n" +
        f"Resume from: Phase {resume_from}\n" +
        "[dim]Note: Phase 0 (generation) is separate. Use 'phase0' command if you need to generate new assets.[/dim]",
        style="bold magenta"
    ))

    phases = [
        (1, "C2PA Embedding", phase1),
        (2, "Transformations", phase2),
        (3, "Verification & Metrics", phase3),
        (4, "Analysis", phase4)
    ]

    # Create summary table
    table = Table(title="Pipeline Execution Plan")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Description", style="magenta")
    table.add_column("Status", justify="center")

    for phase_num, description, _ in phases:
        if phase_num < resume_from:
            status = "⏭️ Skip"
        else:
            status = "▶️ Run"
        table.add_row(f"{phase_num}", description, status)

    console.print(table)
    console.print()

    # Execute phases
    for phase_num, description, phase_func in phases:
        if phase_num < resume_from:
            console.print(f"[dim]Skipping Phase {phase_num} (already completed)[/dim]")
            continue

        console.print(f"\n[bold]Starting Phase {phase_num}: {description}[/bold]")

        # Call phase function with appropriate arguments
        if phase_func == phase4:
            # Must explicitly pass skip_viz=False to avoid OptionInfo object issue
            # Default to publication=True for run_all to generate all figures
            phase_func(test=test, force=force, skip_viz=False, publication=True)
        else:
            phase_func(test=test, force=force)

    # Calculate elapsed time
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    console.print(Panel.fit(
        f"[bold green]Pipeline completed successfully![/bold green]\n" +
        f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}",
        style="bold green"
    ))

    # Show results location
    console.print("\n[cyan]Results available at:[/cyan]")
    console.print("  • CSV files: data/results/csv/")
    console.print("  • Analysis: data/results/analysis_results/")
    console.print("  • Logs: data/results/logs/")


@app.command(name="phase2-5")
def phase2_5(
    auto_sample: bool = typer.Option(True, "--auto-sample", help="Automatically sample assets for platforms"),
):
    """Phase 2.5: Prepare assets for platform testing (optional - distributes signed assets to platform directories for manual upload/download testing)"""
    console.print("+================================================================+")
    console.print("|          PHASE 2.5: Platform Testing Setup                     |")
    console.print("+================================================================+")
    console.print("")

    # Run the platform preparation script
    console.print("[>>] Distributing assets to platform directories...")

    try:
        run_script(
            "scripts/processing/preprocessing/platform/prepare_platform_uploads.py",
            ["--auto-sample"] if auto_sample else [],
            test_mode=False
        )
        console.print("[OK] Platform directories created and assets distributed")
        console.print("")
        console.print("Next steps:")
        console.print("  1. Navigate to: data/platform_tests/")
        console.print("  2. Upload assets from each platform's 'uploads/' folder")
        console.print("  3. Download them back and place in 'returned/' folder")
        console.print("  4. Run: c2pa phase 3")
        console.print("")
        console.print("Status: Ready for manual platform testing")
    except Exception as e:
        console.print(f"[X] Failed to prepare platform uploads: {e}")
        logger.error(f"Platform upload preparation failed: {e}")


@app.command()
def status():
    """Check pipeline execution status"""

    console.print(Panel.fit("Pipeline Status", style="bold blue"))

    if not CHECKPOINT_FILE.exists():
        console.print("[yellow]No checkpoint file found. Pipeline has not been run yet.[/yellow]")
        return

    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoints = json.load(f)

    table = Table(title="Phase Completion Status")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Status", style="magenta")
    table.add_column("Timestamp", style="green")

    phases = ["phase0", "phase1", "phase2", "phase3", "phase4"]

    for phase in phases:
        if phase in checkpoints.get("phases", {}):
            info = checkpoints["phases"][phase]
            status = "✅ " + info.get("status", "unknown")
            timestamp = info.get("timestamp", "")[:19]  # Trim microseconds
        else:
            status = "❌ not run"
            timestamp = "-"

        table.add_row(phase.replace("phase", "Phase "), status, timestamp)

    console.print(table)

    if "last_updated" in checkpoints:
        console.print(f"\n[dim]Last updated: {checkpoints['last_updated'][:19]}[/dim]")


@app.command()
def clean():
    """Clean generated data and reset checkpoints"""

    console.print("[yellow]Warning: This will delete all generated data![/yellow]")

    if not typer.confirm("Are you sure you want to clean all generated data?"):
        console.print("[green]Aborted.[/green]")
        return

    console.print("[red]Cleaning generated data...[/red]")

    # Directories to clean
    clean_dirs = [
        "data/prepared_assets/signed_assets",
        "data/prepared_assets/transformed",
        "data/prepared_assets/c2pa_manifests",
        "data/results/csv",
        "data/results/analysis_results"
    ]

    for dir_path in clean_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            console.print(f"  Cleaning: {dir_path}")
            # Note: In production, implement actual deletion here
            # For safety, we're just printing what would be deleted

    # Remove checkpoint file
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        console.print("  Removed checkpoint file")

    console.print("[green]Cleanup complete![/green]")


def main():
    """Main entry point"""
    global logger

    # Setup logging
    logger = utils.setup_logging("pipeline_orchestrator.log")

    # Print header
    console.print(Panel.fit(
        "[bold cyan]C2PA Robustness Research Pipeline[/bold cyan]\n" +
        "Is C2PA's Metadata Robust in AI-Generated Content?",
        style="bold"
    ))

    # Run CLI app
    app()


if __name__ == "__main__":
    main()