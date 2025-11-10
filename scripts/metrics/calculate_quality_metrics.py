"""
Quality Metrics Calculation Script for C2PA Robustness Testing
===============================================================

This script calculates image and video quality metrics for all transformed assets.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio) - Images only
- SSIM (Structural Similarity Index) - Images only
- VMAF (Video Multimethod Assessment Fusion) - Videos only

Features:
- opencv-python for fast PSNR/SSIM calculation
- ffmpeg subprocess for VMAF scores
- Parallel processing with ProcessPoolExecutor
- Runtime tracking per asset
- Error resilience with detailed logging

Usage:
    python scripts/metrics/calculate_quality_metrics.py [--test]

    --test: Process only one asset from each category (smoke test)

Output:
    data/metrics/quality_metrics.csv
"""

import cv2
import logging
import numpy as np
import subprocess
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

from tqdm import tqdm

# Ensure log directory exists
Path("data/metrics").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/metrics/quality_metrics.log')
    ]
)

# Configuration
TRANSFORMED_BASE_DIR = Path("data/transformed")
MANIFESTS_DIR = Path("data/manifests")
OUTPUT_CSV = Path("data/metrics/quality_metrics.csv")

# CSV Column headers
CSV_HEADERS = [
    'filename',
    'asset_type',
    'psnr',
    'ssim',
    'vmaf',
    'processing_time_ms',
    'calculation_error',
    'timestamp'
]


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate SSIM using OpenCV-compatible implementation.

    Based on Wang et al. "Image quality assessment: from error visibility to
    structural similarity" (2004).

    Args:
        img1: First grayscale image
        img2: Second grayscale image

    Returns:
        SSIM score (0-1, higher is better)
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Compute means
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))


def find_original_asset(transformed_path: Path) -> Optional[Path]:
    """
    Find the original signed asset corresponding to transformed asset.

    Examples:
        img_000_seed42_20251109_220519_jpeg_q95.jpg
        → img_000_seed42_20251109_220519_signed.png

        video_000_seed100_20251109_231519_h264_bitrate5000k.mp4
        → video_000_seed100_20251109_231519_signed.mp4

    Args:
        transformed_path: Path to transformed asset

    Returns:
        Path to original signed asset, or None if not found
    """
    filename = transformed_path.name

    # Extract the base identifier (everything before transformation suffix)
    # Remove transform suffixes
    base = filename

    # Remove file extension first
    base = Path(base).stem

    # Remove compression suffixes
    base = re.sub(r'_(jpeg|png|h264|h265)_.*$', '', base)
    # Remove FPS suffixes
    base = re.sub(r'_fps\d+$', '', base)
    # Remove editing suffixes
    base = re.sub(r'_(crop|resize|rotate|brightness|contrast|saturation).*$', '', base)

    # Reconstruct original filename
    # Images: .png, Videos: .mp4
    if transformed_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        original_filename = f"{base}_signed.png"
    else:
        original_filename = f"{base}_signed.mp4"

    original_path = MANIFESTS_DIR / original_filename

    if original_path.exists():
        return original_path
    else:
        logging.warning(f"Original not found for {filename}: expected {original_filename}")
        return None


def calculate_image_metrics(original_path: Path, transformed_path: Path) -> Tuple[Optional[float], Optional[float], Optional[str], float]:
    """
    Calculate PSNR and SSIM for image pair.

    Args:
        original_path: Path to original image
        transformed_path: Path to transformed image

    Returns:
        Tuple of (psnr, ssim, error_message, processing_time_ms)
    """
    start_time = time.time()

    try:
        # Read images
        img1 = cv2.imread(str(original_path))
        img2 = cv2.imread(str(transformed_path))

        if img1 is None:
            error_msg = f"Failed to read original: {original_path.name}"
            elapsed_ms = (time.time() - start_time) * 1000
            return None, None, error_msg, elapsed_ms

        if img2 is None:
            error_msg = f"Failed to read transformed: {transformed_path.name}"
            elapsed_ms = (time.time() - start_time) * 1000
            return None, None, error_msg, elapsed_ms

        # Resize transformed to match original if dimensions differ
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)

        # Calculate PSNR
        psnr = cv2.PSNR(img1, img2)

        # Calculate SSIM (convert to grayscale)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ssim = calculate_ssim(gray1, gray2)

        elapsed_ms = (time.time() - start_time) * 1000

        return psnr, ssim, None, elapsed_ms

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = f"Image metrics error: {str(e)}"
        logging.error(f"{transformed_path.name}: {error_msg}")
        return None, None, error_msg, elapsed_ms


def calculate_video_vmaf(original_path: Path, transformed_path: Path) -> Tuple[Optional[float], Optional[str], float]:
    """
    Calculate VMAF score for video pair using ffmpeg.

    Args:
        original_path: Path to original video
        transformed_path: Path to transformed video

    Returns:
        Tuple of (vmaf_score, error_message, processing_time_ms)
    """
    start_time = time.time()

    try:
        # ffmpeg command with libvmaf filter
        cmd = [
            'ffmpeg',
            '-i', str(transformed_path),  # Distorted video
            '-i', str(original_path),      # Reference video
            '-lavfi', '[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf=log_fmt=json:log_path=NUL',
            '-f', 'null', '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=False   # Don't raise on non-zero exit
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Parse VMAF score from stderr (ffmpeg outputs to stderr)
        # Look for pattern: "VMAF score: XX.XXXX"
        match = re.search(r'VMAF score:\s*(\d+\.\d+)', result.stderr)

        if match:
            vmaf_score = float(match.group(1))
            return vmaf_score, None, elapsed_ms
        else:
            # Try alternate pattern from libvmaf output
            match = re.search(r'"vmaf":\s*(\d+\.\d+)', result.stderr)
            if match:
                vmaf_score = float(match.group(1))
                return vmaf_score, None, elapsed_ms
            else:
                error_msg = "VMAF score not found in ffmpeg output"
                logging.error(f"{transformed_path.name}: {error_msg}")
                return None, error_msg, elapsed_ms

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = "VMAF calculation timeout"
        logging.error(f"{transformed_path.name}: {error_msg}")
        return None, error_msg, elapsed_ms
    except FileNotFoundError:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = "ffmpeg not found or libvmaf not available"
        logging.error(f"{transformed_path.name}: {error_msg}")
        return None, error_msg, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = f"VMAF error: {str(e)}"
        logging.error(f"{transformed_path.name}: {error_msg}")
        return None, error_msg, elapsed_ms


def process_single_asset(transformed_path: Path) -> Dict:
    """
    Calculate metrics for single asset.

    Args:
        transformed_path: Path to transformed asset

    Returns:
        Dict with metrics row data
    """
    # Find original asset
    original_path = find_original_asset(transformed_path)

    if original_path is None:
        return {
            'filename': transformed_path.name,
            'asset_type': 'image' if transformed_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else 'video',
            'psnr': '',
            'ssim': '',
            'vmaf': '',
            'processing_time_ms': '0.00',
            'calculation_error': 'original_not_found',
            'timestamp': datetime.now().isoformat()
        }

    # Determine asset type and calculate appropriate metrics
    asset_type = 'image' if transformed_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else 'video'

    if asset_type == 'image':
        psnr, ssim, error, proc_time = calculate_image_metrics(original_path, transformed_path)

        return {
            'filename': transformed_path.name,
            'asset_type': 'image',
            'psnr': f"{psnr:.4f}" if psnr is not None else '',
            'ssim': f"{ssim:.6f}" if ssim is not None else '',
            'vmaf': '',  # Not applicable for images
            'processing_time_ms': f"{proc_time:.2f}",
            'calculation_error': error if error else '',
            'timestamp': datetime.now().isoformat()
        }
    else:
        vmaf, error, proc_time = calculate_video_vmaf(original_path, transformed_path)

        return {
            'filename': transformed_path.name,
            'asset_type': 'video',
            'psnr': '',  # Not applicable for videos
            'ssim': '',  # Not applicable for videos
            'vmaf': f"{vmaf:.4f}" if vmaf is not None else '',
            'processing_time_ms': f"{proc_time:.2f}",
            'calculation_error': error if error else '',
            'timestamp': datetime.now().isoformat()
        }


def collect_transformed_assets(test_mode: bool = False) -> list:
    """
    Collect all transformed assets for metric calculation.

    Args:
        test_mode: If True, return only one asset from each category

    Returns:
        List of asset paths
    """
    assets = []

    # Collect all transformed assets
    assets.extend(TRANSFORMED_BASE_DIR.glob("compression/images/**/*.png"))
    assets.extend(TRANSFORMED_BASE_DIR.glob("compression/images/**/*.jpg"))
    assets.extend(TRANSFORMED_BASE_DIR.glob("compression/videos/**/*.mp4"))
    assets.extend(TRANSFORMED_BASE_DIR.glob("editing/images/**/*.png"))
    assets.extend(TRANSFORMED_BASE_DIR.glob("editing/videos/**/*.mp4"))

    assets = sorted(assets)

    if test_mode:
        # Select test assets from different categories
        test_assets = []
        categories = {
            'compression_image': None,
            'compression_video': None,
            'editing_image': None,
            'editing_video': None
        }

        for asset in assets:
            parts = asset.parts
            if 'compression' in parts and 'images' in parts and not categories['compression_image']:
                categories['compression_image'] = asset
            elif 'compression' in parts and 'videos' in parts and not categories['compression_video']:
                categories['compression_video'] = asset
            elif 'editing' in parts and 'images' in parts and not categories['editing_image']:
                categories['editing_image'] = asset
            elif 'editing' in parts and 'videos' in parts and not categories['editing_video']:
                categories['editing_video'] = asset

        test_assets = [v for v in categories.values() if v is not None]
        logging.info(f"TEST MODE: Selected {len(test_assets)} assets")
        return test_assets

    return assets


def process_assets_parallel(assets: list, max_workers: int = 4):
    """
    Process assets in parallel using ProcessPoolExecutor.

    Args:
        assets: List of asset paths
        max_workers: Number of parallel workers
    """
    logging.info(f"Processing {len(assets)} assets with {max_workers} workers")

    # Write CSV header
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_asset, asset): asset for asset in assets}

        # Process results as they complete
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating metrics", unit="asset"):
                try:
                    row_data = future.result()
                    writer.writerow(row_data)
                    csvfile.flush()
                except Exception as e:
                    asset = futures[future]
                    logging.error(f"Failed to process {asset.name}: {e}")


def process_assets_sequential(assets: list):
    """
    Process assets sequentially (for debugging or test mode).

    Args:
        assets: List of asset paths
    """
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for asset in tqdm(assets, desc="Calculating metrics", unit="asset"):
            try:
                row_data = process_single_asset(asset)
                writer.writerow(row_data)
                csvfile.flush()
            except Exception as e:
                logging.error(f"Failed to process {asset.name}: {e}")


def main():
    """Main entry point."""
    logging.info("=" * 60)
    logging.info("Quality Metrics Calculation Script")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"OpenCV version: {cv2.__version__}")

    # Check for ffmpeg (optional, only needed for videos)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        ffmpeg_version = result.stdout.split('\n')[0]
        logging.info(f"ffmpeg: {ffmpeg_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("ffmpeg not found - video metrics (VMAF) will be skipped")

    logging.info("=" * 60)

    # Parse arguments
    test_mode = '--test' in sys.argv

    if test_mode:
        logging.info("TEST MODE: Processing one asset from each category")

    # Collect assets
    assets = collect_transformed_assets(test_mode=test_mode)

    if not assets:
        logging.error(f"No transformed assets found in {TRANSFORMED_BASE_DIR}")
        return

    # Process assets
    if test_mode:
        # Sequential processing for test mode (easier debugging)
        process_assets_sequential(assets)
    else:
        # Parallel processing for full run
        process_assets_parallel(assets, max_workers=4)

    logging.info("=" * 60)
    logging.info("Quality Metrics Calculation Complete")
    logging.info(f"  Processed: {len(assets)} assets")
    logging.info(f"  Output: {OUTPUT_CSV.absolute()}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
