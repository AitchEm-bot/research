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
    data/results/quality_metrics.csv
"""

import cv2
import numpy as np
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import utils

# Configure logging using shared utility
logger = utils.setup_logging(log_file='data/results/logs/quality_metrics.log')
utils.log_environment_info()

# Configuration - using shared constants
TRANSFORMED_BASE_DIR = utils.DIRS['transformed']
MANIFESTS_IMAGES_DIR = utils.DIRS['manifests_images']
MANIFESTS_VIDEOS_DIRS = [
    utils.DIRS['manifests_videos_internal'],
    utils.DIRS['manifests_videos_external']
]
OUTPUT_CSV = utils.DIRS['results'] / "quality_metrics.csv"

# Use shared lossless transforms definition
LOSSLESS_TRANSFORMS = utils.LOSSLESS_TRANSFORMS

# Use shared CSV headers
CSV_HEADERS = utils.CSV_HEADERS['quality_metrics']


# Use shared metadata extraction function
def extract_seed_and_model(filename: str, asset_type: str) -> tuple:
    """Extract seed and model_version from filename using shared utility."""
    metadata = utils.extract_metadata_from_filename(filename)

    # Handle external videos
    if asset_type == 'video' and not metadata['seed']:
        metadata['seed'] = 'NA'
        metadata['model_version'] = 'Veo3.1'
    elif asset_type == 'image':
        metadata['model_version'] = 'SD1.4'
    elif asset_type == 'video' and metadata['seed']:
        metadata['model_version'] = 'SVD'

    return metadata['seed'], metadata['model_version']


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


# Use shared find_original_asset function
def find_original_asset(transformed_path: Path) -> Optional[Path]:
    """Find the original signed asset using shared utility."""
    manifests_dirs = {
        'images': [MANIFESTS_IMAGES_DIR],
        'videos': MANIFESTS_VIDEOS_DIRS
    }
    return utils.find_original_asset(transformed_path, manifests_dirs)


def calculate_image_metrics(original_path: Path, transformed_path: Path, transform_type: Optional[str] = None) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float], str, int, Optional[str], float]:
    """
    Calculate PSNR and SSIM for image pair (both stretched and aligned versions).

    Computes two sets of metrics:
    1. Stretched: Scales transformed to match original dimensions (includes geometric distortion)
    2. Aligned: Center-crops original to match transformed aspect ratio (isolates content distortion)

    Handles lossless operations correctly:
    - PNG c0/c9 compression produces pixel-identical output (lossless DEFLATE)
    - Returns PSNR as "inf" string for perfect matches
    - Sets lossless_match = 1 when images are pixel-identical

    Args:
        original_path: Path to original signed image
        transformed_path: Path to transformed image
        transform_type: Optional transform type from filename metadata (e.g., 'resize', 'crop')

    Returns:
        Tuple of (psnr, ssim, psnr_aligned, ssim_aligned, alignment_method, lossless_match, error_message, processing_time_ms)
        - psnr: String ("inf" for lossless) or formatted float - stretched version
        - ssim: Float (0-1) - stretched version
        - psnr_aligned: String/float - aligned version (cropped reference)
        - ssim_aligned: Float (0-1) - aligned version (cropped reference)
        - alignment_method: String ('same_aspect_ratio', 'crop_reference_center_square', etc.)
        - lossless_match: Integer (1 if pixels identical, 0 otherwise)
        - error_message: String error description or None if successful
        - processing_time_ms: Float milliseconds elapsed
    """
    start_time = time.time()

    try:
        # Read images
        img_orig = cv2.imread(str(original_path))
        img_trans = cv2.imread(str(transformed_path))

        if img_orig is None:
            error_msg = f"Failed to read original: {original_path.name}"
            elapsed_ms = (time.time() - start_time) * 1000
            return None, None, None, None, '', 0, error_msg, elapsed_ms

        if img_trans is None:
            error_msg = f"Failed to read transformed: {transformed_path.name}"
            elapsed_ms = (time.time() - start_time) * 1000
            return None, None, None, None, '', 0, error_msg, elapsed_ms

        orig_h, orig_w = img_orig.shape[:2]
        trans_h, trans_w = img_trans.shape[:2]

        # Calculate aspect ratios
        orig_aspect = orig_w / orig_h
        trans_aspect = trans_w / trans_h
        aspect_changed = abs(orig_aspect - trans_aspect) > 0.01

        alignment_method = 'same_aspect_ratio'

        # ========== STRETCHED METRICS ==========
        # Resize transformed to match original dimensions
        if img_orig.shape != img_trans.shape:
            img_trans_stretched = cv2.resize(img_trans, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        else:
            img_trans_stretched = img_trans.copy()

        # Calculate PSNR (stretched)
        psnr_value = cv2.PSNR(img_orig, img_trans_stretched)

        # Detect lossless match based on PSNR threshold
        # PNG c0/c9 compression produces PSNR ~361 dB (mathematically lossless)
        # Use threshold of 100 dB to catch truly lossless operations
        lossless_match = 0
        if psnr_value >= 100.0:
            psnr = "inf"
            lossless_match = 1
            logger.debug(f"{transformed_path.name}: Lossless match detected (PSNR = {psnr_value:.2f} dB → inf)")
        else:
            psnr = f"{psnr_value:.4f}"

        # Calculate SSIM (stretched)
        gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray_trans_stretched = cv2.cvtColor(img_trans_stretched, cv2.COLOR_BGR2GRAY)
        ssim = calculate_ssim(gray_orig, gray_trans_stretched)

        # ========== ALIGNED METRICS ==========
        if aspect_changed:
            # Check if this is a resize transform (full frame scale) or actual crop
            is_resize_transform = transform_type == 'resize'
            is_rotation_transform = transform_type == 'rotate'

            if is_resize_transform or is_rotation_transform:
                # Resize and rotation transforms preserve full frame - no content is cropped
                # Use same metrics as stretched (no alignment needed)
                psnr_aligned = psnr
                ssim_aligned = ssim
                alignment_method = 'same_aspect_ratio'
                logger.debug(f"{transformed_path.name}: Using same_aspect_ratio for {transform_type} transform")
            elif orig_w > orig_h and trans_w == trans_h:
                # Horizontal to square: center-crop original width
                # This handles actual crop transforms or platform square crops
                crop_w = orig_h
                crop_x = (orig_w - crop_w) // 2
                img_orig_cropped = img_orig[:, crop_x:crop_x+crop_w]
                alignment_method = 'crop_reference_center_square'

                # Resize transformed to match cropped original
                img_trans_aligned = cv2.resize(img_trans, (img_orig_cropped.shape[1], img_orig_cropped.shape[0]), interpolation=cv2.INTER_CUBIC)

                # Calculate aligned PSNR
                psnr_aligned_value = cv2.PSNR(img_orig_cropped, img_trans_aligned)
                if psnr_aligned_value >= 100.0:
                    psnr_aligned = "inf"
                else:
                    psnr_aligned = f"{psnr_aligned_value:.4f}"

                # Calculate aligned SSIM
                gray_orig_cropped = cv2.cvtColor(img_orig_cropped, cv2.COLOR_BGR2GRAY)
                gray_trans_aligned = cv2.cvtColor(img_trans_aligned, cv2.COLOR_BGR2GRAY)
                ssim_aligned = calculate_ssim(gray_orig_cropped, gray_trans_aligned)
            elif orig_h > orig_w and trans_w == trans_h:
                # Vertical to square: center-crop original height
                crop_h = orig_w
                crop_y = (orig_h - crop_h) // 2
                img_orig_cropped = img_orig[crop_y:crop_y+crop_h, :]
                alignment_method = 'crop_reference_center_square'

                # Resize transformed to match cropped original
                img_trans_aligned = cv2.resize(img_trans, (img_orig_cropped.shape[1], img_orig_cropped.shape[0]), interpolation=cv2.INTER_CUBIC)

                # Calculate aligned PSNR
                psnr_aligned_value = cv2.PSNR(img_orig_cropped, img_trans_aligned)
                if psnr_aligned_value >= 100.0:
                    psnr_aligned = "inf"
                else:
                    psnr_aligned = f"{psnr_aligned_value:.4f}"

                # Calculate aligned SSIM
                gray_orig_cropped = cv2.cvtColor(img_orig_cropped, cv2.COLOR_BGR2GRAY)
                gray_trans_aligned = cv2.cvtColor(img_trans_aligned, cv2.COLOR_BGR2GRAY)
                ssim_aligned = calculate_ssim(gray_orig_cropped, gray_trans_aligned)
            else:
                # Generic aspect ratio change: scale both to minimum dimensions
                common_w = min(orig_w, trans_w)
                common_h = min(orig_h, trans_h)
                img_orig_cropped = cv2.resize(img_orig, (common_w, common_h), interpolation=cv2.INTER_CUBIC)
                img_trans_aligned = cv2.resize(img_trans, (common_w, common_h), interpolation=cv2.INTER_CUBIC)
                alignment_method = 'scale_both_to_minimum'

                # Calculate aligned PSNR
                psnr_aligned_value = cv2.PSNR(img_orig_cropped, img_trans_aligned)
                if psnr_aligned_value >= 100.0:
                    psnr_aligned = "inf"
                else:
                    psnr_aligned = f"{psnr_aligned_value:.4f}"

                # Calculate aligned SSIM
                gray_orig_cropped = cv2.cvtColor(img_orig_cropped, cv2.COLOR_BGR2GRAY)
                gray_trans_aligned = cv2.cvtColor(img_trans_aligned, cv2.COLOR_BGR2GRAY)
                ssim_aligned = calculate_ssim(gray_orig_cropped, gray_trans_aligned)

        else:
            # Same aspect ratio - aligned = stretched
            psnr_aligned = psnr
            ssim_aligned = ssim
            alignment_method = 'same_aspect_ratio'

        elapsed_ms = (time.time() - start_time) * 1000

        return psnr, ssim, psnr_aligned, ssim_aligned, alignment_method, lossless_match, None, elapsed_ms

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = f"Image metrics error: {str(e)}"
        logger.error(f"{transformed_path.name}: {error_msg}")
        return None, None, None, None, '', 0, error_msg, elapsed_ms


def get_video_properties(video_path: Path) -> Optional[Tuple[int, int, str]]:
    """
    Extract video dimensions and frame rate using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height, fps_string) or None if extraction fails
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-of', 'csv=p=0',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        output = result.stdout.strip()

        # Parse output: "width,height,fps_num/fps_den"
        parts = output.split(',')
        if len(parts) >= 3:
            width = int(parts[0])
            height = int(parts[1])
            fps = parts[2]
            return width, height, fps
        else:
            logger.error(f"Failed to parse ffprobe output: {output}")
            return None

    except Exception as e:
        logger.error(f"ffprobe failed for {video_path.name}: {e}")
        return None


def calculate_video_vmaf(original_path: Path, transformed_path: Path, transform_type: Optional[str] = None) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], float]:
    """
    Calculate comprehensive VMAF scores for video pair with aspect ratio handling.

    Calculates two VMAF scores to handle platform transforms accurately:
    1. vmaf: Traditional VMAF - scales distorted video to match reference dimensions
       - May show artificially low scores when aspect ratios differ (e.g., Instagram 16:9 → 1:1 crop)
    2. vmaf_aligned: Aspect-ratio corrected VMAF - intelligently aligns videos before comparison
       - Crops reference to match distorted aspect ratio (e.g., center square crop for Instagram)
       - More accurately reflects perceptual quality when platforms apply cropping
       - Identical to vmaf when aspect ratios match

    Args:
        original_path: Path to original signed video (reference)
        transformed_path: Path to transformed video (distorted)
        transform_type: Optional transform type from filename metadata (e.g., 'resize', 'crop')

    Returns:
        Tuple of (vmaf, vmaf_aligned, vmaf_method, error_message, processing_time_ms)
        - vmaf: Float (0-100, higher = better quality) or None if failed
        - vmaf_aligned: Float (0-100, higher = better quality) or None if failed
        - vmaf_method: String describing alignment method used:
            * "same_aspect_ratio" - No alignment needed, both metrics identical
            * "crop_reference_center_square" - Reference cropped to square (Instagram-style)
            * "scale_both_to_minimum" - Both scaled to smallest common dimensions
        - error_message: String error description or None if successful
        - processing_time_ms: Float milliseconds elapsed
    """
    start_time = time.time()

    try:
        # Get video properties for both files
        ref_props = get_video_properties(original_path)
        dist_props = get_video_properties(transformed_path)

        if ref_props is None or dist_props is None:
            elapsed_ms = (time.time() - start_time) * 1000
            error_msg = "Failed to extract video properties with ffprobe"
            logger.error(f"{transformed_path.name}: {error_msg}")
            return None, None, None, error_msg, elapsed_ms

        ref_width, ref_height, ref_fps = ref_props
        dist_width, dist_height, dist_fps = dist_props

        # Calculate aspect ratios
        ref_aspect = ref_width / ref_height
        dist_aspect = dist_width / dist_height
        aspect_changed = abs(ref_aspect - dist_aspect) > 0.01

        vmaf = None
        vmaf_aligned = None
        method = 'none'

        # VMAF 1: Traditional method - scale distorted to reference
        if (dist_width != ref_width or dist_height != ref_height or dist_fps != ref_fps):
            filter_chain_stretched = (
                f"[0:v]scale={ref_width}:{ref_height}:flags=lanczos,"
                f"fps={ref_fps},format=yuv420p,setpts=PTS-STARTPTS[dist];"
                f"[1:v]format=yuv420p,setpts=PTS-STARTPTS[ref];"
                f"[dist][ref]libvmaf=log_fmt=json:log_path=NUL"
            )
        else:
            filter_chain_stretched = (
                "[0:v]format=yuv420p,setpts=PTS-STARTPTS[dist];"
                "[1:v]format=yuv420p,setpts=PTS-STARTPTS[ref];"
                "[dist][ref]libvmaf=log_fmt=json:log_path=NUL"
            )

        cmd_stretched = [
            'ffmpeg',
            '-i', str(transformed_path),
            '-i', str(original_path),
            '-lavfi', filter_chain_stretched,
            '-f', 'null', '-'
        ]

        result_stretched = subprocess.run(cmd_stretched, capture_output=True, text=True, timeout=600, check=False)
        match = re.search(r'VMAF score:\s*(\d+\.\d+)', result_stretched.stderr)
        if not match:
            match = re.search(r'"vmaf":\s*(\d+\.\d+)', result_stretched.stderr)
        if match:
            vmaf = float(match.group(1))

        # VMAF 2: Aligned (intelligent cropping/scaling)
        if aspect_changed:
            # Check if this is a resize transform (full frame scale) or actual crop
            is_resize_transform = transform_type == 'resize'
            is_rotation_transform = transform_type == 'rotate'

            if is_resize_transform or is_rotation_transform:
                # Resize and rotation transforms scale/rotate the entire frame - no content is cropped
                # Both videos have same field of view, just different resolution/orientation
                # Use same alignment as traditional VMAF (no cropping needed)
                vmaf_aligned = vmaf
                method = 'same_aspect_ratio'
                logger.debug(f"{transformed_path.name}: Using same_aspect_ratio for {transform_type} transform")
            elif ref_width > ref_height and dist_width == dist_height:
                # Horizontal to square crop - center crop reference
                # This handles actual crop transforms or platform square crops
                crop_width = ref_height
                crop_x = (ref_width - crop_width) // 2
                method = 'crop_reference_center_square'

                filter_chain_aligned = (
                    f"[1:v]crop={crop_width}:{ref_height}:{crop_x}:0,"
                    f"scale={dist_width}:{dist_height}:flags=lanczos,"
                    f"fps={dist_fps},format=yuv420p,setpts=PTS-STARTPTS[ref];"
                    f"[0:v]fps={dist_fps},format=yuv420p,setpts=PTS-STARTPTS[dist];"
                    f"[dist][ref]libvmaf=log_fmt=json:log_path=NUL"
                )

                cmd_aligned = [
                    'ffmpeg',
                    '-i', str(transformed_path),
                    '-i', str(original_path),
                    '-lavfi', filter_chain_aligned,
                    '-f', 'null', '-'
                ]

                result_aligned = subprocess.run(cmd_aligned, capture_output=True, text=True, timeout=600, check=False)
                match_aligned = re.search(r'VMAF score:\s*(\d+\.\d+)', result_aligned.stderr)
                if not match_aligned:
                    match_aligned = re.search(r'"vmaf":\s*(\d+\.\d+)', result_aligned.stderr)
                if match_aligned:
                    vmaf_aligned = float(match_aligned.group(1))
            elif ref_height > ref_width and dist_width == dist_height:
                # Vertical to square crop - center crop reference
                crop_height = ref_width
                crop_y = (ref_height - crop_height) // 2
                method = 'crop_reference_center_square'

                filter_chain_aligned = (
                    f"[1:v]crop={ref_width}:{crop_height}:0:{crop_y},"
                    f"scale={dist_width}:{dist_height}:flags=lanczos,"
                    f"fps={dist_fps},format=yuv420p,setpts=PTS-STARTPTS[ref];"
                    f"[0:v]fps={dist_fps},format=yuv420p,setpts=PTS-STARTPTS[dist];"
                    f"[dist][ref]libvmaf=log_fmt=json:log_path=NUL"
                )

                cmd_aligned = [
                    'ffmpeg',
                    '-i', str(transformed_path),
                    '-i', str(original_path),
                    '-lavfi', filter_chain_aligned,
                    '-f', 'null', '-'
                ]

                result_aligned = subprocess.run(cmd_aligned, capture_output=True, text=True, timeout=600, check=False)
                match_aligned = re.search(r'VMAF score:\s*(\d+\.\d+)', result_aligned.stderr)
                if not match_aligned:
                    match_aligned = re.search(r'"vmaf":\s*(\d+\.\d+)', result_aligned.stderr)
                if match_aligned:
                    vmaf_aligned = float(match_aligned.group(1))
            else:
                # Unknown aspect ratio change - scale both to smaller dimension
                common_width = min(ref_width, dist_width)
                common_height = min(ref_height, dist_height)
                method = 'scale_both_to_minimum'

                filter_chain_aligned = (
                    f"[0:v]scale={common_width}:{common_height}:flags=lanczos,"
                    f"fps={dist_fps},format=yuv420p,setpts=PTS-STARTPTS[dist];"
                    f"[1:v]scale={common_width}:{common_height}:flags=lanczos,"
                    f"fps={dist_fps},format=yuv420p,setpts=PTS-STARTPTS[ref];"
                    f"[dist][ref]libvmaf=log_fmt=json:log_path=NUL"
                )

                cmd_aligned = [
                    'ffmpeg',
                    '-i', str(transformed_path),
                    '-i', str(original_path),
                    '-lavfi', filter_chain_aligned,
                    '-f', 'null', '-'
                ]

                result_aligned = subprocess.run(cmd_aligned, capture_output=True, text=True, timeout=600, check=False)
                match_aligned = re.search(r'VMAF score:\s*(\d+\.\d+)', result_aligned.stderr)
                if not match_aligned:
                    match_aligned = re.search(r'"vmaf":\s*(\d+\.\d+)', result_aligned.stderr)
                if match_aligned:
                    vmaf_aligned = float(match_aligned.group(1))
        else:
            # Same aspect ratio - aligned = vmaf
            vmaf_aligned = vmaf
            method = 'same_aspect_ratio'

        elapsed_ms = (time.time() - start_time) * 1000
        return vmaf, vmaf_aligned, method, None, elapsed_ms

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = "VMAF calculation timeout"
        logger.error(f"{transformed_path.name}: {error_msg}")
        return None, None, None, error_msg, elapsed_ms
    except FileNotFoundError:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = "ffmpeg not found or libvmaf not available"
        logger.error(f"{transformed_path.name}: {error_msg}")
        return None, None, None, error_msg, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_msg = f"VMAF error: {str(e)}"
        logger.error(f"{transformed_path.name}: {error_msg}")
        return None, None, None, error_msg, elapsed_ms


def process_single_asset(transformed_path: Path) -> Dict:
    """
    Calculate metrics for single asset.

    Detects lossless transforms by parsing filename for transform type.
    Examples:
    - img_000_seed42_20251109_220519_png_c9.png → lossless_transform = 1
    - img_000_seed42_20251109_220519_jpeg_q95.jpg → lossless_transform = 0
    - video_000_seed100_20251109_231519_h264_bitrate5000k.mp4 → lossless_transform = 0

    Args:
        transformed_path: Path to transformed asset

    Returns:
        Dict with metrics row data
    """
    # Find original asset
    original_path = find_original_asset(transformed_path)

    # Detect lossless transform from filename
    filename = transformed_path.name
    lossless_transform = 0
    for transform_key in LOSSLESS_TRANSFORMS:
        if transform_key in filename:
            lossless_transform = 1
            break

    # Determine asset type
    asset_type = 'image' if transformed_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else 'video'

    # Extract seed and model_version
    seed, model_version = extract_seed_and_model(filename, asset_type)

    # Detect media source
    media_source = utils.detect_media_source(transformed_path.name)

    # Extract transform type from filename metadata
    metadata = utils.extract_metadata_from_filename(filename)
    transform_type = metadata.get('transform_type', None)

    if original_path is None:
        return {
            'filename': transformed_path.name,
            'asset_type': asset_type,
            'seed': seed,
            'model_version': model_version,
            'media_source': media_source,
            'psnr': '',
            'psnr_aligned': '',
            'ssim': '',
            'ssim_aligned': '',
            'vmaf': '',
            'vmaf_aligned': '',
            'alignment_method': '',
            'lossless_match': '0',
            'lossless_transform': str(lossless_transform),
            'processing_time_ms': '0.00',
            'calculation_error': 'original_not_found',
            'timestamp': datetime.now().isoformat()
        }

    # Calculate appropriate metrics based on asset type
    if asset_type == 'image':
        psnr, ssim, psnr_aligned, ssim_aligned, alignment_method, lossless_match, error, proc_time = calculate_image_metrics(original_path, transformed_path, transform_type=transform_type)

        return {
            'filename': transformed_path.name,
            'asset_type': 'image',
            'seed': seed,
            'model_version': model_version,
            'media_source': media_source,
            'psnr': psnr if psnr is not None else '',
            'psnr_aligned': psnr_aligned if psnr_aligned is not None else '',
            'ssim': f"{ssim:.6f}" if ssim is not None else '',
            'ssim_aligned': f"{ssim_aligned:.6f}" if ssim_aligned is not None else '',
            'vmaf': 'NA',  # Not applicable for images
            'vmaf_aligned': 'NA',  # Not applicable for images
            'alignment_method': alignment_method if alignment_method else '',
            'lossless_match': str(lossless_match),
            'lossless_transform': str(lossless_transform),
            'processing_time_ms': f"{proc_time:.2f}",
            'calculation_error': error if error else 'NA',
            'timestamp': datetime.now().isoformat()
        }
    else:
        vmaf, vmaf_aligned, vmaf_method, error, proc_time = calculate_video_vmaf(original_path, transformed_path, transform_type=transform_type)

        return {
            'filename': transformed_path.name,
            'asset_type': 'video',
            'seed': seed,
            'model_version': model_version,
            'media_source': media_source,
            'psnr': 'NA',  # Not applicable for videos
            'psnr_aligned': 'NA',  # Not applicable for videos
            'ssim': 'NA',  # Not applicable for videos
            'ssim_aligned': 'NA',  # Not applicable for videos
            'vmaf': f"{vmaf:.4f}" if vmaf is not None else '',
            'vmaf_aligned': f"{vmaf_aligned:.4f}" if vmaf_aligned is not None else '',
            'alignment_method': vmaf_method if vmaf_method else '',
            'lossless_match': '0',  # Not applicable for videos (always lossy)
            'lossless_transform': str(lossless_transform),
            'processing_time_ms': f"{proc_time:.2f}",
            'calculation_error': error if error else 'NA',
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
        logger.info(f"TEST MODE: Selected {len(test_assets)} assets")
        return test_assets

    return assets


def process_assets_parallel(assets: list, max_workers: int = 4):
    """
    Process assets in parallel using ProcessPoolExecutor.

    Args:
        assets: List of asset paths
        max_workers: Number of parallel workers
    """
    logger.info(f"Processing {len(assets)} assets with {max_workers} workers")

    # Write CSV header using shared utility
    utils.write_csv_header(OUTPUT_CSV, header_type='quality_metrics')

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_asset, asset): asset for asset in assets}

        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating metrics", unit="asset"):
            try:
                row_data = future.result()
                utils.append_csv_row(OUTPUT_CSV, row_data, header_type='quality_metrics')
            except Exception as e:
                asset = futures[future]
                logger.error(f"Failed to process {asset.name}: {e}")


def process_assets_sequential(assets: list):
    """
    Process assets sequentially (for debugging or test mode).

    Args:
        assets: List of asset paths
    """
    # Write CSV header using shared utility
    utils.write_csv_header(OUTPUT_CSV, header_type='quality_metrics')

    for asset in tqdm(assets, desc="Calculating metrics", unit="asset"):
        try:
            row_data = process_single_asset(asset)
            utils.append_csv_row(OUTPUT_CSV, row_data, header_type='quality_metrics')
        except Exception as e:
            logger.error(f"Failed to process {asset.name}: {e}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Quality Metrics Calculation Script")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")

    # Check for ffmpeg (optional, only needed for videos)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        ffmpeg_version = result.stdout.split('\n')[0]
        logger.info(f"ffmpeg: {ffmpeg_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffmpeg not found - video metrics (VMAF) will be skipped")

    logger.info("=" * 60)

    # Parse arguments
    test_mode = '--test' in sys.argv

    if test_mode:
        logger.info("TEST MODE: Processing one asset from each category")

    # Collect assets
    assets = collect_transformed_assets(test_mode=test_mode)

    if not assets:
        logger.error(f"No transformed assets found in {TRANSFORMED_BASE_DIR}")
        return

    # Process assets
    if test_mode:
        # Sequential processing for test mode (easier debugging)
        process_assets_sequential(assets)
    else:
        # Parallel processing for full run
        process_assets_parallel(assets, max_workers=4)

    logger.info("=" * 60)
    logger.info("Quality Metrics Calculation Complete")
    logger.info(f"  Processed: {len(assets)} assets")
    logger.info(f"  Output: {OUTPUT_CSV.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
