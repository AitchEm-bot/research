"""
Common Utility Functions for C2PA Robustness Testing Pipeline
=============================================================

This module provides shared functionality across all pipeline scripts
to reduce code duplication and improve maintainability.

Key Features:
- Centralized logging configuration
- Safe subprocess execution with timeout and validation
- CSV operations with consistent headers
- C2PA tool wrappers for signing and verification
- File path operations and metadata extraction
- Directory structure management

Security Features:
- Input validation for subprocess commands
- Shell injection prevention
- Timeout enforcement on external commands
- Safe path handling
"""

import csv
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


# ==================== CONFIGURATION ====================

# Support for environment variables (Docker compatibility)
# Falls back to local paths if environment variables not set

# Project root directory (scripts/common/../.. = project root)
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', Path(__file__).parent.parent.parent.resolve()))

# Standard directory structure
DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / "data"))
SCRIPTS_DIR = Path(os.getenv('SCRIPTS_DIR', PROJECT_ROOT / "scripts"))

# Data subdirectories (NEW reorganized structure)
DIRS = {
    # Assets (raw generated/external)
    'raw_images': DATA_DIR / "assets/raw_images",
    'raw_videos': DATA_DIR / "assets/raw_videos",
    'raw_images_for_videos': DATA_DIR / "assets/raw_images_for_videos",
    'raw_out_videos': DATA_DIR / "assets/raw_out_videos",

    # Prepared assets (processed)
    'signed_images': DATA_DIR / "prepared_assets/signed_assets/images",
    'signed_videos_internal': DATA_DIR / "prepared_assets/signed_assets/videos/internal",
    'signed_videos_external': DATA_DIR / "prepared_assets/signed_assets/videos/external",
    'c2pa_manifests': DATA_DIR / "prepared_assets/c2pa_manifests",
    'transformed': DATA_DIR / "prepared_assets/transformed",
    'compression_images': DATA_DIR / "prepared_assets/transformed/compression/images",
    'compression_videos': DATA_DIR / "prepared_assets/transformed/compression/videos",
    'editing_images': DATA_DIR / "prepared_assets/transformed/editing/images",
    'editing_videos': DATA_DIR / "prepared_assets/transformed/editing/videos",
    'platform_tests': DATA_DIR / "prepared_assets/platform_tests",

    # Results (all CSV outputs and logs)
    'results': DATA_DIR / "results",
    'results_logs': DATA_DIR / "results/logs",

    # Phase 4: Analysis results directories
    'results_csv': DATA_DIR / "results/csv",
    'analysis_results': DATA_DIR / "results/analysis_results",
    'analysis_csv': DATA_DIR / "results/analysis_results/csv",
    'analysis_plots': DATA_DIR / "results/analysis_results/plots",
}

# C2PA tool configuration
# Support environment variable for c2patool path (Docker compatibility)
if os.getenv('C2PATOOL_PATH'):
    C2PATOOL_CMD = os.getenv('C2PATOOL_PATH')
    C2PATOOL_LOCAL = Path(C2PATOOL_CMD)
else:
    # Check multiple locations in order of preference
    possible_paths = [
        PROJECT_ROOT / "tools/c2patool/c2patool/c2patool.exe",  # Windows local
        PROJECT_ROOT / "tools/c2patool/c2patool",               # Linux local (mounted in Docker)
        PROJECT_ROOT / "tools/c2patool",                        # Direct binary
        Path("/usr/local/bin/c2patool"),                        # System install
    ]
    C2PATOOL_LOCAL = None
    for path in possible_paths:
        if path.exists():
            C2PATOOL_LOCAL = path
            break
    C2PATOOL_CMD = str(C2PATOOL_LOCAL) if C2PATOOL_LOCAL else "c2patool"

# Lossless transform types (from CLAUDE.md)
LOSSLESS_TRANSFORMS = {'png_c0', 'png_c9'}

# Standard CSV headers for different metric types
CSV_HEADERS = {
    'quality_metrics': [
        'filename', 'asset_type', 'seed', 'model_version', 'media_source',
        'psnr', 'psnr_aligned', 'ssim', 'ssim_aligned',
        'vmaf', 'vmaf_aligned', 'alignment_method',
        'lossless_match', 'lossless_transform',
        'processing_time_ms', 'calculation_error', 'timestamp'
    ],
    'c2pa_validation': [
        'filename', 'asset_type', 'transform_type', 'transform_level',
        'seed', 'model_version', 'media_source', 'manifest_present', 'verified',
        'signature_valid', 'hash_match', 'assertion_uris_match',
        'trust_verified', 'validation_state', 'failure_reason',
        'processing_time_ms', 'timestamp'
    ],
    'platform_results': [
        'filename', 'platform', 'platform_mode', 'video_source',
        'upload_timestamp', 'download_timestamp', 'manifest_present',
        'verified', 'psnr', 'ssim', 'vmaf', 'processing_time_ms'
    ]
}


# ==================== LOGGING ====================

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure standardized logging for scripts.

    Args:
        log_file: Optional log file path. If None, logs to stdout only.
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    handlers = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    return logger


def log_environment_info():
    """Log Python and system environment information for reproducibility."""
    logger = logging.getLogger()
    logger.info("=" * 60)
    logger.info("Environment Information")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")

    # Check for optional tools
    tools = {
        'ffmpeg': ['ffmpeg', '-version'],
        'ffprobe': ['ffprobe', '-version'],
        'c2patool': [C2PATOOL_CMD, '--version']
    }

    for tool_name, cmd in tools.items():
        try:
            result = run_command(cmd, timeout=5, capture_output=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0] if result.stdout else "unknown version"
                logger.info(f"{tool_name}: {version_line}")
            else:
                logger.warning(f"{tool_name}: Not available or error occurred")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(f"{tool_name}: Not found")

    logger.info("=" * 60)


# ==================== DIRECTORY MANAGEMENT ====================

def ensure_directories():
    """Create all required project directories if they don't exist."""
    for dir_name, dir_path in DIRS.items():
        dir_path.mkdir(parents=True, exist_ok=True)


def get_output_dir(transform_type: str, asset_type: str, level: Optional[str] = None) -> Path:
    """
    Get standardized output directory for transformed assets.

    Args:
        transform_type: Type of transformation (jpeg, png, h264, crop, resize, etc.)
        asset_type: 'image' or 'video'
        level: Optional quality/parameter level (q95, bitrate2000k, etc.)

    Returns:
        Path to output directory
    """
    asset_plural = "images" if asset_type == "image" else "videos"

    # Determine base category
    if transform_type in ['jpeg', 'png', 'h264', 'h265', 'fps']:
        category = "compression"
    else:
        category = "editing"

    base_dir = DATA_DIR / "transformed" / category / asset_plural

    if level:
        output_dir = base_dir / transform_type / level
    else:
        output_dir = base_dir / transform_type

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ==================== SUBPROCESS EXECUTION ====================

def validate_command(cmd: List[str]) -> bool:
    """
    Validate subprocess command for security.

    Args:
        cmd: Command list to validate

    Returns:
        True if command is safe, False otherwise
    """
    if not cmd:
        return False

    # Allowed executables (whitelist approach)
    allowed_executables = {
        'ffmpeg', 'ffprobe', 'c2patool', C2PATOOL_CMD,
        str(C2PATOOL_LOCAL), 'python', sys.executable
    }

    # Get base executable name
    executable = Path(cmd[0]).name.lower()
    executable_full = cmd[0]

    # Check against whitelist
    if executable_full not in allowed_executables and executable not in allowed_executables:
        logging.warning(f"Blocked unauthorized command: {cmd[0]}")
        return False

    # Check for shell metacharacters in arguments
    # Exception: Allow certain safe patterns like ffmpeg filter chains
    dangerous_chars = [';', '&', '|', '>', '<', '`', '$', '(', ')', '{', '}', '\n', '\r']
    for i, arg in enumerate(cmd[1:], 1):
        # Skip validation for known safe patterns
        # ffmpeg filter chains with -lavfi or -filter_complex are safe when properly quoted
        if i > 1 and cmd[i-1] in ['-lavfi', '-filter_complex', '-vf', '-af']:
            continue  # These are filter expressions, not shell commands

        # Check for dangerous characters in other arguments
        if any(char in str(arg) for char in dangerous_chars):
            # Additional check: Allow brackets in filter expressions
            if '[' in str(arg) and ']' in str(arg) and 'ffmpeg' in cmd[0]:
                continue  # Likely a filter chain, allow it

            logging.warning(f"Blocked command with shell metacharacters: {arg}")
            return False

    return True


def run_command(
    cmd: List[str],
    timeout: int = 60,
    check: bool = True,
    capture_output: bool = True,
    cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """
    Safely execute subprocess command with validation and timeout.

    Args:
        cmd: Command and arguments as list
        timeout: Maximum execution time in seconds
        check: Raise exception on non-zero exit
        capture_output: Capture stdout/stderr
        cwd: Working directory for command

    Returns:
        CompletedProcess instance

    Raises:
        ValueError: If command validation fails
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command exceeds timeout
    """
    if not validate_command(cmd):
        raise ValueError(f"Command failed security validation: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=cwd,
            shell=False  # Never use shell=True for security
        )
        return result
    except subprocess.TimeoutExpired as e:
        logging.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with code {e.returncode}: {' '.join(cmd)}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        raise


# ==================== FILE OPERATIONS ====================

def find_original_asset(transformed_path: Path, signed_dirs: Optional[Dict] = None) -> Optional[Path]:
    """
    Find the original signed asset for a transformed file.

    Args:
        transformed_path: Path to transformed asset
        signed_dirs: Optional dict of signed asset directories to search

    Returns:
        Path to original signed asset, or None if not found
    """
    if signed_dirs is None:
        signed_dirs = {
            'images': [DIRS['signed_images']],
            'videos': [DIRS['signed_videos_internal'], DIRS['signed_videos_external']]
        }

    filename = transformed_path.name
    base = Path(filename).stem

    # Remove transformation suffixes
    base = re.sub(r'_(jpeg|png|h264|h265)_.*$', '', base)
    base = re.sub(r'_fps\d+$', '', base)
    base = re.sub(r'_(crop|resize|rotate|brightness|contrast|saturation).*$', '', base)

    # Determine asset type and search directories
    if transformed_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
        original_filename = f"{base}_signed.png"
        search_dirs = signed_dirs.get('images', [])
    else:
        original_filename = f"{base}_signed.mp4"
        search_dirs = signed_dirs.get('videos', [])

    # Search for original file
    for search_dir in search_dirs:
        original_path = search_dir / original_filename
        if original_path.exists():
            return original_path

    logging.warning(f"Original not found for {filename}: expected {original_filename}")
    return None


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract seed, model version, and transform details from filename.

    Args:
        filename: Asset filename

    Returns:
        Dict with keys: seed, model_version, transform_type, transform_level
    """
    metadata = {
        'seed': '',
        'model_version': '',
        'transform_type': '',
        'transform_level': ''
    }

    # Check for external video pattern
    if re.match(r'video_\d+', filename) and 'seed' not in filename:
        metadata['model_version'] = 'Veo3.1'
        metadata['seed'] = 'NA'
    else:
        # Extract seed
        seed_match = re.search(r'seed(\d+)', filename)
        if seed_match:
            metadata['seed'] = seed_match.group(1)

        # Determine model version
        if filename.startswith('img_'):
            metadata['model_version'] = 'SD1.4'
        elif filename.startswith('video_'):
            metadata['model_version'] = 'SVD'

    # Extract transform type and level
    stem = Path(filename).stem

    # Compression transforms
    if '_jpeg_q' in stem:
        metadata['transform_type'] = 'jpeg_compression'
        quality_match = re.search(r'q(\d+)', stem)
        if quality_match:
            metadata['transform_level'] = f"q{quality_match.group(1)}"
    elif '_png_c' in stem:
        metadata['transform_type'] = 'png_compression'
        compress_match = re.search(r'c(\d+)', stem)
        if compress_match:
            metadata['transform_level'] = f"c{compress_match.group(1)}"
    elif '_h264_bitrate' in stem:
        metadata['transform_type'] = 'h264_compression'
        bitrate_match = re.search(r'bitrate(\d+k)', stem)
        if bitrate_match:
            metadata['transform_level'] = bitrate_match.group(1)
    elif '_h265_bitrate' in stem:
        metadata['transform_type'] = 'h265_compression'
        bitrate_match = re.search(r'bitrate(\d+k)', stem)
        if bitrate_match:
            metadata['transform_level'] = bitrate_match.group(1)
    elif '_fps' in stem and not any(x in stem for x in ['brightness', 'contrast', 'saturation']):
        metadata['transform_type'] = 'fps_adjustment'
        fps_match = re.search(r'fps(\d+)', stem)
        if fps_match:
            metadata['transform_level'] = f"{fps_match.group(1)}fps"
    # Editing transforms
    elif '_crop' in stem:
        metadata['transform_type'] = 'crop'
        crop_match = re.search(r'crop(\d+)', stem)
        if crop_match:
            metadata['transform_level'] = f"{crop_match.group(1)}pct"
    elif '_resize' in stem:
        metadata['transform_type'] = 'resize'
        resize_match = re.search(r'resize(\d+x\d+)', stem)
        if resize_match:
            metadata['transform_level'] = resize_match.group(1)
    elif '_rotate' in stem:
        metadata['transform_type'] = 'rotation'
        rotate_match = re.search(r'rotate(\d+)', stem)
        if rotate_match:
            metadata['transform_level'] = f"{rotate_match.group(1)}deg"
    elif '_brightness_' in stem:
        metadata['transform_type'] = 'brightness_adjustment'
        brightness_match = re.search(r'brightness_(minus|plus)?(\d+)', stem)
        if brightness_match:
            sign = '-' if brightness_match.group(1) == 'minus' else '+'
            metadata['transform_level'] = f"{sign}{brightness_match.group(2)}"
    elif '_contrast_' in stem:
        metadata['transform_type'] = 'contrast_adjustment'
        contrast_match = re.search(r'contrast_(minus|plus)?(\d+)', stem)
        if contrast_match:
            sign = '-' if contrast_match.group(1) == 'minus' else '+'
            metadata['transform_level'] = f"{sign}{contrast_match.group(2)}"
    elif '_saturation_' in stem:
        metadata['transform_type'] = 'saturation_adjustment'
        saturation_match = re.search(r'saturation_(minus|plus)?(\d+)', stem)
        if saturation_match:
            sign = '-' if saturation_match.group(1) == 'minus' else '+'
            metadata['transform_level'] = f"{sign}{saturation_match.group(2)}"
    elif '_trim_' in stem:
        metadata['transform_type'] = 'trim'
        trim_match = re.search(r'trim_(\d+)s', stem)
        if trim_match:
            metadata['transform_level'] = f"{trim_match.group(1)}s"

    return metadata


# ==================== CSV OPERATIONS ====================

def write_csv_header(csv_path: Path, header_type: str = 'quality_metrics'):
    """
    Write standardized CSV header.

    Args:
        csv_path: Path to CSV file
        header_type: Type of CSV headers to use from CSV_HEADERS
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = CSV_HEADERS.get(header_type, CSV_HEADERS['quality_metrics'])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()


def append_csv_row(csv_path: Path, row_data: Dict, header_type: str = 'quality_metrics'):
    """
    Append row to CSV file with proper headers.

    Args:
        csv_path: Path to CSV file
        row_data: Dictionary of row data
        header_type: Type of CSV headers to use
    """
    headers = CSV_HEADERS.get(header_type, CSV_HEADERS['quality_metrics'])

    # Ensure all required fields are present
    for header in headers:
        if header not in row_data:
            row_data[header] = ''

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row_data)
        f.flush()


# ==================== C2PA OPERATIONS ====================

def verify_c2pa_manifest(
    asset_path: Path,
    timeout: int = 30
) -> Tuple[bool, Dict[str, any]]:
    """
    Verify C2PA manifest using c2patool.

    Args:
        asset_path: Path to asset to verify
        timeout: Command timeout in seconds

    Returns:
        Tuple of (manifest_present, validation_details)
    """
    try:
        cmd = [C2PATOOL_CMD, '-o', 'json', str(asset_path)]
        result = run_command(cmd, timeout=timeout, check=False)

        if result.returncode != 0:
            # No manifest found
            return False, {'error': 'No manifest found'}

        # Parse JSON output
        try:
            manifest = json.loads(result.stdout)

            # Extract validation status
            validation_status = manifest.get('validation_status', [])

            # Check for various validation aspects
            details = {
                'manifest_present': 1,
                'validation_status': validation_status,
                'verified': 1 if not validation_status else 0,
                'raw_output': manifest
            }

            # Parse specific validation codes
            for status in validation_status:
                code = status.get('code', '')
                if 'claim.signature' in code:
                    details['signature_valid'] = 0
                if 'assertion.dataHash' in code or 'assertion.bmffHash' in code:
                    details['hash_match'] = 0

            return True, details

        except json.JSONDecodeError:
            return False, {'error': 'Invalid JSON output from c2patool'}

    except FileNotFoundError:
        return False, {'error': 'c2patool not found'}
    except subprocess.TimeoutExpired:
        return False, {'error': f'Verification timeout after {timeout}s'}
    except Exception as e:
        return False, {'error': str(e)}


def detect_media_source(filename: str) -> str:
    """
    Detect if media is internal or external based on filename pattern.

    Internal media (generated by our pipeline) contains 'seed' in the filename.
    External media (from Sora, Veo, Midjourney, etc.) does not contain 'seed'.

    Args:
        filename: Filename to check (e.g., "video_22_signed.mp4" or "img_003_seed45.png")

    Returns:
        "internal" if filename contains 'seed', "external" otherwise
    """
    # Check if filename contains 'seed' pattern (e.g., seed42, seed100)
    if re.search(r'seed\d+', filename):
        return "internal"
    else:
        return "external"


def sign_with_c2pa(
    input_path: Path,
    output_path: Path,
    manifest_path: Optional[Path] = None,
    force: bool = False,
    timeout: int = 60
) -> bool:
    """
    Sign asset with C2PA manifest using c2patool.

    Args:
        input_path: Path to input asset
        output_path: Path to output signed asset
        manifest_path: Optional path to manifest JSON
        force: Overwrite output if exists
        timeout: Command timeout in seconds

    Returns:
        True if signing successful, False otherwise
    """
    try:
        if output_path.exists() and not force:
            logging.warning(f"Output already exists: {output_path}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [C2PATOOL_CMD, str(input_path), '-o', str(output_path)]

        if manifest_path and manifest_path.exists():
            cmd.extend(['-m', str(manifest_path)])
        else:
            cmd.extend(['-f'])  # Force signing with test certificate

        result = run_command(cmd, timeout=timeout, check=False)

        if result.returncode == 0:
            logging.info(f"Successfully signed: {output_path.name}")
            return True
        else:
            logging.error(f"Failed to sign {input_path.name}: {result.stderr}")
            return False

    except Exception as e:
        logging.error(f"Error signing {input_path.name}: {e}")
        return False


# ==================== PERFORMANCE TRACKING ====================

class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        import time
        if self.start_time:
            self.elapsed_ms = (time.time() - self.start_time) * 1000


# ==================== VALIDATION ====================

def validate_asset(asset_path: Path) -> bool:
    """
    Validate that an asset file is valid and readable.

    Args:
        asset_path: Path to asset file

    Returns:
        True if asset is valid, False otherwise
    """
    if not asset_path.exists():
        return False

    if asset_path.stat().st_size == 0:
        return False

    # Check file extension
    valid_extensions = {'.png', '.jpg', '.jpeg', '.mp4', '.mov', '.avi'}
    if asset_path.suffix.lower() not in valid_extensions:
        return False

    return True


def get_asset_type(file_path: Path) -> str:
    """
    Determine if file is an image or video.

    Args:
        file_path: Path to file

    Returns:
        'image', 'video', or 'unknown'
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

    suffix = file_path.suffix.lower()
    if suffix in image_extensions:
        return 'image'
    elif suffix in video_extensions:
        return 'video'
    else:
        return 'unknown'


# ==================== BATCH PROCESSING ====================

def collect_assets(
    directory: Path,
    pattern: str = "*",
    recursive: bool = True,
    asset_type: Optional[str] = None
) -> List[Path]:
    """
    Collect asset files from directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for files
        recursive: Search subdirectories
        asset_type: Filter by 'image' or 'video'

    Returns:
        List of asset file paths
    """
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return []

    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    # Filter by asset type if specified
    if asset_type:
        files = [f for f in files if get_asset_type(f) == asset_type]

    return sorted(files)


# ==================== MAIN MODULE TEST ====================

def test_utils():
    """Test utility functions (for module verification)."""
    logger = setup_logging()
    logger.info("Testing utility module...")

    # Test environment logging
    log_environment_info()

    # Test directory creation
    ensure_directories()
    logger.info(f"Directories verified: {len(DIRS)}")

    # Test command validation
    safe_cmd = ['ffmpeg', '-version']
    unsafe_cmd = ['rm', '-rf', '/']
    assert validate_command(safe_cmd) == True
    assert validate_command(unsafe_cmd) == False
    logger.info("Command validation tests passed")

    # Test metadata extraction
    test_filename = "img_000_seed42_20251109_220519_jpeg_q95.jpg"
    metadata = extract_metadata_from_filename(test_filename)
    assert metadata['seed'] == '42'
    assert metadata['model_version'] == 'SD1.4'
    assert metadata['transform_type'] == 'jpeg_compression'
    assert metadata['transform_level'] == 'q95'
    logger.info("Metadata extraction tests passed")

    logger.info("All utility tests passed!")


def log_torch_environment():
    """
    Log PyTorch and CUDA environment information for generation scripts.

    This function is specific to generation scripts that use torch/CUDA.
    For general environment info, use log_environment_info().
    """
    try:
        import torch
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"GPU device: {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"GPU VRAM: {vram_gb:.2f} GB")
    except ImportError:
        logging.warning("PyTorch not available - skipping CUDA environment info")


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility across numpy, torch, and Python random.

    Args:
        seed: Integer seed value

    Note:
        Requires torch and numpy to be installed. For generation scripts only.
    """
    try:
        import torch
        import numpy as np
        import random

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # For full determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logging.info(f"Random seed set to: {seed}")
    except ImportError as e:
        logging.warning(f"Could not set random seed: {e}")


def save_transformation_metadata(output_path: Path, source_path: Path,
                                 transform_type: str, params: dict):
    """
    Save transformation metadata as JSON sidecar file.

    Args:
        output_path: Path to transformed asset
        source_path: Path to source asset
        transform_type: Type of transformation (e.g., 'jpeg_compression', 'crop')
        params: Dictionary of transformation parameters

    Creates a JSON file alongside the output with metadata about the transformation.
    """
    from datetime import datetime
    import json

    try:
        metadata = {
            "source_file": source_path.name,
            "transform_type": transform_type,
            "parameters": params,
            "timestamp": datetime.now().isoformat(),
            "output_file": output_path.name
        }

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        logging.warning(f"Failed to save metadata for {output_path.name}: {e}")


if __name__ == "__main__":
    test_utils()