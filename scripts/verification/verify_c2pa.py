"""
C2PA Verification Script for Robustness Testing
================================================

This script verifies C2PA manifests on all transformed assets and generates
a detailed verification report with failure classification.

C2PA Manifest Preservation Context:
-----------------------------------
C2PA manifests are container-level metadata stored in dedicated boxes/chunks.
The C2PA spec SUPPORTS manifest preservation across edits through:
- Ingredient-level assertions (referencing previous manifests)
- Selective hashing (excluding manifest box from hash calculation)
- Update chains (new manifest references old as ingredient)

However, most standard media tools (Pillow, OpenCV, ffmpeg without -map_metadata,
etc.) re-encode files into NEW containers without C2PA-aware copying. This causes
manifest LOSS, not corruption. The manifest isn't damaged - it's simply not carried
forward during re-encoding.

This is a REAL-WORLD IMPLEMENTATION FAILURE, not a C2PA spec limitation.

Features:
- Parses c2patool JSON output for validation status
- Handles both image (dataHash) and video (bmffHash) verification
- Classifies failure reasons:
  * manifest_not_copied: Tool rewrote container without copying C2PA metadata
  * hash_or_signature_mismatch: Manifest exists but integrity validation failed
- Extracts metadata (seed, model version, transform type)
- Tracks processing runtime per asset

Usage:
    python scripts/verification/verify_c2pa.py [--test]

    --test: Process only one asset from each category (smoke test)

Output:
    data/metrics/c2pa_validation.csv
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import csv
import re

from tqdm import tqdm

# Ensure log directory exists
Path("data/metrics").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/metrics/verify_c2pa.log')
    ]
)

# Configuration
TRANSFORMED_BASE_DIR = Path("data/transformed")
OUTPUT_CSV = Path("data/metrics/c2pa_validation.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# c2patool path (try local installation first, then PATH)
C2PATOOL_LOCAL = Path("tools/c2patool/c2patool/c2patool.exe")
C2PATOOL_CMD = str(C2PATOOL_LOCAL) if C2PATOOL_LOCAL.exists() else "c2patool"

# CSV Column headers (matches CLAUDE.md specification)
CSV_HEADERS = [
    'filename',
    'asset_type',
    'transform_type',
    'transform_level',
    'seed',
    'model_version',
    'manifest_present',
    'verified',
    'signature_valid',
    'hash_match',
    'assertion_uris_match',
    'trust_verified',
    'validation_state',
    'failure_reason',
    'processing_time_ms',
    'timestamp'
]


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extract seed, model version, and transform details from filename.

    Examples:
        img_000_seed42_20251109_220519_jpeg_q95.jpg
        → seed=42, model=sd-v1.4, transform=jpeg_compression, level=q95

        video_000_seed100_20251109_231519_h264_bitrate5000k.mp4
        → seed=100, model=svd-xt, transform=h264_compression, level=bitrate5000k

        img_000_seed42_20251109_220519_brightness_plus19.png
        → seed=42, model=sd-v1.4, transform=brightness_adjustment, level=plus19

    Args:
        filename: Asset filename

    Returns:
        Dict with keys: seed, model_version, transform_type, transform_level
    """
    metadata = {
        'seed': 'unknown',
        'model_version': 'unknown',
        'transform_type': 'unknown',
        'transform_level': 'unknown'
    }

    # Extract seed (seedXX pattern)
    seed_match = re.search(r'seed(\d+)', filename)
    if seed_match:
        metadata['seed'] = seed_match.group(1)

    # Determine model version based on filename prefix and seed range
    if filename.startswith('img_'):
        metadata['model_version'] = 'sd-v1.4'
    elif filename.startswith('video_'):
        seed = int(metadata['seed']) if metadata['seed'].isdigit() else 0
        # Legacy videos: seed 4 (actually seed42 for images converted to video)
        # SVD videos: seed 100-101
        if seed in [4, 42, 43]:
            metadata['model_version'] = 'sd-v1.4-legacy-video'
        else:
            metadata['model_version'] = 'svd-xt'

    # Extract transform type and level from filename suffix
    # Remove extension first
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
        # Extract size and interpolation
        resize_match = re.search(r'resize(\d+x\d+)', stem)
        interp_match = re.search(r'_(bicubic|lanczos)', stem)
        if resize_match:
            size = resize_match.group(1)
            interp = interp_match.group(1) if interp_match else 'default'
            metadata['transform_level'] = f"{size}_{interp}"
    elif '_rotate' in stem:
        metadata['transform_type'] = 'rotation'
        rotate_match = re.search(r'rotate(\d+)', stem)
        if rotate_match:
            metadata['transform_level'] = f"{rotate_match.group(1)}deg"
    elif '_brightness_' in stem:
        metadata['transform_type'] = 'brightness_adjustment'
        factor_match = re.search(r'brightness_(plus|minus)(\d+)', stem)
        if factor_match:
            metadata['transform_level'] = f"{factor_match.group(1)}{factor_match.group(2)}"
    elif '_contrast_' in stem:
        metadata['transform_type'] = 'contrast_adjustment'
        factor_match = re.search(r'contrast_(plus|minus)(\d+)', stem)
        if factor_match:
            metadata['transform_level'] = f"{factor_match.group(1)}{factor_match.group(2)}"
    elif '_saturation_' in stem:
        metadata['transform_type'] = 'saturation_adjustment'
        factor_match = re.search(r'saturation_(plus|minus)(\d+)', stem)
        if factor_match:
            metadata['transform_level'] = f"{factor_match.group(1)}{factor_match.group(2)}"

    return metadata


def run_c2patool(asset_path: Path) -> Tuple[Optional[Dict], float]:
    """
    Run c2patool on asset and return parsed JSON output with timing.

    Args:
        asset_path: Path to asset file

    Returns:
        Tuple of (parsed JSON dict or None, processing time in milliseconds)
    """
    start_time = time.time()

    try:
        cmd = [C2PATOOL_CMD, str(asset_path), '--output', 'json']
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Parse JSON output
        json_data = json.loads(result.stdout)
        return json_data, elapsed_ms

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.time() - start_time) * 1000
        logging.error(f"Timeout verifying {asset_path.name}")
        return None, elapsed_ms
    except subprocess.CalledProcessError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logging.error(f"c2patool failed for {asset_path.name}: {e.stderr}")
        return None, elapsed_ms
    except json.JSONDecodeError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logging.error(f"Invalid JSON from c2patool for {asset_path.name}: {e}")
        return None, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logging.error(f"Unexpected error verifying {asset_path.name}: {e}")
        return None, elapsed_ms


def classify_failure_reason(json_data: Optional[Dict], validation_flags: Dict) -> str:
    """
    Classify the reason for C2PA verification failure.

    Failure categories (updated interpretation):
    - manifest_not_copied: Tool rewrote container without copying C2PA metadata
      (most common case - manifest dropped during re-encoding, not corrupted)
    - hash_or_signature_mismatch: Manifest exists but integrity validation failed
      (signature invalid, hash mismatch, or assertion mismatch)
    - c2patool_parse_error: c2patool could not parse file (rare)
    - success: All validations passed

    Context:
    C2PA manifests are container-level metadata. Standard media tools (Pillow,
    OpenCV, ffmpeg) re-encode files into NEW containers without C2PA-aware
    copying. The manifest isn't corrupted - it's simply not present in the
    new container. This is a real-world implementation failure, not a C2PA
    spec limitation.

    Args:
        json_data: Parsed c2patool JSON output (or None)
        validation_flags: Dict with manifest_present, verified, etc.

    Returns:
        Failure reason string
    """
    # No JSON output from c2patool (parse error or file access issue)
    if json_data is None:
        return "c2patool_parse_error"

    # No manifests found - most common case after transformations
    # This means the tool rewrote the container without copying C2PA metadata
    if not validation_flags['manifest_present']:
        return "manifest_not_copied"

    # Manifest present, check what failed
    if validation_flags['verified']:
        return "success"

    # Manifest exists but validation failed
    # This could be: signature invalid, hash mismatch, or assertion mismatch
    # Group these together as integrity failures
    if not validation_flags['signature_valid'] or \
       not validation_flags['hash_match'] or \
       not validation_flags['assertion_uris_match']:
        return "hash_or_signature_mismatch"

    # Manifest present but verification failed for unknown reason
    return "verification_failed_unknown"


def parse_c2pa_validation(json_data: Optional[Dict]) -> Dict:
    """
    Parse c2patool JSON output and extract validation flags.

    Args:
        json_data: Parsed c2patool JSON output (or None if tool failed)

    Returns:
        Dict with validation flags and state
    """
    if json_data is None:
        return {
            'manifest_present': 0,
            'verified': 0,
            'signature_valid': 0,
            'hash_match': 0,
            'assertion_uris_match': 0,
            'trust_verified': 0,
            'validation_state': 'ERROR'
        }

    # Check if manifests exist
    manifests = json_data.get('manifests', {})
    if not manifests:
        return {
            'manifest_present': 0,
            'verified': 0,
            'signature_valid': 0,
            'hash_match': 0,
            'assertion_uris_match': 0,
            'trust_verified': 0,
            'validation_state': 'NO_MANIFEST'
        }

    # Extract validation status codes from first manifest
    status_codes = []
    validation_state = 'UNKNOWN'

    for manifest_id, manifest_data in manifests.items():
        validation_status = manifest_data.get('validation_status', [])

        for status in validation_status:
            code = status.get('code', '')
            status_codes.append(code)

        # Get overall validation state (if available)
        if 'validation_state' in manifest_data:
            validation_state = manifest_data['validation_state']

        break  # Only process first manifest

    # Check for specific validation codes
    signature_valid = any('claimSignature.validated' in code for code in status_codes)
    hash_match = any(('assertion.dataHash.match' in code or
                      'assertion.bmffHash.match' in code) for code in status_codes)
    assertion_uris = any('assertion.hashedURI.match' in code for code in status_codes)
    trust_verified = any('signingCredential.trusted' in code for code in status_codes)

    # Overall verification: signature AND hash must match
    verified = signature_valid and hash_match

    return {
        'manifest_present': 1,
        'verified': 1 if verified else 0,
        'signature_valid': 1 if signature_valid else 0,
        'hash_match': 1 if hash_match else 0,
        'assertion_uris_match': 1 if assertion_uris else 0,
        'trust_verified': 1 if trust_verified else 0,
        'validation_state': validation_state
    }


def verify_asset(asset_path: Path) -> Dict:
    """
    Verify single asset and return complete row data.

    Args:
        asset_path: Path to transformed asset

    Returns:
        Dict with all CSV columns
    """
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(asset_path.name)

    # Determine asset type
    asset_type = 'image' if asset_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else 'video'

    # Run c2patool verification
    json_data, processing_time = run_c2patool(asset_path)

    # Parse validation results
    validation_flags = parse_c2pa_validation(json_data)

    # Classify failure reason
    failure_reason = classify_failure_reason(json_data, validation_flags)

    # Build row data
    row = {
        'filename': asset_path.name,
        'asset_type': asset_type,
        'transform_type': metadata['transform_type'],
        'transform_level': metadata['transform_level'],
        'seed': metadata['seed'],
        'model_version': metadata['model_version'],
        'manifest_present': validation_flags['manifest_present'],
        'verified': validation_flags['verified'],
        'signature_valid': validation_flags['signature_valid'],
        'hash_match': validation_flags['hash_match'],
        'assertion_uris_match': validation_flags['assertion_uris_match'],
        'trust_verified': validation_flags['trust_verified'],
        'validation_state': validation_flags['validation_state'],
        'failure_reason': failure_reason,
        'processing_time_ms': f"{processing_time:.2f}",
        'timestamp': datetime.now().isoformat()
    }

    return row


def collect_transformed_assets(test_mode: bool = False) -> list:
    """
    Collect all transformed assets for verification.

    Args:
        test_mode: If True, return only one asset from each category

    Returns:
        List of asset paths
    """
    assets = []

    # Image compression
    assets.extend(TRANSFORMED_BASE_DIR.glob("compression/images/**/*.png"))
    assets.extend(TRANSFORMED_BASE_DIR.glob("compression/images/**/*.jpg"))

    # Video compression
    assets.extend(TRANSFORMED_BASE_DIR.glob("compression/videos/**/*.mp4"))

    # Image editing
    assets.extend(TRANSFORMED_BASE_DIR.glob("editing/images/**/*.png"))

    # Video editing
    assets.extend(TRANSFORMED_BASE_DIR.glob("editing/videos/**/*.mp4"))

    assets = sorted(assets)

    if test_mode:
        # Take first asset from each major category
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
        logging.info(f"TEST MODE: Selected {len(test_assets)} assets from each category")
        return test_assets

    return assets


def process_assets(test_mode: bool = False):
    """
    Process all transformed assets and generate verification CSV.

    Args:
        test_mode: If True, process only test assets
    """
    assets = collect_transformed_assets(test_mode=test_mode)

    if not assets:
        logging.error(f"No transformed assets found in {TRANSFORMED_BASE_DIR}")
        return

    logging.info(f"Found {len(assets)} transformed assets to verify")

    # Write CSV with results
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for asset_path in tqdm(assets, desc="Verifying assets", unit="asset"):
            try:
                row_data = verify_asset(asset_path)
                writer.writerow(row_data)
                csvfile.flush()  # Write immediately for resumability

            except Exception as e:
                logging.error(f"Failed to process {asset_path.name}: {e}")
                continue

    logging.info("=" * 60)
    logging.info("C2PA Verification Complete")
    logging.info(f"  Processed: {len(assets)} assets")
    logging.info(f"  Output: {OUTPUT_CSV.absolute()}")
    logging.info("=" * 60)


def main():
    """Main entry point."""
    logging.info("=" * 60)
    logging.info("C2PA Verification Script - Robustness Testing")
    logging.info(f"Python version: {sys.version}")

    # Check for c2patool
    try:
        result = subprocess.run([C2PATOOL_CMD, '--version'], capture_output=True, text=True, check=True)
        logging.info(f"c2patool: {result.stdout.strip()}")
        if C2PATOOL_CMD != "c2patool":
            logging.info(f"  Using local c2patool: {C2PATOOL_CMD}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("c2patool not found! Please install c2patool or check tools/c2patool/ directory")
        sys.exit(1)

    logging.info("=" * 60)

    # Parse arguments
    test_mode = '--test' in sys.argv

    if test_mode:
        logging.info("TEST MODE: Processing one asset from each category")

    # Process assets
    process_assets(test_mode=test_mode)


if __name__ == "__main__":
    main()
