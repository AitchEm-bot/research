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
    data/results/c2pa_validation.csv
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

from tqdm import tqdm

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import utils

# Configure logging using shared utility
logger = utils.setup_logging(log_file='data/results/logs/verify_c2pa.log')
utils.log_environment_info()

# Configuration - using shared constants
TRANSFORMED_BASE_DIR = utils.DIRS['transformed']
OUTPUT_CSV = utils.DIRS['results_csv'] / "c2pa_validation.csv"
C2PATOOL_CMD = utils.C2PATOOL_CMD

# Use shared CSV headers
CSV_HEADERS = utils.CSV_HEADERS['c2pa_validation']


# Use shared metadata extraction function
def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from filename using shared utility."""
    return utils.extract_metadata_from_filename(filename)


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
        cmd = [C2PATOOL_CMD, str(asset_path)]
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
        logger.debug(f"Timeout verifying {asset_path.name}")
        return None, elapsed_ms
    except subprocess.CalledProcessError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        # No claim found is expected for transformed assets - don't log as error
        if "No claim found" not in e.stderr:
            logger.warning(f"c2patool failed for {asset_path.name}: {e.stderr}")
        return None, elapsed_ms
    except json.JSONDecodeError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Invalid JSON from c2patool for {asset_path.name}: {e}")
        return None, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Unexpected error verifying {asset_path.name}: {e}")
        return None, elapsed_ms


def classify_failure_reason(json_data: Optional[Dict], validation_flags: Dict) -> str:
    """
    Classify the reason for C2PA verification failure.

    Failure categories:
    - manifest_dropped: C2PA manifest was stripped/lost during transformation
      (most common case - transformation tools don't preserve C2PA metadata)
    - hash_or_signature_mismatch: Manifest exists but integrity validation failed
      (signature invalid, hash mismatch, or assertion mismatch)
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
    # No JSON output from c2patool OR no manifests found
    # Both cases mean: manifest was dropped during transformation
    if json_data is None or not validation_flags['manifest_present']:
        return "manifest_dropped"

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
            'validation_state': 'no_manifest'
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
            'validation_state': 'no_manifest'
        }

    # Extract validation status codes from validation_results
    # c2patool format: validation_results -> activeManifest -> success/failure arrays
    validation_state = json_data.get('validation_state', 'UNKNOWN')

    validation_results = json_data.get('validation_results', {})
    active_manifest_results = validation_results.get('activeManifest', {})

    # Collect codes from success array
    success_codes = []
    for result in active_manifest_results.get('success', []):
        code = result.get('code', '')
        if code:
            success_codes.append(code)

    # Collect codes from failure array (for trust status)
    failure_codes = []
    for result in active_manifest_results.get('failure', []):
        code = result.get('code', '')
        if code:
            failure_codes.append(code)

    # Check for specific validation codes in success array
    signature_valid = any('claimSignature.validated' in code for code in success_codes)
    hash_match = any(('assertion.dataHash.match' in code or
                      'assertion.bmffHash.match' in code) for code in success_codes)
    assertion_uris = any('assertion.hashedURI.match' in code for code in success_codes)

    # Trust is typically in failure array (untrusted) or would be in success (trusted)
    trust_verified = any('signingCredential.trusted' in code for code in success_codes)

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

    # Detect media source
    media_source = utils.detect_media_source(asset_path.name)

    # Build row data
    row = {
        'filename': asset_path.name,
        'asset_type': asset_type,
        'transform_type': metadata['transform_type'],
        'transform_level': metadata['transform_level'],
        'seed': metadata['seed'],
        'model_version': metadata['model_version'],
        'media_source': media_source,
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
        logger.info(f"TEST MODE: Selected {len(test_assets)} assets from each category")
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
        logger.error(f"No transformed assets found in {TRANSFORMED_BASE_DIR}")
        return

    logger.info(f"Found {len(assets)} transformed assets to verify")

    # Write CSV header using shared utility
    utils.write_csv_header(OUTPUT_CSV, header_type='c2pa_validation')

    # Process assets
    for asset_path in tqdm(assets, desc="Verifying assets", unit="asset"):
        try:
            row_data = verify_asset(asset_path)
            utils.append_csv_row(OUTPUT_CSV, row_data, header_type='c2pa_validation')
        except Exception as e:
            logger.error(f"Failed to process {asset_path.name}: {e}")
            continue

    logger.info("=" * 60)
    logger.info("C2PA Verification Complete")
    logger.info(f"  Processed: {len(assets)} assets")
    logger.info(f"  Output: {OUTPUT_CSV.absolute()}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("C2PA Verification Script - Robustness Testing")
    logger.info(f"Python version: {sys.version}")

    # Check for c2patool
    try:
        result = subprocess.run([C2PATOOL_CMD, '--version'], capture_output=True, text=True, check=True)
        logger.info(f"c2patool: {result.stdout.strip()}")
        if C2PATOOL_CMD != "c2patool":
            logger.info(f"  Using local c2patool: {C2PATOOL_CMD}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("c2patool not found! Please install c2patool or check tools/c2patool/ directory")
        sys.exit(1)

    logger.info("=" * 60)

    # Parse arguments
    test_mode = '--test' in sys.argv

    if test_mode:
        logger.info("TEST MODE: Processing one asset from each category")

    # Process assets
    process_assets(test_mode=test_mode)


if __name__ == "__main__":
    main()
