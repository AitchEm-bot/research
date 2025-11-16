"""
C2PA Verification Script for Original Signed Assets (Pre-Transformation Baseline)
==================================================================================

This script verifies C2PA manifests on all ORIGINAL signed assets (before any
transformations) to establish a baseline for comparison with post-transformation
verification results.

Purpose:
- Verify manifests on original signed images and videos in data/prepared_assets/manifests/
- Generate baseline verification metrics (manifest_present, verified, signature_valid, etc.)
- Compare with post-transformation results to measure C2PA robustness

Input Assets:
- data/prepared_assets/manifests/images/*_signed.png (100 images, SD1.4)
- data/prepared_assets/manifests/videos/internal/*_signed.mp4 (50 videos, SVD)
- data/prepared_assets/manifests/videos/external/*_signed.mp4 (60 videos, Veo3.1)

Output:
- data/results/c2pa_validation_baseline.csv

Comparison Workflow:
1. Run this script → c2pa_validation_baseline.csv (pre-transformation)
2. Run transformations → transformed assets
3. Run verify_c2pa.py → c2pa_validation.csv (post-transformation)
4. Compare baseline vs post-transformation to measure manifest loss/corruption

Usage:
    python scripts/verification/verify_original_manifests.py
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import re

from tqdm import tqdm

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import utils

# Configure logging using shared utility
logger = utils.setup_logging(log_file='data/results/logs/verify_original_manifests.log')
utils.log_environment_info()

# Configuration
MANIFESTS_BASE = Path("data/prepared_assets/manifests")
OUTPUT_CSV = Path("data/results/c2pa_validation_baseline.csv")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# c2patool path (try local installation first, then PATH)
C2PATOOL_LOCAL = Path("tools/c2patool/c2patool/c2patool.exe")
C2PATOOL_CMD = str(C2PATOOL_LOCAL) if C2PATOOL_LOCAL.exists() else "c2patool"

# CSV Column headers (matching verify_c2pa.py schema for comparison)
CSV_HEADERS = [
    'filename',
    'asset_type',
    'asset_source',          # 'internal' or 'external' (for videos)
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


def detect_media_source(filename: str) -> str:
    """
    Detect if media is internal or external based on filename pattern.

    Args:
        filename: Asset filename

    Returns:
        "internal" if filename contains 'seed', "external" otherwise
    """
    if re.search(r'seed\d+', filename):
        return "internal"
    else:
        return "external"


def extract_metadata_from_filename(filename: str, asset_type: str) -> Dict[str, str]:
    """
    Extract seed and model version from filename.

    Args:
        filename: Asset filename
        asset_type: 'image' or 'video'

    Returns:
        Dict with keys: seed, model_version, asset_source
    """
    metadata = {
        'seed': '',
        'model_version': '',
        'asset_source': ''
    }

    # Detect media source
    asset_source = detect_media_source(filename)
    metadata['asset_source'] = asset_source

    if asset_type == 'image':
        # Images are always internal (SD1.4)
        metadata['model_version'] = 'SD1.4'
        metadata['asset_source'] = 'internal'

        # Extract seed
        seed_match = re.search(r'seed(\d+)', filename)
        if seed_match:
            metadata['seed'] = seed_match.group(1)

    elif asset_type == 'video':
        # Videos can be internal (SVD) or external (Veo3.1)
        if asset_source == 'external':
            metadata['model_version'] = 'Veo3.1'
            metadata['seed'] = ''
        else:
            metadata['model_version'] = 'SVD'
            # Extract seed
            seed_match = re.search(r'seed(\d+)', filename)
            if seed_match:
                metadata['seed'] = seed_match.group(1)

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
        logger.error(f"Timeout verifying {asset_path.name}")
        return None, elapsed_ms
    except subprocess.CalledProcessError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"c2patool failed for {asset_path.name}: {e.stderr}")
        return None, elapsed_ms
    except json.JSONDecodeError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Invalid JSON from c2patool for {asset_path.name}: {e}")
        return None, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"Unexpected error verifying {asset_path.name}: {e}")
        return None, elapsed_ms


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

    # Overall verification: signature AND hash must match (INTEGRITY check)
    # Trust is informational only, not a failure metric
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


def classify_failure_reason(json_data: Optional[Dict], validation_flags: Dict) -> str:
    """
    Classify the reason for C2PA verification failure.

    Args:
        json_data: Parsed c2patool JSON output (or None)
        validation_flags: Dict with manifest_present, verified, etc.

    Returns:
        Failure reason string
    """
    # No JSON output from c2patool OR no manifests found
    # For baseline verification, this should rarely happen (original assets should have manifests)
    if json_data is None or not validation_flags['manifest_present']:
        return "manifest_dropped"

    # Manifest present, check what failed
    if validation_flags['verified']:
        return "success"

    # Manifest exists but validation failed
    if not validation_flags['signature_valid'] or \
       not validation_flags['hash_match'] or \
       not validation_flags['assertion_uris_match']:
        return "hash_or_signature_mismatch"

    # Unknown failure
    return "verification_failed_unknown"


def verify_asset(asset_path: Path, asset_type: str) -> Dict:
    """
    Verify single asset and return complete row data.

    Args:
        asset_path: Path to signed asset
        asset_type: 'image' or 'video'

    Returns:
        Dict with all CSV columns
    """
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(asset_path.name, asset_type)

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
        'asset_source': metadata['asset_source'],
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


def collect_signed_assets(test_mode: bool = False) -> list:
    """
    Collect all signed assets from data/prepared_assets/manifests/.

    Args:
        test_mode: If True, return only 3 assets (1 image, 1 internal video, 1 external video)

    Returns:
        List of (asset_path, asset_type) tuples
    """
    assets = []

    # Images
    images_dir = MANIFESTS_BASE / "images"
    if images_dir.exists():
        for img_path in sorted(images_dir.glob("*_signed.png")):
            assets.append((img_path, 'image'))
        for img_path in sorted(images_dir.glob("*_signed.jpg")):
            assets.append((img_path, 'image'))

    # Videos - Internal
    videos_internal_dir = MANIFESTS_BASE / "videos" / "internal"
    if videos_internal_dir.exists():
        for vid_path in sorted(videos_internal_dir.glob("*_signed.mp4")):
            assets.append((vid_path, 'video'))

    # Videos - External
    videos_external_dir = MANIFESTS_BASE / "videos" / "external"
    if videos_external_dir.exists():
        for vid_path in sorted(videos_external_dir.glob("*_signed.mp4")):
            assets.append((vid_path, 'video'))

    # Test mode: select 3 sample assets
    if test_mode:
        test_assets = []
        # 1 image
        images = [a for a in assets if a[1] == 'image']
        if images:
            test_assets.append(images[0])

        # 1 internal video (from videos/internal/)
        internal_videos = [a for a in assets if a[1] == 'video' and 'internal' in str(a[0])]
        if internal_videos:
            test_assets.append(internal_videos[0])

        # 1 external video (from videos/external/)
        external_videos = [a for a in assets if a[1] == 'video' and 'external' in str(a[0])]
        if external_videos:
            test_assets.append(external_videos[0])

        logger.info(f"TEST MODE: Selected 3 sample assets (1 image, 1 internal video, 1 external video)")
        return test_assets

    return assets


def process_assets(test_mode: bool = False):
    """
    Process all signed assets and generate baseline verification CSV.

    Args:
        test_mode: If True, process only 3 sample assets
    """
    assets = collect_signed_assets(test_mode=test_mode)

    if not assets:
        logger.error(f"No signed assets found in {MANIFESTS_BASE}")
        return

    logger.info(f"Found {len(assets)} signed assets to verify")

    # Count by type and source
    images = sum(1 for _, t in assets if t == 'image')
    videos = sum(1 for _, t in assets if t == 'video')

    logger.info(f"  Images: {images}")
    logger.info(f"  Videos: {videos}")
    logger.info("=" * 60)

    # Write CSV with results
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for asset_path, asset_type in tqdm(assets, desc="Verifying original assets", unit="asset"):
            try:
                row_data = verify_asset(asset_path, asset_type)
                writer.writerow(row_data)
                csvfile.flush()  # Write immediately for resumability

            except Exception as e:
                logger.error(f"Failed to process {asset_path.name}: {e}")
                continue

    logger.info("=" * 60)
    logger.info("C2PA Baseline Verification Complete")
    logger.info(f"  Processed: {len(assets)} assets")
    logger.info(f"  Output: {OUTPUT_CSV.absolute()}")
    logger.info("=" * 60)

    # Print summary statistics
    print_summary_statistics()


def print_summary_statistics():
    """
    Print summary statistics from the baseline verification results.
    """
    import pandas as pd

    try:
        df = pd.read_csv(OUTPUT_CSV)

        logger.info("")
        logger.info("Baseline Verification Summary:")
        logger.info(f"  Total assets: {len(df)}")
        logger.info(f"  Manifests present: {df['manifest_present'].sum()} / {len(df)}")
        logger.info(f"  Verified (INTEGRITY): {df['verified'].sum()} / {len(df)}")
        logger.info(f"  Signature valid: {df['signature_valid'].sum()} / {len(df)}")
        logger.info(f"  Hash match: {df['hash_match'].sum()} / {len(df)}")
        logger.info(f"  Trust verified: {df['trust_verified'].sum()} / {len(df)} (informational)")

        logger.info("")
        logger.info("By Asset Type:")
        for asset_type in df['asset_type'].unique():
            subset = df[df['asset_type'] == asset_type]
            logger.info(f"  {asset_type}:")
            logger.info(f"    Total: {len(subset)}")
            logger.info(f"    Verified: {subset['verified'].sum()} / {len(subset)}")

        logger.info("")
        logger.info("By Asset Source:")
        for source in df['asset_source'].unique():
            subset = df[df['asset_source'] == source]
            logger.info(f"  {source}:")
            logger.info(f"    Total: {len(subset)}")
            logger.info(f"    Verified: {subset['verified'].sum()} / {len(subset)}")

        logger.info("")
        logger.info("Validation State Distribution:")
        for state, count in df['validation_state'].value_counts().items():
            logger.info(f"  {state}: {count}")

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to generate summary statistics: {e}")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("C2PA Baseline Verification - Original Signed Assets")
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
        logger.info("TEST MODE: Processing 3 sample assets (1 image, 1 internal video, 1 external video)")

    # Process assets
    process_assets(test_mode=test_mode)


if __name__ == "__main__":
    main()
