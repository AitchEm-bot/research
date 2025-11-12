"""
Platform Returns Processing Script for C2PA Robustness Testing
==============================================================

This script processes assets downloaded from social media platforms after
manual upload/download round-trip testing. It verifies C2PA signatures,
calculates quality metrics, and merges with manual logging data.

Workflow:
1. Scan platform returned/ folders for downloaded files
2. Parse filename convention to extract metadata
3. Run C2PA verification using verify_c2pa.py logic
4. Calculate quality metrics using calculate_quality_metrics.py logic
5. Load manual CSV log (platform_manifest.csv)
6. Join automated metrics with manual logs
7. Generate platform_results.csv

Research Context:
- Tests real-world C2PA manifest persistence through platform round-trips
- Most platforms are expected to STRIP C2PA manifests during transcoding
- Quality degradation varies by platform compression policies

Usage:
    python scripts/platform/process_platform_returns.py

    Processes all files in platform_tests/*/returned/ folders
    Outputs to: data/results/platform_results.csv

File Naming Convention (from prepare_platform_uploads.py):
    {original}__{platform}__{mode}__{timestamp}.{ext}
    Example: seed42_h264_bitrate2000k__instagram__reel__20250112-143022.mp4
"""

import logging
import sys
import subprocess
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import time

import cv2
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/platform_tests/processing.log')
    ]
)

# Configuration
PLATFORM_TESTS_BASE = Path("data/platform_tests")
RESULTS_DIR = Path("data/results")
MANIFESTS_IMAGES_DIR = Path("data/manifests/images")
MANIFESTS_VIDEOS_DIRS = [
    Path("data/manifests/videos/internal"),
    Path("data/manifests/videos/external")
]
TRANSFORMED_BASE = Path("data/transformed")

# Filename pattern: {original}__{platform}__{mode}__{timestamp}.{ext}
FILENAME_PATTERN = re.compile(r'^(.+)__([^_]+)__([^_]+)__(\d{8}-\d{6})\.(mp4|jpg|png|mov|avi)$')


def parse_filename(filename: str) -> Optional[Dict]:
    """
    Parse platform return filename to extract metadata.

    Args:
        filename: Filename following convention

    Returns:
        Dict with parsed metadata or None
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        logging.warning(f"Filename does not match convention: {filename}")
        return None

    return {
        'original_filename': match.group(1),
        'platform': match.group(2),
        'mode': match.group(3),
        'timestamp': match.group(4),
        'extension': match.group(5)
    }


def find_original_asset(base_filename: str, asset_type: str) -> Optional[Path]:
    """
    Find the original signed asset to compare against.

    Args:
        base_filename: Base filename without platform suffix
        asset_type: 'image' or 'video'

    Returns:
        Path to original signed asset or None
    """
    # Try to extract seed/transform info from filename
    # e.g., "seed42_h264_bitrate2000k" -> need "seed42_signed.mp4"

    # Extract seed pattern
    seed_match = re.search(r'(seed\d+)', base_filename)
    if not seed_match:
        logging.warning(f"Cannot extract seed from: {base_filename}")
        return None

    seed_base = seed_match.group(1)

    # Search in manifests
    if asset_type == 'image':
        original_filename = f"{seed_base}_signed.png"
        original_path = MANIFESTS_IMAGES_DIR / original_filename
        if original_path.exists():
            return original_path
    else:
        original_filename = f"{seed_base}_signed.mp4"
        for video_dir in MANIFESTS_VIDEOS_DIRS:
            original_path = video_dir / original_filename
            if original_path.exists():
                return original_path

    logging.warning(f"Original asset not found for: {base_filename}")
    return None


def verify_c2pa(file_path: Path) -> Dict:
    """
    Verify C2PA manifest (reused from verify_c2pa.py).

    Args:
        file_path: Path to file

    Returns:
        Dict with verification results
    """
    start_time = time.time()

    try:
        cmd = [
            'c2patool',
            str(file_path),
            '--output', 'json'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        elapsed_ms = (time.time() - start_time) * 1000

        if result.returncode == 0:
            try:
                manifest_data = json.loads(result.stdout)
                active_manifest = manifest_data.get('active_manifest')

                if active_manifest:
                    # Extract verification metrics
                    claim_signature = active_manifest.get('claim_signature', {})
                    signature_valid = 1 if claim_signature.get('validated') else 0

                    # Check hash match (both dataHash and bmffHash)
                    hash_match = 0
                    assertions = active_manifest.get('assertions', [])
                    for assertion in assertions:
                        if 'dataHash' in assertion:
                            if assertion['dataHash'].get('match', False):
                                hash_match = 1
                        elif 'bmffHash' in assertion:
                            if assertion['bmffHash'].get('match', False):
                                hash_match = 1

                    # Check assertion URIs
                    assertion_uris_match = 1
                    for assertion in assertions:
                        if 'hashedURI' in assertion:
                            if not assertion['hashedURI'].get('match', False):
                                assertion_uris_match = 0
                                break

                    # Overall verification
                    verified = 1 if signature_valid and hash_match else 0

                    # Trust status (informational)
                    signing_credential = active_manifest.get('signingCredential', {})
                    trust_verified = 0 if 'untrusted' in signing_credential else 1

                    validation_state = claim_signature.get('validation_state', 'unknown')

                    return {
                        'manifest_present': 1,
                        'verified': verified,
                        'signature_valid': signature_valid,
                        'hash_match': hash_match,
                        'assertion_uris_match': assertion_uris_match,
                        'trust_verified': trust_verified,
                        'validation_state': validation_state,
                        'failure_reason': '' if verified else 'Hash or signature mismatch',
                        'processing_time_ms': elapsed_ms
                    }

            except json.JSONDecodeError:
                pass

        # No manifest or invalid
        return {
            'manifest_present': 0,
            'verified': 0,
            'signature_valid': 0,
            'hash_match': 0,
            'assertion_uris_match': 0,
            'trust_verified': 0,
            'validation_state': 'no_manifest',
            'failure_reason': 'No C2PA manifest found',
            'processing_time_ms': elapsed_ms
        }

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logging.error(f"C2PA verification failed for {file_path.name}: {e}")
        return {
            'manifest_present': 0,
            'verified': 0,
            'signature_valid': 0,
            'hash_match': 0,
            'assertion_uris_match': 0,
            'trust_verified': 0,
            'validation_state': 'error',
            'failure_reason': f'Verification error: {str(e)[:100]}',
            'processing_time_ms': elapsed_ms
        }


def calculate_image_metrics(original_path: Path, platform_path: Path) -> Dict:
    """
    Calculate PSNR and SSIM for images (reused from calculate_quality_metrics.py).

    Args:
        original_path: Path to original signed image
        platform_path: Path to platform-returned image

    Returns:
        Dict with quality metrics
    """
    start_time = time.time()

    try:
        # Load images
        img1 = cv2.imread(str(original_path))
        img2 = cv2.imread(str(platform_path))

        if img1 is None or img2 is None:
            return {
                'psnr': 'NA',
                'ssim': 'NA',
                'vmaf': 'NA',
                'lossless_match': 0,
                'processing_time_ms': (time.time() - start_time) * 1000
            }

        # Resize if dimensions don't match
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Check for lossless match
        lossless_match = 0
        if np.array_equal(img1, img2):
            psnr = "inf"
            lossless_match = 1
        else:
            psnr_value = cv2.PSNR(img1, img2)
            psnr = f"{psnr_value:.4f}"

        # Calculate SSIM
        from skimage.metrics import structural_similarity
        ssim_value = structural_similarity(img1, img2, channel_axis=2)
        ssim = f"{ssim_value:.6f}"

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            'psnr': psnr,
            'ssim': ssim,
            'vmaf': 'NA',
            'lossless_match': lossless_match,
            'processing_time_ms': elapsed_ms
        }

    except Exception as e:
        logging.error(f"Image metrics calculation failed: {e}")
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            'psnr': 'NA',
            'ssim': 'NA',
            'vmaf': 'NA',
            'lossless_match': 0,
            'processing_time_ms': elapsed_ms
        }


def calculate_video_vmaf(original_path: Path, platform_path: Path) -> Dict:
    """
    Calculate VMAF for videos (reused from calculate_quality_metrics.py).

    Args:
        original_path: Path to original signed video
        platform_path: Path to platform-returned video

    Returns:
        Dict with VMAF score
    """
    start_time = time.time()

    try:
        # Get video properties using ffprobe
        def get_video_properties(video_path: Path) -> Optional[Tuple[int, int, str]]:
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
            parts = output.split(',')
            return int(parts[0]), int(parts[1]), parts[2]

        ref_width, ref_height, ref_fps = get_video_properties(original_path)
        dist_width, dist_height, dist_fps = get_video_properties(platform_path)

        # Build ffmpeg filter
        if (dist_width != ref_width or dist_height != ref_height or dist_fps != ref_fps):
            filter_chain = (
                f"[0:v]scale={ref_width}:{ref_height}:flags=lanczos,"
                f"fps={ref_fps},format=yuv420p,setpts=PTS-STARTPTS[dist];"
                f"[1:v]format=yuv420p,setpts=PTS-STARTPTS[ref];"
                f"[dist][ref]libvmaf=log_fmt=json:log_path=NUL"
            )
        else:
            filter_chain = "[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf=log_fmt=json:log_path=NUL"

        # Run ffmpeg with VMAF
        cmd = [
            'ffmpeg',
            '-i', str(platform_path),  # distorted
            '-i', str(original_path),  # reference
            '-filter_complex', filter_chain,
            '-f', 'null',
            '-'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse VMAF from stderr
        vmaf_match = re.search(r'VMAF score:\s*([\d.]+)', result.stderr)
        if vmaf_match:
            vmaf = f"{float(vmaf_match.group(1)):.4f}"
        else:
            vmaf = 'NA'

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            'psnr': 'NA',
            'ssim': 'NA',
            'vmaf': vmaf,
            'lossless_match': 0,
            'processing_time_ms': elapsed_ms
        }

    except Exception as e:
        logging.error(f"VMAF calculation failed: {e}")
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            'psnr': 'NA',
            'ssim': 'NA',
            'vmaf': 'NA',
            'lossless_match': 0,
            'processing_time_ms': elapsed_ms
        }


def load_manual_csv() -> pd.DataFrame:
    """
    Load manual platform_manifest.csv log.

    Returns:
        DataFrame with manual log data
    """
    csv_path = PLATFORM_TESTS_BASE / "platform_manifest.csv"

    if not csv_path.exists():
        logging.warning("platform_manifest.csv not found - returning empty DataFrame")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows from platform_manifest.csv")
        return df
    except Exception as e:
        logging.error(f"Failed to load platform_manifest.csv: {e}")
        return pd.DataFrame()


def process_platform_returns() -> List[Dict]:
    """
    Process all platform return files.

    Returns:
        List of result dictionaries
    """
    results = []

    # Scan all platform returned folders
    platforms = ['instagram', 'twitter', 'facebook', 'youtube_shorts', 'tiktok', 'whatsapp']

    for platform in platforms:
        returned_dir = PLATFORM_TESTS_BASE / platform / "returned"

        if not returned_dir.exists():
            continue

        # Find all returned files
        returned_files = list(returned_dir.glob("*.*"))
        returned_files = [f for f in returned_files if f.suffix.lower() in ['.mp4', '.jpg', '.png', '.mov', '.avi']]

        if not returned_files:
            continue

        logging.info(f"Processing {len(returned_files)} files from {platform}")

        for file_path in returned_files:
            logging.info(f"  Processing: {file_path.name}")

            # Parse filename
            parsed = parse_filename(file_path.name)
            if not parsed:
                logging.warning(f"  Skipping (invalid filename): {file_path.name}")
                continue

            # Determine asset type
            asset_type = 'video' if parsed['extension'] in ['mp4', 'mov', 'avi'] else 'image'

            # Find original signed asset
            original_path = find_original_asset(parsed['original_filename'], asset_type)
            if not original_path:
                logging.warning(f"  Skipping (original not found): {file_path.name}")
                continue

            # Run C2PA verification
            c2pa_results = verify_c2pa(file_path)

            # Calculate quality metrics
            if asset_type == 'image':
                quality_results = calculate_image_metrics(original_path, file_path)
            else:
                quality_results = calculate_video_vmaf(original_path, file_path)

            # Compile result
            result = {
                'filename': file_path.name,
                'asset_type': asset_type,
                'transform_type': 'platform_roundtrip',
                'transform_level': platform,
                'platform': platform,
                'platform_mode': parsed['mode'],
                'video_source': 'internal',  # Default; will be updated from CSV if available
                'original_filename': parsed['original_filename'],
                'download_timestamp': parsed['timestamp'],
                'seed': '',  # Will extract from filename
                'model_version': '',  # Will extract from filename

                # C2PA metrics
                'manifest_present': c2pa_results['manifest_present'],
                'verified': c2pa_results['verified'],
                'signature_valid': c2pa_results['signature_valid'],
                'hash_match': c2pa_results['hash_match'],
                'assertion_uris_match': c2pa_results['assertion_uris_match'],
                'trust_verified': c2pa_results['trust_verified'],
                'validation_state': c2pa_results['validation_state'],
                'failure_reason': c2pa_results['failure_reason'],
                'c2pa_processing_time_ms': f"{c2pa_results['processing_time_ms']:.2f}",

                # Quality metrics
                'psnr': quality_results['psnr'],
                'ssim': quality_results['ssim'],
                'vmaf': quality_results['vmaf'],
                'lossless_match': quality_results['lossless_match'],
                'lossless_transform': 0,
                'quality_processing_time_ms': f"{quality_results['processing_time_ms']:.2f}",

                # Timestamp
                'timestamp': datetime.now().isoformat()
            }

            # Extract seed from filename
            seed_match = re.search(r'seed(\d+)', parsed['original_filename'])
            if seed_match:
                result['seed'] = seed_match.group(1)

            # Model version (assume SVD for videos, SD for images)
            result['model_version'] = 'SVD' if asset_type == 'video' else 'SD1.4'

            results.append(result)

    return results


def merge_with_manual_csv(results: List[Dict], manual_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge automated results with manual CSV log.

    Args:
        results: List of automated result dicts
        manual_df: Manual CSV DataFrame

    Returns:
        Merged DataFrame
    """
    results_df = pd.DataFrame(results)

    if manual_df.empty:
        logging.warning("No manual CSV data to merge")
        return results_df

    # Merge on filename match (original_filename in results, original_filename in manual_df)
    # Manual CSV has: original_filename, platform, mode, upload_timestamp, download_timestamp, notes

    # Create join key
    results_df['join_key'] = results_df['original_filename'] + '__' + results_df['platform'] + '__' + results_df['platform_mode']
    manual_df['join_key'] = manual_df['original_filename'] + '__' + manual_df['platform'] + '__' + manual_df['mode']

    # Merge
    merged_df = results_df.merge(
        manual_df[['join_key', 'upload_timestamp', 'notes', 'video_source']],
        on='join_key',
        how='left'
    )

    # Update upload_timestamp and video_source if available
    if 'upload_timestamp' in merged_df.columns:
        results_df['upload_timestamp'] = merged_df['upload_timestamp']
    if 'video_source' in merged_df.columns:
        results_df['video_source'] = merged_df['video_source'].fillna('internal')

    # Drop join key
    results_df.drop(columns=['join_key'], inplace=True, errors='ignore')

    return results_df


def main():
    """Main entry point."""
    logging.info("="*60)
    logging.info("Platform Returns Processing - C2PA Robustness Testing")
    logging.info(f"Python version: {sys.version}")
    logging.info("="*60)

    # Check dependencies
    try:
        subprocess.run(['c2patool', '--version'], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logging.error("c2patool not found! Please install c2patool.")
        sys.exit(1)

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logging.error("ffmpeg not found! Please install ffmpeg with libvmaf support.")
        sys.exit(1)

    # Process platform returns
    logging.info("Processing platform return files...")
    results = process_platform_returns()

    if not results:
        logging.warning("No platform return files found to process!")
        logging.info("Expected files in: data/platform_tests/*/returned/")
        logging.info("File naming convention: {original}__{platform}__{mode}__{timestamp}.{ext}")
        sys.exit(0)

    logging.info(f"Processed {len(results)} platform return files")

    # Load manual CSV
    logging.info("Loading manual CSV log...")
    manual_df = load_manual_csv()

    # Merge with manual data
    logging.info("Merging with manual log data...")
    final_df = merge_with_manual_csv(results, manual_df)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "platform_results.csv"

    final_df.to_csv(output_path, index=False)
    logging.info(f"Saved results to: {output_path}")
    logging.info(f"  Total rows: {len(final_df)}")
    logging.info(f"  Columns: {len(final_df.columns)}")

    # Print summary
    logging.info("="*60)
    logging.info("Platform Results Summary")
    logging.info(f"  Total files processed: {len(final_df)}")
    logging.info(f"  Manifests present: {final_df['manifest_present'].sum()}")
    logging.info(f"  Verified: {final_df['verified'].sum()}")
    logging.info(f"  Manifest stripped: {len(final_df) - final_df['manifest_present'].sum()}")
    logging.info("="*60)

    # Per-platform summary
    logging.info("Per-Platform Breakdown:")
    for platform in final_df['platform'].unique():
        platform_df = final_df[final_df['platform'] == platform]
        logging.info(f"  {platform}:")
        logging.info(f"    Files: {len(platform_df)}")
        logging.info(f"    Manifests present: {platform_df['manifest_present'].sum()}")
        logging.info(f"    Verified: {platform_df['verified'].sum()}")

    logging.info("="*60)
    logging.info("Next step: Run scripts/metrics/merge_results.py to merge with final_metrics.csv")


if __name__ == "__main__":
    main()
