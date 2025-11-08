#!/usr/bin/env python3
"""
Extract C2PA manifests from signed assets and save as JSON files.

This script reads all signed images and videos in data/manifests/, extracts their
embedded C2PA manifests using c2patool, and saves them as separate JSON files for
analysis and inspection.

Usage:
  python extract_manifests.py --input-dir data/manifests/ --output-dir data/manifests_json/
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# Log environment info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to c2patool executable
C2PATOOL_PATH = Path("tools/c2patool/c2patool/c2patool.exe")


def extract_manifest(asset_path: Path, output_path: Path) -> bool:
    """
    Extract C2PA manifest from a signed asset using c2patool.

    Args:
        asset_path: Path to signed asset (image or video)
        output_path: Path to save extracted manifest JSON

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting manifest from: {asset_path.name}")

        # Call c2patool to read the manifest
        cmd = [str(C2PATOOL_PATH), str(asset_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"c2patool failed with exit code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return False

        # Parse the JSON output
        try:
            manifest_data = json.loads(result.stdout)

            # Save manifest to output file
            with open(output_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)

            logger.info(f"✅ Saved manifest: {output_path.name}")

            # Log basic info about the manifest
            active_manifest = manifest_data.get("active_manifest", "N/A")
            validation_state = manifest_data.get("validation_state", "Unknown")
            logger.info(f"   Active Manifest: {active_manifest}")
            logger.info(f"   Validation State: {validation_state}")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse c2patool JSON output: {e}")
            logger.error(f"stdout: {result.stdout}")
            return False

    except Exception as e:
        logger.error(f"Failed to extract manifest from {asset_path.name}: {e}")
        logger.exception("Detailed error:")
        return False


def extract_all_manifests(input_dir: Path, output_dir: Path):
    """
    Extract C2PA manifests from all signed assets in the input directory.

    Args:
        input_dir: Directory containing signed assets
        output_dir: Directory to save extracted manifest JSON files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0

    # Find all signed assets
    signed_images = sorted(input_dir.glob("*_signed.png")) + sorted(input_dir.glob("*_signed.jpg"))
    signed_videos = sorted(input_dir.glob("*_signed.mp4")) + sorted(input_dir.glob("*_signed.avi"))

    all_signed_assets = signed_images + signed_videos

    if not all_signed_assets:
        logger.warning(f"No signed assets found in {input_dir}")
        return

    logger.info(f"Found {len(all_signed_assets)} signed assets:")
    logger.info(f"  Images: {len(signed_images)}")
    logger.info(f"  Videos: {len(signed_videos)}")
    logger.info("=" * 60)

    # Extract manifests
    for asset_path in all_signed_assets:
        # Generate output filename (replace _signed.ext with _manifest.json)
        output_filename = asset_path.stem.replace("_signed", "_manifest") + ".json"
        output_path = output_dir / output_filename

        # Extract manifest
        success = extract_manifest(asset_path, output_path)

        if success:
            total_processed += 1
        else:
            total_failed += 1

        logger.info("")  # Blank line for readability

    logger.info("=" * 60)
    logger.info(f"Manifest Extraction Complete")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract C2PA manifests from signed assets to JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory containing signed assets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests_json"),
        help="Output directory for extracted manifest JSON files"
    )

    args = parser.parse_args()

    # Check if c2patool exists
    if not C2PATOOL_PATH.exists():
        logger.error(f"c2patool not found at: {C2PATOOL_PATH}")
        logger.error("Please download c2patool from: https://github.com/contentauth/c2pa-rs/releases")
        sys.exit(1)

    logger.info(f"Using c2patool at: {C2PATOOL_PATH}")

    # Extract manifests
    extract_all_manifests(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
