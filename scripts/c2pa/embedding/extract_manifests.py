#!/usr/bin/env python3
"""
Extract C2PA manifests from signed assets and save as JSON files.

This script reads all signed images and videos in data/prepared_assets/signed_assets/, extracts their
embedded C2PA manifests using c2patool, and saves them as separate JSON files for
analysis and inspection.

Directory Structure:
  Input:  data/prepared_assets/signed_assets/images/*.png
          data/prepared_assets/signed_assets/videos/internal/*.mp4
          data/prepared_assets/signed_assets/videos/external/*.mp4

  Output: data/prepared_assets/c2pa_manifests/images/*.json
          data/prepared_assets/c2pa_manifests/videos/internal/*.json
          data/prepared_assets/c2pa_manifests/videos/external/*.json

Usage:
  python scripts/embedding/extract_manifests.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import utils

# Configure logging using shared utility
logger = utils.setup_logging()
utils.log_environment_info()

# Path to c2patool executable (use centralized path from utils)
C2PATOOL_PATH = Path(utils.C2PATOOL_CMD)

# Base directories
SIGNED_ASSETS_BASE = Path("data/prepared_assets/signed_assets")
OUTPUT_BASE = Path("data/prepared_assets/c2pa_manifests")


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
        logger.debug(f"Extracting manifest from: {asset_path.name}")

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

            logger.debug(f"SUCCESS: Saved manifest: {output_path.name}")

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


def extract_all_manifests():
    """
    Extract C2PA manifests from all signed assets in data/prepared_assets/signed_assets/.

    Scans:
    - data/prepared_assets/signed_assets/images/*.png
    - data/prepared_assets/signed_assets/videos/internal/*.mp4
    - data/prepared_assets/signed_assets/videos/external/*.mp4

    Outputs to:
    - data/prepared_assets/c2pa_manifests/images/*.json
    - data/prepared_assets/c2pa_manifests/videos/internal/*.json
    - data/prepared_assets/c2pa_manifests/videos/external/*.json
    """
    total_processed = 0
    total_failed = 0

    # Define input/output directory pairs
    extraction_tasks = [
        {
            'input_dir': SIGNED_ASSETS_BASE / "images",
            'output_dir': OUTPUT_BASE / "images",
            'category': 'Images'
        },
        {
            'input_dir': SIGNED_ASSETS_BASE / "videos" / "internal",
            'output_dir': OUTPUT_BASE / "videos" / "internal",
            'category': 'Videos (Internal)'
        },
        {
            'input_dir': SIGNED_ASSETS_BASE / "videos" / "external",
            'output_dir': OUTPUT_BASE / "videos" / "external",
            'category': 'Videos (External)'
        }
    ]

    logger.info("=" * 60)
    logger.info("C2PA Manifest Extraction")
    logger.info("=" * 60)

    for task in extraction_tasks:
        input_dir = task['input_dir']
        output_dir = task['output_dir']
        category = task['category']

        # Skip if input directory doesn't exist
        if not input_dir.exists():
            logger.warning(f"Input directory not found: {input_dir}")
            logger.warning(f"Skipping {category}")
            continue

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all signed assets
        signed_assets = sorted(input_dir.glob("*_signed.png")) + \
                       sorted(input_dir.glob("*_signed.jpg")) + \
                       sorted(input_dir.glob("*_signed.mp4")) + \
                       sorted(input_dir.glob("*_signed.avi"))

        if not signed_assets:
            logger.warning(f"No signed assets found in {input_dir}")
            logger.warning(f"Skipping {category}")
            continue

        logger.info(f"\n{category}: {len(signed_assets)} assets")
        logger.info(f"  Input:  {input_dir}")
        logger.info(f"  Output: {output_dir}")
        logger.info("-" * 60)

        # Extract manifests
        for asset_path in tqdm(signed_assets, desc=f"Extracting {category}", unit="asset"):
            # Generate output filename (replace _signed.ext with _manifest.json)
            output_filename = asset_path.stem.replace("_signed", "_manifest") + ".json"
            output_path = output_dir / output_filename

            # Extract manifest
            success = extract_manifest(asset_path, output_path)

            if success:
                total_processed += 1
            else:
                total_failed += 1

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Manifest Extraction Complete")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Output directory: {OUTPUT_BASE}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    # Check if c2patool exists
    if not C2PATOOL_PATH.exists():
        logger.error(f"c2patool not found at: {C2PATOOL_PATH}")
        logger.error("Please download c2patool from: https://github.com/contentauth/c2pa-rs/releases")
        sys.exit(1)

    logger.info(f"Using c2patool at: {C2PATOOL_PATH}")
    logger.info(f"Input directory: {SIGNED_ASSETS_BASE}")
    logger.info(f"Output directory: {OUTPUT_BASE}")

    # Extract manifests
    extract_all_manifests()


if __name__ == "__main__":
    main()
