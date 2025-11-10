"""
Image Compression Script for C2PA Robustness Testing
=====================================================

This script applies various compression transformations to signed images
to test C2PA manifest robustness under compression operations.

Transformations:
- JPEG compression at quality levels: [95, 75, 50, 25]
- PNG optimization at compress_level: [9, 0]

Research Context:
- Based on research findings, compression will likely INVALIDATE C2PA manifests
  due to hash changes, but manifest structure may survive
- This script tests which compression levels break C2PA validation and how

Usage:
    python scripts/transformations/compress_images.py [--test]

    --test: Run on single image only (smoke test)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import json

from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/transformed/transform_images.log')
    ]
)

# Configuration
INPUT_DIR = Path("data/manifests")
OUTPUT_BASE_DIR = Path("data/transformed/compression/images")
JPEG_QUALITIES = [95, 75, 50, 25]
PNG_COMPRESS_LEVELS = [9, 0]  # 9 = max compression, 0 = no compression

# Generate output directories for each quality/compress level
def get_jpeg_output_dir(quality: int) -> Path:
    return OUTPUT_BASE_DIR / "jpeg" / f"q{quality}"

def get_png_output_dir(compress_level: int) -> Path:
    return OUTPUT_BASE_DIR / "png" / f"c{compress_level}"


def compress_jpeg(image_path: Path, quality: int, output_dir: Path) -> Tuple[Path, bool]:
    """
    Compress image to JPEG at specified quality level.

    Args:
        image_path: Path to source image
        quality: JPEG quality (1-100, higher = better quality)
        output_dir: Output directory

    Returns:
        Tuple of (output_path, success_status)
    """
    try:
        # Generate output filename
        stem = image_path.stem.replace('_signed', '')  # Remove '_signed' suffix
        output_path = output_dir / f"{stem}_jpeg_q{quality}.jpg"

        # Open and convert image
        img = Image.open(image_path)

        # Convert to RGB if necessary (JPEG doesn't support transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparent images
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save with specified quality
        img.save(output_path, 'JPEG', quality=quality, optimize=True)

        # Get file size for logging
        size_kb = output_path.stat().st_size / 1024
        logging.debug(f"Created {output_path.name} ({size_kb:.1f} KB)")

        return output_path, True

    except Exception as e:
        logging.error(f"Failed to compress {image_path.name} to JPEG q{quality}: {e}")
        return None, False


def compress_png(image_path: Path, compress_level: int, output_dir: Path) -> Tuple[Path, bool]:
    """
    Compress image to PNG at specified compression level.

    Args:
        image_path: Path to source image
        compress_level: PNG compression level (0-9, higher = smaller file)
        output_dir: Output directory

    Returns:
        Tuple of (output_path, success_status)
    """
    try:
        # Generate output filename
        stem = image_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_png_c{compress_level}.png"

        # Open image
        img = Image.open(image_path)

        # Save with specified compression level
        img.save(output_path, 'PNG', compress_level=compress_level, optimize=True)

        # Get file size for logging
        size_kb = output_path.stat().st_size / 1024
        logging.debug(f"Created {output_path.name} ({size_kb:.1f} KB)")

        return output_path, True

    except Exception as e:
        logging.error(f"Failed to compress {image_path.name} to PNG c{compress_level}: {e}")
        return None, False


def save_metadata(output_path: Path, source_path: Path, transform_type: str, params: dict):
    """
    Save transformation metadata as JSON sidecar.

    Args:
        output_path: Path to transformed image
        source_path: Path to source image
        transform_type: Type of transformation (e.g., 'jpeg_compression')
        params: Transformation parameters
    """
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


def process_images(test_mode: bool = False):
    """
    Process all signed images with compression transformations.

    Args:
        test_mode: If True, process only first image (smoke test)
    """
    # Find all signed PNG images
    signed_images = sorted(INPUT_DIR.glob("*_signed.png"))

    if not signed_images:
        logging.error(f"No signed images found in {INPUT_DIR}")
        return

    if test_mode:
        signed_images = signed_images[:1]
        logging.info("TEST MODE: Processing only first image")

    logging.info(f"Found {len(signed_images)} signed image(s) to process")
    logging.info(f"JPEG quality levels: {JPEG_QUALITIES}")
    logging.info(f"PNG compression levels: {PNG_COMPRESS_LEVELS}")

    # Create output directories for each quality/compress level
    for quality in JPEG_QUALITIES:
        get_jpeg_output_dir(quality).mkdir(parents=True, exist_ok=True)
    for compress_level in PNG_COMPRESS_LEVELS:
        get_png_output_dir(compress_level).mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_input': len(signed_images),
        'jpeg_success': 0,
        'jpeg_failed': 0,
        'png_success': 0,
        'png_failed': 0
    }

    # Process each image
    for img_path in tqdm(signed_images, desc="Processing images", unit="image"):
        logging.info(f"Processing: {img_path.name}")

        # Apply JPEG compression at various quality levels
        for quality in JPEG_QUALITIES:
            output_dir = get_jpeg_output_dir(quality)
            output_path, success = compress_jpeg(img_path, quality, output_dir)
            if success:
                stats['jpeg_success'] += 1
                save_metadata(output_path, img_path, 'jpeg_compression', {'quality': quality})
            else:
                stats['jpeg_failed'] += 1

        # Apply PNG compression at various levels
        for compress_level in PNG_COMPRESS_LEVELS:
            output_dir = get_png_output_dir(compress_level)
            output_path, success = compress_png(img_path, compress_level, output_dir)
            if success:
                stats['png_success'] += 1
                save_metadata(output_path, img_path, 'png_compression', {'compress_level': compress_level})
            else:
                stats['png_failed'] += 1

    # Print summary
    logging.info("=" * 60)
    logging.info("Image Compression Complete")
    logging.info(f"  Input images: {stats['total_input']}")
    logging.info(f"  JPEG compressions: {stats['jpeg_success']} succeeded, {stats['jpeg_failed']} failed")
    logging.info(f"  PNG compressions: {stats['png_success']} succeeded, {stats['png_failed']} failed")
    logging.info(f"  Total output files: {stats['jpeg_success'] + stats['png_success']}")
    logging.info(f"  Output base directory: {OUTPUT_BASE_DIR.absolute()}")
    logging.info(f"    JPEG quality folders: q95/, q75/, q50/, q25/")
    logging.info(f"    PNG compress folders: c9/, c0/")
    logging.info("=" * 60)

    # Expected output calculation
    expected_output = len(signed_images) * (len(JPEG_QUALITIES) + len(PNG_COMPRESS_LEVELS))
    actual_output = stats['jpeg_success'] + stats['png_success']

    if actual_output == expected_output:
        logging.info("✅ All transformations completed successfully!")
    else:
        logging.warning(f"⚠️ Expected {expected_output} files, got {actual_output}")


def main():
    """Main entry point."""
    # Print environment info
    logging.info("=" * 60)
    logging.info("Image Compression Script - C2PA Robustness Testing")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"PIL/Pillow version: {Image.__version__ if hasattr(Image, '__version__') else 'Unknown'}")
    logging.info("=" * 60)

    # Check for test mode flag
    test_mode = '--test' in sys.argv

    # Process images
    process_images(test_mode=test_mode)


if __name__ == "__main__":
    main()
