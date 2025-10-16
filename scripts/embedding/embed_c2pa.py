#!/usr/bin/env python3
"""
Embed C2PA content credentials (manifests) into images and videos.

This script creates C2PA manifests for AI-generated images and videos, embedding
cryptographic signatures and metadata for provenance tracking. The manifests include
information about the content's origin, creation details, and digital signatures.

C2PA Specification:
  Coalition for Content Provenance and Authenticity (C2PA) Technical Specification
  https://c2pa.org/specifications/specifications/1.0/index.html
  (Industry standard specification, not peer-reviewed academic work)

Usage:
  python embed_c2pa.py --images-dir data/raw_images/ --videos-dir data/raw_videos/ --output-dir data/manifests/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Log environment info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Attempt to import c2pa-python library
C2PA_AVAILABLE = False
try:
    import c2pa
    C2PA_AVAILABLE = True
    logger.info("c2pa-python library loaded successfully")
except ImportError:
    logger.warning("c2pa-python not available - using fallback shim")
    logger.warning("Install with: pip install c2pa-python")
    logger.warning("Fallback: Will create manifest JSON files without real signing")


def log_environment():
    """Log Python and C2PA environment information."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"C2PA library available: {C2PA_AVAILABLE}")
    if C2PA_AVAILABLE:
        try:
            logger.info(f"c2pa version: {c2pa.__version__}")
        except AttributeError:
            logger.info("c2pa version: unknown")


def create_test_keypair(output_dir: Path) -> tuple[Path, Path]:
    """
    Create or locate test keypair for C2PA signing.

    In production, use proper key management (HSM, key vault, etc.).
    For testing, we create placeholder key files.

    Args:
        output_dir: Directory to store test keys

    Returns:
        Tuple of (private_key_path, certificate_path)

    TODO: Replace with actual key generation when c2pa-python is available
    """
    keys_dir = output_dir / "test_keys"
    keys_dir.mkdir(parents=True, exist_ok=True)

    private_key_path = keys_dir / "test_private.key"
    cert_path = keys_dir / "test_cert.pem"

    if not private_key_path.exists():
        logger.info("Creating test keypair placeholders...")
        # In real implementation, use c2pa key generation or openssl
        private_key_path.write_text("# Test private key placeholder\n# TODO: Generate real key")
        cert_path.write_text("# Test certificate placeholder\n# TODO: Generate real certificate")
        logger.warning("Created placeholder keys - NOT FOR PRODUCTION USE")

    return private_key_path, cert_path


def create_manifest_metadata(asset_path: Path, asset_type: str,
                             seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create C2PA manifest metadata structure.

    Args:
        asset_path: Path to the asset file
        asset_type: Type of asset ('image' or 'video')
        seed: Random seed used to generate the asset (if available)

    Returns:
        Dictionary containing manifest metadata
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Read asset metadata from accompanying JSON file if it exists
    metadata_path = asset_path.with_suffix('.json')
    asset_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            asset_metadata = json.load(f)

    # Build C2PA manifest structure (following C2PA spec)
    manifest = {
        "claim_generator": "research-c2pa-pipeline/0.1.0",
        "claim_generator_info": [
            {
                "name": "C2PA Robustness Research Pipeline",
                "version": "0.1.0"
            }
        ],
        "title": f"AI-generated {asset_type}",
        "format": asset_path.suffix[1:].upper(),  # e.g., 'PNG', 'MP4'
        "instance_id": f"xmp:iid:{asset_path.stem}",
        "document_id": f"xmp:did:{asset_path.stem}",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "when": timestamp,
                            "software_agent": asset_metadata.get("model", "unknown"),
                            "parameters": {
                                "seed": seed or asset_metadata.get("seed", "unknown"),
                                "model_version": asset_metadata.get("model_version", "unknown")
                            }
                        }
                    ]
                }
            },
            {
                "label": "c2pa.hash.data",
                "data": {
                    "algorithm": "sha256",
                    "hash": "placeholder_hash",  # TODO: Compute actual hash
                    "name": asset_path.name
                }
            }
        ],
        "signature_info": {
            "issuer": "Test CA - Research Use Only",
            "time": timestamp,
            "cert_serial_number": "placeholder_serial"
        },
        "asset_metadata": asset_metadata,
        "created": timestamp,
        "modified": timestamp
    }

    return manifest


def embed_c2pa_real(asset_path: Path, output_path: Path, manifest: Dict[str, Any],
                    private_key: Path, certificate: Path) -> bool:
    """
    Embed C2PA manifest using the real c2pa-python library.

    TODO: Implement this function when c2pa-python is available and properly installed.
    This is a placeholder showing the intended API structure.

    Args:
        asset_path: Input asset file
        output_path: Output asset file with embedded manifest
        manifest: Manifest metadata dictionary
        private_key: Path to signing private key
        certificate: Path to signing certificate

    Returns:
        True if successful, False otherwise
    """
    if not C2PA_AVAILABLE:
        logger.error("c2pa library not available for real embedding")
        return False

    try:
        # TODO: Replace with actual c2pa-python API calls
        # Example (API may differ - check c2pa-python documentation):
        #
        # builder = c2pa.Builder(manifest)
        # builder.sign(private_key=str(private_key), cert=str(certificate))
        # builder.embed(str(asset_path), str(output_path))

        logger.error("Real C2PA embedding not yet implemented")
        logger.error("TODO: Implement c2pa.Builder().sign().embed() workflow")
        return False

    except Exception as e:
        logger.error(f"Failed to embed C2PA manifest: {e}")
        return False


def embed_c2pa_shim(asset_path: Path, manifest_output: Path,
                   manifest: Dict[str, Any]) -> bool:
    """
    Fallback shim: Save manifest as JSON file alongside asset.

    This is used when c2pa-python is not available. The manifest is saved as a
    separate JSON file that can be validated later when the library is available.

    Args:
        asset_path: Input asset file
        manifest_output: Output path for manifest JSON
        manifest: Manifest metadata dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)

        # Add note about shim usage
        manifest["_note"] = (
            "This manifest was created by the fallback shim. "
            "TODO: Re-generate with real c2pa-python library for proper signing."
        )

        with open(manifest_output, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved manifest shim: {manifest_output}")
        return True

    except Exception as e:
        logger.error(f"Failed to save manifest shim: {e}")
        return False


def process_assets(images_dir: Path, videos_dir: Path, output_dir: Path,
                  use_real_c2pa: bool = False):
    """
    Process all images and videos, creating C2PA manifests.

    Args:
        images_dir: Directory containing raw images
        videos_dir: Directory containing raw videos
        output_dir: Directory to save manifests
        use_real_c2pa: If True, attempt real C2PA embedding (requires c2pa-python)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create test keypair
    private_key, certificate = create_test_keypair(output_dir)

    total_processed = 0
    total_failed = 0

    # Process images
    if images_dir.exists():
        logger.info(f"Processing images from {images_dir}")
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))

        for img_path in image_files:
            logger.info(f"Processing image: {img_path.name}")

            # Extract seed from filename if present
            seed = None
            if "seed" in img_path.stem:
                try:
                    seed_str = img_path.stem.split("seed")[1].split("_")[0]
                    seed = int(seed_str)
                except (IndexError, ValueError):
                    pass

            # Create manifest
            manifest = create_manifest_metadata(img_path, "image", seed)

            # Embed or save manifest
            manifest_path = output_dir / f"{img_path.stem}_manifest.json"

            if use_real_c2pa and C2PA_AVAILABLE:
                # TODO: Enable when real C2PA is implemented
                success = embed_c2pa_real(img_path, img_path, manifest,
                                         private_key, certificate)
            else:
                success = embed_c2pa_shim(img_path, manifest_path, manifest)

            if success:
                total_processed += 1
            else:
                total_failed += 1

    # Process videos
    if videos_dir.exists():
        logger.info(f"Processing videos from {videos_dir}")
        video_files = sorted(videos_dir.glob("*.mp4")) + sorted(videos_dir.glob("*.avi"))

        for vid_path in video_files:
            logger.info(f"Processing video: {vid_path.name}")

            # Extract seed from filename if present
            seed = None
            if "seed" in vid_path.stem:
                try:
                    seed_str = vid_path.stem.split("seed")[1].split("_")[0]
                    seed = int(seed_str)
                except (IndexError, ValueError):
                    pass

            # Create manifest
            manifest = create_manifest_metadata(vid_path, "video", seed)

            # Embed or save manifest
            manifest_path = output_dir / f"{vid_path.stem}_manifest.json"

            if use_real_c2pa and C2PA_AVAILABLE:
                # TODO: Enable when real C2PA is implemented
                success = embed_c2pa_real(vid_path, vid_path, manifest,
                                         private_key, certificate)
            else:
                success = embed_c2pa_shim(vid_path, manifest_path, manifest)

            if success:
                total_processed += 1
            else:
                total_failed += 1

    logger.info("=" * 60)
    logger.info(f"Manifest generation complete")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Embed C2PA content credentials into images and videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw_images"),
        help="Directory containing raw images"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/raw_videos"),
        help="Directory containing raw videos"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Output directory for C2PA manifests"
    )
    parser.add_argument(
        "--use-real-c2pa",
        action="store_true",
        help="Attempt to use real c2pa-python library (if available)"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    if not C2PA_AVAILABLE:
        logger.warning("=" * 60)
        logger.warning("c2pa-python library is NOT available")
        logger.warning("Using fallback shim - manifests will be JSON files only")
        logger.warning("Install c2pa-python for real signing:")
        logger.warning("  pip install c2pa-python")
        logger.warning("=" * 60)

    # Process assets
    process_assets(
        images_dir=args.images_dir,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        use_real_c2pa=args.use_real_c2pa
    )


if __name__ == "__main__":
    main()
