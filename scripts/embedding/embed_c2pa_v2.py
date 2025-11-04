#!/usr/bin/env python3
"""
Embed C2PA content credentials (manifests) into images and videos using c2patool.

This script creates C2PA manifests for AI-generated images and videos, embedding
cryptographic signatures and metadata for provenance tracking using the official
c2patool CLI from contentauth/c2pa-rs.

C2PA Specification:
  Coalition for Content Provenance and Authenticity (C2PA) Technical Specification
  https://c2pa.org/specifications/specifications/1.0/index.html
  (Industry standard specification, not peer-reviewed academic work)

Implementation Method:
  Uses c2patool (v0.24.0+) with built-in ES256 test certificates for cryptographic signing.
  The built-in test certificates produce spec-compliant C2PA manifests with authentic
  cryptographic signatures, suitable for robustness research without requiring CA infrastructure.

Usage:
  python embed_c2pa_v2.py --images-dir data/raw_images/ --videos-dir data/raw_videos/ --output-dir data/manifests/
"""

import argparse
import json
import logging
import subprocess
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

# Path to c2patool executable
C2PATOOL_PATH = Path("tools/c2patool/c2patool/c2patool.exe")


def log_environment():
    """Log Python and c2patool environment information."""
    logger.info(f"Python version: {sys.version}")

    # Get c2patool version
    try:
        result = subprocess.run(
            [str(C2PATOOL_PATH), "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"c2patool version: {result.stdout.strip()}")
    except Exception as e:
        logger.error(f"Failed to get c2patool version: {e}")


def create_manifest_json(asset_path: Path, asset_type: str,
                         seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create C2PA manifest JSON structure.

    Args:
        asset_path: Path to the asset file
        asset_type: Type of asset ('image' or 'video')
        seed: Random seed used to generate the asset (if available)

    Returns:
        Dictionary containing manifest metadata following C2PA v2 spec
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Read asset metadata from accompanying JSON file if it exists
    metadata_path = asset_path.with_suffix('.json')
    asset_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            asset_metadata = json.load(f)

    # Extract generation parameters
    model = asset_metadata.get("model", "unknown")
    actual_seed = seed or asset_metadata.get("seed", 42)
    model_version = asset_metadata.get("model_version", "unknown")

    # Build C2PA manifest (V2 format)
    # NOTE: Do NOT include private_key or sign_cert fields
    # This causes c2patool to use its built-in test certificate
    manifest = {
        "claim_generator": "c2pa-robustness-research/0.1.0",
        "claim_generator_info": [
            {
                "name": "C2PA Robustness Research Pipeline",
                "version": "0.1.0"
            }
        ],
        "title": f"AI-generated {asset_type}: {asset_path.name}",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "when": timestamp,
                            "softwareAgent": {
                                "name": model,
                                "version": model_version
                            },
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                            "parameters": {
                                "seed": actual_seed,
                                "model_version": model_version,
                                "generation_timestamp": timestamp
                            }
                        }
                    ]
                }
            },
            {
                "label": "cawg.training-mining",
                "data": {
                    "entries": {
                        "cawg.ai_inference": {"use": "notAllowed"},
                        "cawg.ai_generative_training": {"use": "notAllowed"}
                    }
                }
            }
        ]
    }

    return manifest


def sign_asset_with_c2patool(asset_path: Path, output_path: Path,
                              manifest_json: Dict[str, Any]) -> bool:
    """
    Sign an asset with C2PA manifest using c2patool CLI.

    Uses c2patool's built-in test certificate for signing by not specifying
    private_key or sign_cert in the manifest JSON.

    Args:
        asset_path: Input asset file path
        output_path: Output signed asset file path
        manifest_json: Manifest metadata dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create temporary manifest file
        manifest_path = output_path.parent / f"{output_path.stem}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_json, f, indent=2)

        logger.info(f"Signing asset: {asset_path.name}")

        # Call c2patool to sign the asset
        cmd = [
            str(C2PATOOL_PATH),
            str(asset_path),
            "-m", str(manifest_path),
            "-o", str(output_path),
            "-f"  # Force overwrite
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit, we'll check manually
        )

        # Check if signing was successful
        if result.returncode != 0:
            logger.error(f"c2patool failed with exit code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return False

        # Parse the JSON output to verify signing
        try:
            output_json = json.loads(result.stdout)
            if "active_manifest" in output_json:
                logger.info(f"✅ Successfully signed: {output_path.name}")

                # Log validation status
                validation_status = output_json.get("validation_status", [])
                if validation_status:
                    for status in validation_status:
                        if "untrusted" in status.get("code", ""):
                            logger.info(f"   ℹ️  Expected: {status['explanation']} (test certificate)")
                        else:
                            logger.warning(f"   ⚠️  {status.get('explanation', 'Unknown validation issue')}")

                # Clean up temporary manifest file
                manifest_path.unlink()
                return True
            else:
                logger.error(f"No active manifest found in c2patool output")
                return False

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse c2patool JSON output: {e}")
            logger.error(f"stdout: {result.stdout}")
            return False

    except Exception as e:
        logger.error(f"Failed to sign {asset_path.name}: {e}")
        logger.exception("Detailed error:")
        return False


def verify_signed_asset(asset_path: Path) -> Dict[str, Any]:
    """
    Verify C2PA manifest in a signed asset using c2patool.

    For research purposes, we distinguish between:
    - Integrity validation (cryptographic signatures, hashes) - CRITICAL for research
    - Trust validation (PKI certificate chains) - NOT relevant for robustness testing

    Args:
        asset_path: Path to signed asset

    Returns:
        Dictionary containing detailed verification results
    """
    try:
        # Call c2patool to read the manifest
        cmd = [str(C2PATOOL_PATH), str(asset_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"c2patool verification failed: {result.stderr}")
            return {
                "manifest_present": False,
                "integrity_verified": False,
                "trust_verified": False,
                "signature_valid": False,
                "hash_match": False,
                "assertion_uris_match": False,
                "details": "c2patool failed to read manifest"
            }

        # Parse c2patool JSON output
        manifest_data = json.loads(result.stdout)

        if "active_manifest" not in manifest_data:
            logger.warning(f"No active manifest found in {asset_path.name}")
            return {
                "manifest_present": False,
                "integrity_verified": False,
                "trust_verified": False,
                "signature_valid": False,
                "hash_match": False,
                "assertion_uris_match": False,
                "details": "No active manifest"
            }

        # Analyze validation results
        validation_results = manifest_data.get("validation_results", {}).get("activeManifest", {})
        success_checks = validation_results.get("success", [])
        failure_checks = validation_results.get("failure", [])

        # Check for critical integrity validations
        signature_valid = any("claimSignature.validated" in check.get("code", "")
                            for check in success_checks)
        hash_match = any("assertion.dataHash.match" in check.get("code", "")
                        for check in success_checks)
        assertion_uris_match = any("assertion.hashedURI.match" in check.get("code", "")
                                   for check in success_checks)

        # Separate trust issues from integrity issues
        trust_issues = [check for check in failure_checks
                       if "trust" in check.get("code", "").lower() or
                          "untrusted" in check.get("code", "").lower()]
        integrity_issues = [check for check in failure_checks
                          if check not in trust_issues]

        # For research: only integrity matters
        integrity_verified = (signature_valid and hash_match and
                            assertion_uris_match and len(integrity_issues) == 0)
        trust_verified = len(trust_issues) == 0

        result_dict = {
            "manifest_present": True,
            "integrity_verified": integrity_verified,
            "trust_verified": trust_verified,
            "signature_valid": signature_valid,
            "hash_match": hash_match,
            "assertion_uris_match": assertion_uris_match,
            "trust_issues": trust_issues,
            "integrity_issues": integrity_issues,
            "validation_state": manifest_data.get("validation_state", "Unknown"),
            "claim_generator": manifest_data.get("manifests", {}).get(
                manifest_data.get("active_manifest"), {}
            ).get("claim_generator", "N/A")
        }

        # Log appropriately based on research focus
        if trust_issues and not integrity_issues:
            logger.info(f"📝 Manifest integrity verified: {asset_path.name}")
            logger.debug(f"   Trust issues (expected with test cert): {len(trust_issues)}")
        elif integrity_verified:
            logger.info(f"✅ Manifest fully verified: {asset_path.name}")
        else:
            logger.warning(f"⚠️ Manifest integrity issues found: {asset_path.name}")
            logger.warning(f"   Issues: {integrity_issues}")

        logger.info(f"   Claim Generator: {result_dict['claim_generator']}")
        logger.info(f"   Integrity: {'PASS' if integrity_verified else 'FAIL'}")
        logger.info(f"   Signature Valid: {'YES' if signature_valid else 'NO'}")
        logger.info(f"   Hash Match: {'YES' if hash_match else 'NO'}")
        logger.info(f"   Trust: {'PASS' if trust_verified else 'SKIP (test cert)'}")

        return result_dict

    except Exception as e:
        logger.error(f"Failed to verify {asset_path.name}: {e}")
        return {
            "manifest_present": False,
            "integrity_verified": False,
            "trust_verified": False,
            "signature_valid": False,
            "hash_match": False,
            "assertion_uris_match": False,
            "error": str(e)
        }


def process_assets(images_dir: Path, videos_dir: Path, output_dir: Path,
                  verify: bool = True):
    """
    Process all images and videos, creating and embedding C2PA manifests using c2patool.

    Args:
        images_dir: Directory containing raw images
        videos_dir: Directory containing raw videos
        output_dir: Directory to save signed assets
        verify: Whether to verify signatures after signing
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    total_failed = 0
    total_verified = 0

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
            manifest = create_manifest_json(img_path, "image", seed)

            # Output path for signed image
            signed_path = output_dir / f"{img_path.stem}_signed{img_path.suffix}"

            # Sign the image using c2patool
            success = sign_asset_with_c2patool(img_path, signed_path, manifest)

            if success:
                total_processed += 1

                # Verify the signature
                if verify:
                    verification_result = verify_signed_asset(signed_path)
                    # Count as verified if integrity passes (trust is optional for research)
                    if verification_result.get("integrity_verified", False):
                        total_verified += 1
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
            manifest = create_manifest_json(vid_path, "video", seed)

            # Output path for signed video
            signed_path = output_dir / f"{vid_path.stem}_signed{vid_path.suffix}"

            # Sign the video using c2patool
            success = sign_asset_with_c2patool(vid_path, signed_path, manifest)

            if success:
                total_processed += 1

                # Verify the signature
                if verify:
                    verification_result = verify_signed_asset(signed_path)
                    # Count as verified if integrity passes (trust is optional for research)
                    if verification_result.get("integrity_verified", False):
                        total_verified += 1
            else:
                total_failed += 1

    logger.info("=" * 60)
    logger.info(f"C2PA Signing Complete (using c2patool)")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Verified (integrity): {total_verified}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Embed C2PA content credentials using c2patool CLI",
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
        help="Output directory for signed assets"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after signing"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    # Check if c2patool exists
    if not C2PATOOL_PATH.exists():
        logger.error(f"c2patool not found at: {C2PATOOL_PATH}")
        logger.error("Please download c2patool from: https://github.com/contentauth/c2pa-rs/releases")
        sys.exit(1)

    logger.info(f"Using c2patool at: {C2PATOOL_PATH}")
    logger.info("Using c2patool built-in test certificate for signing")

    # Process assets
    process_assets(
        images_dir=args.images_dir,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
