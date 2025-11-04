#!/usr/bin/env python3
"""
Embed C2PA content credentials (manifests) into images and videos.

This script creates C2PA manifests for AI-generated images and videos, embedding
cryptographic signatures and metadata for provenance tracking using the official
c2pa-python library.

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

import c2pa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

# Log environment info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_environment():
    """Log Python and C2PA environment information."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"c2pa SDK version: {c2pa.sdk_version()}")


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

    # Determine format/MIME type
    format_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.mp4': 'video/mp4',
        '.avi': 'video/avi',
        '.mov': 'video/quicktime',
    }
    file_format = format_map.get(asset_path.suffix.lower(), 'application/octet-stream')

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

    # Build C2PA manifest (V2 format as per c2pa-python >= 0.2.0)
    manifest = {
        "claim_generator": "c2pa-robustness-research/0.1.0",
        "claim_generator_info": [
            {
                "name": "C2PA Robustness Research Pipeline",
                "version": "0.1.0"
            }
        ],
        "format": file_format,
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
        ],
        "ingredients": []
    }

    return manifest


def sign_asset(asset_path: Path, output_path: Path, manifest_json: Dict[str, Any],
               private_key_path: Path, cert_path: Path,
               tsa_url: Optional[str] = None) -> bool:
    """
    Sign an asset with C2PA manifest using the c2pa-python library with callback signer.

    Uses callback signer to support self-signed certificates for testing/research.

    Args:
        asset_path: Input asset file path
        output_path: Output signed asset file path
        manifest_json: Manifest metadata dictionary
        private_key_path: Path to private key (PEM format)
        cert_path: Path to certificate chain (PEM format)
        tsa_url: Timestamp authority URL (optional for testing)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load private key and certificate
        with open(private_key_path, "rb") as key_file:
            private_key_bytes = key_file.read()

        with open(cert_path, "rb") as cert_file:
            certs_bytes = cert_file.read()

        # Load private key for callback signing
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
            backend=default_backend()
        )

        # Define callback signer function for ES256
        def callback_signer_es256(data: bytes) -> bytes:
            """Callback function that signs data using ES256 algorithm."""
            signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
            return signature

        # Sign the asset using callback signer
        logger.info(f"Signing asset: {asset_path.name}")

        with c2pa.Signer.from_callback(
            callback=callback_signer_es256,
            alg=c2pa.C2paSigningAlg.ES256,
            certs=certs_bytes.decode('utf-8'),
            tsa_url=tsa_url  # Can be None for testing
        ) as signer:
            with c2pa.Builder(manifest_json) as builder:
                builder.sign_file(
                    source_path=str(asset_path),
                    dest_path=str(output_path),
                    signer=signer
                )

        logger.info(f"✅ Successfully signed: {output_path.name}")
        return True

    except Exception as e:
        logger.error(f"Failed to sign {asset_path.name}: {e}")
        logger.exception("Detailed error:")
        return False


def verify_signed_asset(asset_path: Path) -> Dict[str, Any]:
    """
    Verify C2PA manifest in a signed asset, separating trust from integrity validation.

    For research purposes, we distinguish between:
    - Integrity validation (cryptographic signatures, hashes) - CRITICAL for research
    - Trust validation (PKI certificate chains) - NOT relevant for robustness testing

    Args:
        asset_path: Path to signed asset

    Returns:
        Dictionary containing detailed verification results
    """
    try:
        with c2pa.Reader(str(asset_path)) as reader:
            manifest_store_json = reader.json()
            manifest_store = json.loads(manifest_store_json)

            # Check if active manifest exists
            if "active_manifest" not in manifest_store:
                logger.warning(f"No active manifest found in {asset_path.name}")
                return {
                    "manifest_present": False,
                    "integrity_verified": False,
                    "trust_verified": False,
                    "signature_valid": False,
                    "hash_match": False,
                    "details": "No active manifest"
                }

            active_label = manifest_store["active_manifest"]
            if active_label not in manifest_store.get("manifests", {}):
                logger.warning(f"Active manifest '{active_label}' not found in store")
                return {
                    "manifest_present": False,
                    "integrity_verified": False,
                    "trust_verified": False,
                    "signature_valid": False,
                    "hash_match": False,
                    "details": "Active manifest not in store"
                }

            manifest = manifest_store["manifests"][active_label]

            # Analyze validation status
            validation_status = manifest.get("validation_status", [])

            # Separate trust issues from integrity issues
            trust_issues = []
            integrity_issues = []

            for status_item in validation_status:
                # Get the code or description of the validation issue
                code = status_item.get("code", "")
                description = str(status_item.get("description", ""))

                # Categorize the issue
                if any(keyword in code.lower() + description.lower()
                      for keyword in ["trust", "certificate", "untrusted", "self-signed", "ca", "chain"]):
                    trust_issues.append(status_item)
                else:
                    integrity_issues.append(status_item)

            # Determine verification status
            # For research: only integrity matters, trust is informational
            integrity_verified = len(integrity_issues) == 0
            trust_verified = len(trust_issues) == 0

            # Check specific validation aspects
            signature_valid = not any("signature" in str(issue).lower()
                                     for issue in integrity_issues)
            hash_match = not any("hash" in str(issue).lower()
                                for issue in integrity_issues)

            result = {
                "manifest_present": True,
                "integrity_verified": integrity_verified,  # This is what matters for research
                "trust_verified": trust_verified,  # Track but don't fail on this
                "signature_valid": signature_valid,
                "hash_match": hash_match,
                "trust_issues": trust_issues,
                "integrity_issues": integrity_issues,
                "claim_generator": manifest.get("claim_generator", "N/A"),
                "title": manifest.get("title", "N/A")
            }

            # Log appropriately based on research focus
            if trust_issues and not integrity_issues:
                logger.info(f"📝 Manifest integrity verified (trust validation skipped): {asset_path.name}")
                logger.debug(f"   Trust issues (expected with self-signed): {trust_issues}")
            elif not integrity_issues:
                logger.info(f"✅ Manifest fully verified: {asset_path.name}")
            else:
                logger.warning(f"⚠️ Manifest integrity issues found: {asset_path.name}")
                logger.warning(f"   Issues: {integrity_issues}")

            logger.info(f"   Claim Generator: {result['claim_generator']}")
            logger.info(f"   Title: {result['title']}")
            logger.info(f"   Integrity: {'PASS' if integrity_verified else 'FAIL'}")
            logger.info(f"   Trust: {'PASS' if trust_verified else 'SKIP (self-signed)'}")

            return result

    except Exception as e:
        logger.error(f"Failed to verify {asset_path.name}: {e}")
        return {
            "manifest_present": False,
            "integrity_verified": False,
            "trust_verified": False,
            "signature_valid": False,
            "hash_match": False,
            "error": str(e)
        }


def process_assets(images_dir: Path, videos_dir: Path, output_dir: Path,
                  private_key_path: Path, cert_path: Path,
                  verify: bool = True, tsa_url: Optional[str] = None):
    """
    Process all images and videos, creating and embedding C2PA manifests.

    Args:
        images_dir: Directory containing raw images
        videos_dir: Directory containing raw videos
        output_dir: Directory to save signed assets
        private_key_path: Path to signing private key
        cert_path: Path to signing certificate
        verify: Whether to verify signatures after signing
        tsa_url: Timestamp authority URL
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

            # Sign the image
            success = sign_asset(img_path, signed_path, manifest,
                               private_key_path, cert_path, tsa_url)

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

            # Sign the video
            success = sign_asset(vid_path, signed_path, manifest,
                               private_key_path, cert_path, tsa_url)

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
    logger.info(f"C2PA Signing Complete")
    logger.info(f"  Processed: {total_processed}")
    logger.info(f"  Failed: {total_failed}")
    logger.info(f"  Verified: {total_verified}")
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
        help="Output directory for signed assets"
    )
    parser.add_argument(
        "--private-key",
        type=Path,
        default=Path("data/manifests/test_keys/test_es256_private.key"),
        help="Path to private key (PEM format)"
    )
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path("data/certs/chain.pem"),
        help="Path to certificate chain (PEM format)"
    )
    parser.add_argument(
        "--tsa-url",
        type=str,
        default=None,
        help="Timestamp Authority URL (optional, None for testing with self-signed certs)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after signing"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    # Check if keys exist
    if not args.private_key.exists():
        logger.error(f"Private key not found: {args.private_key}")
        logger.error("Run: python scripts/embedding/generate_test_certs.py")
        sys.exit(1)

    if not args.certificate.exists():
        logger.error(f"Certificate not found: {args.certificate}")
        logger.error("Run: python scripts/embedding/generate_test_certs.py")
        sys.exit(1)

    logger.info(f"Using private key: {args.private_key}")
    logger.info(f"Using certificate: {args.certificate}")
    logger.info(f"TSA URL: {args.tsa_url}")

    # Process assets
    process_assets(
        images_dir=args.images_dir,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        private_key_path=args.private_key,
        cert_path=args.certificate,
        verify=not args.no_verify,
        tsa_url=args.tsa_url
    )


if __name__ == "__main__":
    main()
