"""
External Video Preparation Script for C2PA Robustness Testing
=============================================================

This script processes externally-generated videos (Sora 2, Runway, etc.) from the
raw_out_videos/ folder, signs them with C2PA manifests if needed, and moves them
to the organized manifest structure for transformation testing.

Workflow:
1. Scan data/raw_out_videos/ for video files (.mp4, .mov, .avi)
2. Check each video's C2PA signature status using c2patool
3. Sign unsigned videos with C2PA manifests using built-in test certificates
4. Move signed videos to data/manifests/videos/external/
5. Generate detailed processing log

Research Context:
- External videos (Sora 2, Runway, etc.) may already have C2PA manifests
- If already signed, we preserve the original manifest and move as-is
- If unsigned, we sign with our test certificate for consistency with internal pipeline
- This enables comparative testing between different generative AI platforms

Usage:
    python scripts/external/prepare_external_videos.py [--test]

    --test: Process only first video (smoke test)

Requirements:
- c2patool must be installed and available in PATH
- Videos should be placed in data/raw_out_videos/
- See data/raw_out_videos/README.md for detailed instructions
"""

import logging
import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/raw_out_videos/preparation.log')
    ]
)

# Configuration
RAW_VIDEOS_DIR = Path("data/raw_out_videos")
OUTPUT_DIR = Path("data/manifests/videos/external")
SUPPORTED_EXTENSIONS = ['.mp4', '.mov', '.avi']

# C2PA configuration (reused from embed_c2pa_v2.py)
C2PATOOL_PATH = Path("tools/c2patool/c2patool/c2patool.exe")
C2PATOOL_VERSION_REQUIRED = "0.24.0"

# Manifest template for external videos
MANIFEST_TEMPLATE = {
    "claim_generator": "c2patool/{version}",
    "assertions": [
        {
            "label": "c2pa.actions",
            "data": {
                "actions": [
                    {
                        "action": "c2pa.created",
                        "softwareAgent": "External AI Video Generator",
                        "when": "{timestamp}"
                    }
                ]
            }
        }
    ]
}


def check_c2patool() -> Tuple[bool, str]:
    """
    Check if c2patool is installed and meets version requirements.

    Returns:
        Tuple of (is_valid, version_string)
    """
    try:
        result = subprocess.run(
            [str(C2PATOOL_PATH), '--version'],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )

        version_line = result.stdout.strip()
        logging.info(f"Found: {version_line}")

        # Extract version number (e.g., "c2patool 0.9.13" -> "0.9.13")
        version_parts = version_line.split()
        if len(version_parts) >= 2:
            version = version_parts[1]
            return True, version

        return False, "Unknown version"

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logging.error(f"c2patool check failed: {e}")
        return False, "Not installed"


def check_video_signature(video_path: Path) -> Tuple[bool, Optional[Dict]]:
    """
    Check if video already has a C2PA manifest.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (is_signed, manifest_info)
    """
    try:
        cmd = [
            str(C2PATOOL_PATH),
            str(video_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        # c2patool returns 0 and outputs JSON for signed assets
        if result.returncode == 0 and result.stdout.strip():
            try:
                manifest_data = json.loads(result.stdout)
                # Check if active_manifest exists
                if 'active_manifest' in manifest_data:
                    logging.debug(f"{video_path.name}: Already signed")
                    return True, manifest_data
                else:
                    logging.debug(f"{video_path.name}: No active manifest")
                    return False, None
            except json.JSONDecodeError:
                logging.warning(f"{video_path.name}: Signature check returned invalid JSON")
                return False, None
        else:
            logging.debug(f"{video_path.name}: No signature found")
            return False, None

    except subprocess.TimeoutExpired:
        logging.error(f"Timeout checking signature for {video_path.name}")
        return False, None
    except Exception as e:
        logging.error(f"Failed to check signature for {video_path.name}: {e}")
        return False, None


def sign_video(video_path: Path, output_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Sign video with C2PA manifest using built-in test certificate.

    Args:
        video_path: Path to unsigned video
        output_path: Path for signed video output

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Create manifest JSON with current timestamp
        manifest = MANIFEST_TEMPLATE.copy()
        manifest['assertions'][0]['data']['actions'][0]['when'] = datetime.now().isoformat()

        # Save manifest to temporary file
        manifest_path = video_path.with_suffix('.manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Build c2patool command
        # Uses built-in ES256 test certificate (no --certs or --private-key needed)
        cmd = [
            str(C2PATOOL_PATH),
            str(video_path),
            '--manifest', str(manifest_path),
            '--output', str(output_path),
            '--force'
        ]

        # Run c2patool
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout for large videos
        )

        # Clean up manifest file
        manifest_path.unlink()

        # Verify signing succeeded
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            logging.debug(f"Signed {video_path.name} ({size_mb:.1f} MB)")
            return True, None
        else:
            return False, "Output file not created"

    except subprocess.TimeoutExpired:
        logging.error(f"Timeout signing {video_path.name}")
        if manifest_path.exists():
            manifest_path.unlink()
        return False, "Timeout during signing"

    except subprocess.CalledProcessError as e:
        logging.error(f"c2patool failed for {video_path.name}: {e.stderr}")
        if manifest_path.exists():
            manifest_path.unlink()
        return False, f"c2patool error: {e.stderr[:200]}"

    except Exception as e:
        logging.error(f"Failed to sign {video_path.name}: {e}")
        if manifest_path.exists():
            manifest_path.unlink()
        return False, str(e)


def get_video_info(video_path: Path) -> Dict:
    """
    Get basic video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dict with video properties (duration, codec, dimensions)
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        info = json.loads(result.stdout)

        # Extract video stream
        video_stream = next((s for s in info.get('streams', []) if s['codec_type'] == 'video'), None)

        if video_stream:
            # Parse FPS
            fps_str = video_stream.get('r_frame_rate', '0/1')
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom if denom != 0 else 0

            return {
                'duration': float(info.get('format', {}).get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'width': video_stream.get('width', 0),
                'height': video_stream.get('height', 0),
                'fps': fps,
                'size_mb': int(info.get('format', {}).get('size', 0)) / (1024 * 1024)
            }
    except Exception as e:
        logging.warning(f"Failed to get video info for {video_path.name}: {e}")

    return {}


def save_processing_metadata(video_info: Dict, output_path: Path):
    """
    Save processing metadata as JSON sidecar.

    Args:
        video_info: Video processing information
        output_path: Path to output video
    """
    try:
        metadata = {
            "source_file": video_info['source_file'],
            "source_type": "external",
            "was_already_signed": video_info['was_signed'],
            "processing_timestamp": datetime.now().isoformat(),
            "video_properties": video_info.get('properties', {}),
            "output_file": output_path.name
        }

        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        logging.warning(f"Failed to save metadata for {output_path.name}: {e}")


def process_external_videos(test_mode: bool = False):
    """
    Process all external videos from raw_out_videos folder.

    Args:
        test_mode: If True, process only first video (smoke test)
    """
    # Find all video files
    video_files = []
    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(sorted(RAW_VIDEOS_DIR.glob(f"*{ext}")))

    # Filter out README and log files
    video_files = [v for v in video_files if v.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not video_files:
        logging.error(f"No video files found in {RAW_VIDEOS_DIR}")
        logging.info(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        logging.info(f"Place external videos in {RAW_VIDEOS_DIR.absolute()}")
        return

    if test_mode:
        video_files = video_files[:1]
        logging.info("TEST MODE: Processing only first video")

    logging.info(f"Found {len(video_files)} video(s) to process")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_input': len(video_files),
        'already_signed': 0,
        'newly_signed': 0,
        'failed': 0,
        'skipped': 0
    }

    # Process each video
    for video_path in tqdm(video_files, desc="Processing external videos", unit="video"):
        logging.info(f"Processing: {video_path.name}")

        # Get video info
        video_props = get_video_info(video_path)
        if video_props:
            logging.info(f"  {video_props['width']}x{video_props['height']}, "
                        f"{video_props['fps']:.1f}fps, {video_props['duration']:.1f}s, "
                        f"{video_props['codec']}, {video_props['size_mb']:.1f} MB")

        # Check if already signed
        is_signed, manifest_info = check_video_signature(video_path)

        if is_signed:
            # Already signed - move as-is
            logging.info(f"  Already signed with C2PA manifest - preserving original")
            stats['already_signed'] += 1

            # Generate output filename
            output_path = OUTPUT_DIR / f"{video_path.stem}_signed{video_path.suffix}"

            # Copy to external manifests folder
            try:
                shutil.copy2(video_path, output_path)
                logging.info(f"  Moved to: {output_path}")

                # Save metadata
                video_info = {
                    'source_file': video_path.name,
                    'was_signed': True,
                    'properties': video_props
                }
                save_processing_metadata(video_info, output_path)

            except Exception as e:
                logging.error(f"Failed to copy {video_path.name}: {e}")
                stats['failed'] += 1
                continue

        else:
            # Not signed - sign with our certificate
            logging.info(f"  No C2PA manifest found - signing with test certificate")

            # Generate output filename
            output_path = OUTPUT_DIR / f"{video_path.stem}_signed{video_path.suffix}"

            # Sign video
            success, error = sign_video(video_path, output_path)

            if success:
                stats['newly_signed'] += 1
                logging.info(f"  Signed and moved to: {output_path}")

                # Save metadata
                video_info = {
                    'source_file': video_path.name,
                    'was_signed': False,
                    'properties': video_props
                }
                save_processing_metadata(video_info, output_path)
            else:
                stats['failed'] += 1
                logging.error(f"  Failed to sign: {error}")

    # Print summary
    logging.info("=" * 60)
    logging.info("External Video Preparation Complete")
    logging.info(f"  Input videos: {stats['total_input']}")
    logging.info(f"  Already signed (preserved): {stats['already_signed']}")
    logging.info(f"  Newly signed: {stats['newly_signed']}")
    logging.info(f"  Failed: {stats['failed']}")
    logging.info(f"  Total processed: {stats['already_signed'] + stats['newly_signed']}")
    logging.info(f"  Output directory: {OUTPUT_DIR.absolute()}")
    logging.info("=" * 60)

    if stats['failed'] > 0:
        logging.warning(f"WARNING: {stats['failed']} video(s) failed to process")
    else:
        logging.info("SUCCESS: All videos processed successfully!")

    logging.info("\nNext steps:")
    logging.info("1. Run transformation scripts (compress_videos.py, edit_assets.py)")
    logging.info("2. External videos will be processed alongside internal videos")
    logging.info("3. Check final_metrics.csv for combined results")


def main():
    """Main entry point."""
    # Print environment info
    logging.info("=" * 60)
    logging.info("External Video Preparation - C2PA Robustness Testing")
    logging.info(f"Python version: {sys.version}")

    # Check c2patool
    is_valid, version = check_c2patool()
    if not is_valid:
        logging.error("c2patool not found! Please install c2patool:")
        logging.error("  cargo install c2patool")
        logging.error("  Or download from: https://github.com/contentauth/c2pa-rs")
        sys.exit(1)

    logging.info(f"c2patool version: {version}")

    # Check ffprobe
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, text=True, check=True)
        ffprobe_version = result.stdout.split('\n')[0]
        logging.info(f"ffprobe: {ffprobe_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("ffprobe not found - video info will be limited")

    logging.info("=" * 60)

    # Check for test mode flag
    test_mode = '--test' in sys.argv

    # Process videos
    process_external_videos(test_mode=test_mode)


if __name__ == "__main__":
    main()
