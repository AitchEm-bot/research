"""
Video Compression Script for C2PA Robustness Testing
====================================================

This script applies various compression transformations to signed videos
to test C2PA manifest robustness under video re-encoding operations.

Transformations:
- H.264 encoding at bitrates: [5000k, 2000k, 500k]
- H.265 encoding at bitrates: [2000k, 500k]
- FPS adjustments: [5fps, 3fps]

Research Context:
- Video re-encoding will likely INVALIDATE C2PA manifests due to frame data changes
- FPS changes will invalidate frame-specific assertions
- The `-map_metadata 0` flag preserves C2PA boxes during re-encoding
- This tests whether manifest STRUCTURE survives even when validation fails

Usage:
    python scripts/transformations/compress_videos.py [--test]

    --test: Run on single video only (smoke test)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Tuple
import json

from tqdm import tqdm

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common import utils

# Configure logging using shared utility
logger = utils.setup_logging(log_file='data/prepared_assets/transformed/transform_videos.log')
utils.log_environment_info()

# Configuration - using shared constants
INPUT_DIRS = [
    utils.DIRS['signed_videos_internal'],
    utils.DIRS['signed_videos_external']
]
OUTPUT_BASE_DIR = utils.DIRS["compression_videos"]
H264_BITRATES = ['5000k', '2000k', '500k']
H265_BITRATES = ['2000k', '500k']
FPS_VALUES = [5, 3]  # From native ~7fps (SVD) or varies (legacy)

# Generate output directories for each bitrate/fps value
def get_h264_output_dir(bitrate: str) -> Path:
    return OUTPUT_BASE_DIR / "h264" / f"bitrate{bitrate}"

def get_h265_output_dir(bitrate: str) -> Path:
    return OUTPUT_BASE_DIR / "h265" / f"bitrate{bitrate}"

def get_fps_output_dir(fps: int) -> Path:
    return OUTPUT_BASE_DIR / "fps" / f"fps{fps}"


def get_video_info(video_path: Path) -> dict:
    """
    Get video information using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dict with video info (fps, duration, codec, etc.)
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

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        # Extract video stream info
        video_stream = next((s for s in info.get('streams', []) if s['codec_type'] == 'video'), None)

        if video_stream:
            # Parse FPS from r_frame_rate (e.g., "7/1")
            fps_str = video_stream.get('r_frame_rate', '0/1')
            num, denom = map(int, fps_str.split('/'))
            fps = num / denom if denom != 0 else 0

            return {
                'fps': fps,
                'duration': float(info.get('format', {}).get('duration', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'width': video_stream.get('width', 0),
                'height': video_stream.get('height', 0)
            }

    except Exception as e:
        logger.warning(f"Failed to get video info for {video_path.name}: {e}")

    return {}


def compress_video_bitrate(video_path: Path, codec: str, bitrate: str, output_dir: Path) -> Tuple[Path, bool]:
    """
    Re-encode video with specified codec and bitrate.

    Args:
        video_path: Path to source video
        codec: Codec name ('h264' or 'h265')
        bitrate: Target bitrate (e.g., '2000k')
        output_dir: Output directory

    Returns:
        Tuple of (output_path, success_status)
    """
    try:
        # Generate output filename
        stem = video_path.stem.replace('_signed', '')
        codec_name = 'h264' if codec == 'libx264' else 'h265'
        output_path = output_dir / f"{stem}_{codec_name}_bitrate{bitrate}.mp4"

        # Build ffmpeg command
        # CRITICAL: -map_metadata 0 preserves C2PA boxes
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-c:v', codec,
            '-b:v', bitrate,
            '-c:a', 'aac',  # Re-encode audio to AAC
            '-b:a', '128k',
            '-map_metadata', '0',  # CRITICAL: Preserve metadata/C2PA boxes
            '-y',  # Overwrite output
            str(output_path)
        ]

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )

        # Get file size for logging
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.debug(f"Created {output_path.name} ({size_mb:.1f} MB)")

        return output_path, True

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout encoding {video_path.name} with {codec} at {bitrate}")
        return None, False
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed for {video_path.name} ({codec}, {bitrate}): {e.stderr}")
        return None, False
    except Exception as e:
        logger.error(f"Failed to encode {video_path.name} with {codec} at {bitrate}: {e}")
        return None, False


def adjust_fps(video_path: Path, target_fps: int, output_dir: Path) -> Tuple[Path, bool]:
    """
    Re-encode video with adjusted frame rate.

    Args:
        video_path: Path to source video
        target_fps: Target frames per second
        output_dir: Output directory

    Returns:
        Tuple of (output_path, success_status)
    """
    try:
        # Generate output filename
        stem = video_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_fps{target_fps}.mp4"

        # Build ffmpeg command
        # CRITICAL: -map_metadata 0 preserves C2PA boxes
        # -r sets output frame rate
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-r', str(target_fps),  # Output frame rate
            '-c:v', 'libx264',  # Re-encode with H.264
            '-crf', '18',  # High quality (lower CRF = better quality)
            '-c:a', 'aac',
            '-b:a', '128k',
            '-map_metadata', '0',  # CRITICAL: Preserve metadata
            '-y',
            str(output_path)
        ]

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300
        )

        # Get file size for logging
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.debug(f"Created {output_path.name} ({size_mb:.1f} MB)")

        return output_path, True

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout adjusting FPS for {video_path.name} to {target_fps}fps")
        return None, False
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg failed for {video_path.name} (fps {target_fps}): {e.stderr}")
        return None, False
    except Exception as e:
        logger.error(f"Failed to adjust FPS for {video_path.name} to {target_fps}: {e}")
        return None, False




def process_videos(test_mode: bool = False, external_only: bool = False):
    """
    Process all signed videos with compression transformations.

    Args:
        test_mode: If True, process only first video (smoke test)
        external_only: If True with test_mode, process only external videos
    """
    # Find all signed MP4 videos from both internal and external sources
    signed_videos = []
    for input_dir in INPUT_DIRS:
        if input_dir.exists():
            signed_videos.extend(sorted(input_dir.glob("*_signed.mp4")))
            logger.info(f"Found {len(list(input_dir.glob('*_signed.mp4')))} videos in {input_dir}")

    if not signed_videos:
        logger.error(f"No signed videos found in {INPUT_DIRS}")
        return

    if test_mode:
        if external_only:
            # Test mode with external-only: select first external video
            external_videos = [v for v in signed_videos if v.parent.name == 'external']
            if external_videos:
                signed_videos = external_videos[:1]
                logger.info("TEST MODE: Processing first external video")
            else:
                logger.error("No external videos found for --external-only test")
                return
        else:
            # Test mode default: select first video (legacy preference removed)
            signed_videos = signed_videos[:1]
            logger.info("TEST MODE: Processing only first video")

    logger.info(f"Found {len(signed_videos)} signed video(s) to process")
    logger.info(f"H.264 bitrates: {H264_BITRATES}")
    logger.info(f"H.265 bitrates: {H265_BITRATES}")
    logger.info(f"FPS values: {FPS_VALUES}")

    # Create output directories for each bitrate/fps value
    for bitrate in H264_BITRATES:
        get_h264_output_dir(bitrate).mkdir(parents=True, exist_ok=True)
    for bitrate in H265_BITRATES:
        get_h265_output_dir(bitrate).mkdir(parents=True, exist_ok=True)
    for fps in FPS_VALUES:
        get_fps_output_dir(fps).mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'total_input': len(signed_videos),
        'h264_success': 0,
        'h264_failed': 0,
        'h265_success': 0,
        'h265_failed': 0,
        'fps_success': 0,
        'fps_failed': 0
    }

    # Process each video
    for video_path in tqdm(signed_videos, desc="Processing videos", unit="video"):
        logger.info(f"Processing: {video_path.name}")

        # Get video info
        video_info = get_video_info(video_path)
        if video_info:
            logger.info(f"  Original: {video_info['fps']:.1f}fps, {video_info['codec']}, "
                        f"{video_info['width']}x{video_info['height']}")

        # Apply H.264 compression at various bitrates
        for bitrate in H264_BITRATES:
            output_dir = get_h264_output_dir(bitrate)
            output_path, success = compress_video_bitrate(video_path, 'libx264', bitrate, output_dir)
            if success:
                stats['h264_success'] += 1
                utils.save_transformation_metadata(output_path, video_path, 'h264_compression',
                            {'codec': 'libx264', 'bitrate': bitrate})
            else:
                stats['h264_failed'] += 1

        # Apply H.265 compression at various bitrates
        for bitrate in H265_BITRATES:
            output_dir = get_h265_output_dir(bitrate)
            output_path, success = compress_video_bitrate(video_path, 'libx265', bitrate, output_dir)
            if success:
                stats['h265_success'] += 1
                utils.save_transformation_metadata(output_path, video_path, 'h265_compression',
                            {'codec': 'libx265', 'bitrate': bitrate})
            else:
                stats['h265_failed'] += 1

        # Apply FPS adjustments
        for fps in FPS_VALUES:
            output_dir = get_fps_output_dir(fps)
            output_path, success = adjust_fps(video_path, fps, output_dir)
            if success:
                stats['fps_success'] += 1
                utils.save_transformation_metadata(output_path, video_path, 'fps_adjustment', {'target_fps': fps})
            else:
                stats['fps_failed'] += 1

    # Print summary
    logger.info("=" * 60)
    logger.info("Video Compression Complete")
    logger.info(f"  Input videos: {stats['total_input']}")
    logger.info(f"  H.264 compressions: {stats['h264_success']} succeeded, {stats['h264_failed']} failed")
    logger.info(f"  H.265 compressions: {stats['h265_success']} succeeded, {stats['h265_failed']} failed")
    logger.info(f"  FPS adjustments: {stats['fps_success']} succeeded, {stats['fps_failed']} failed")
    logger.info(f"  Total output files: {stats['h264_success'] + stats['h265_success'] + stats['fps_success']}")
    logger.info(f"  Output base directory: {OUTPUT_BASE_DIR.absolute()}")
    logger.info(f"    H.264 bitrate folders: bitrate5000k/, bitrate2000k/, bitrate500k/")
    logger.info(f"    H.265 bitrate folders: bitrate2000k/, bitrate500k/")
    logger.info(f"    FPS folders: fps5/, fps3/")
    logger.info("=" * 60)

    # Expected output calculation
    expected_output = len(signed_videos) * (len(H264_BITRATES) + len(H265_BITRATES) + len(FPS_VALUES))
    actual_output = stats['h264_success'] + stats['h265_success'] + stats['fps_success']

    if actual_output == expected_output:
        logger.info("SUCCESS: All transformations completed successfully")
    else:
        logger.warning(f"WARNING: Expected {expected_output} files, got {actual_output}")


def main():
    """Main entry point."""
    # Print environment info
    logger.info("=" * 60)
    logger.info("Video Compression Script - C2PA Robustness Testing")
    logger.info(f"Python version: {sys.version}")

    # Check for ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        ffmpeg_version = result.stdout.split('\n')[0]
        logger.info(f"ffmpeg: {ffmpeg_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found! Please install ffmpeg and ensure it's in PATH")
        sys.exit(1)

    logger.info("=" * 60)

    # Check for test mode flag
    test_mode = '--test' in sys.argv
    external_only = '--external-only' in sys.argv

    # Process videos
    process_videos(test_mode=test_mode, external_only=external_only)


if __name__ == "__main__":
    main()
