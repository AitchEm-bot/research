#!/usr/bin/env python3
"""
Generate deterministic AI videos using a frame-based approach.

This script generates short, low-resolution videos (2-4 seconds) for testing C2PA
robustness. It uses a simple frame generation approach with deterministic seeding.

For production use, consider video diffusion models such as:
  - Stable Video Diffusion (Blattmann et al., preprint) - https://stability.ai/research/stable-video-diffusion
  - Image-to-Video models (various, check peer-review status)

Current implementation: Frame-based synthetic video generation using procedural
patterns to ensure deterministic, reproducible outputs.

Usage:
  python generate_videos.py --seed 42 --count 2 --output-dir data/raw_videos/
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Log environment info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_environment():
    """Log Python and OpenCV environment information."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PIL (Pillow) version: {Image.__version__}")


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Integer seed value
    """
    np.random.seed(seed)
    logger.info(f"Random seed set to: {seed}")


def generate_synthetic_frame(frame_idx: int, total_frames: int, width: int,
                             height: int, seed: int, video_type: str) -> np.ndarray:
    """
    Generate a single synthetic video frame using procedural patterns.

    Args:
        frame_idx: Current frame index
        total_frames: Total number of frames in video
        width: Frame width in pixels
        height: Frame height in pixels
        seed: Random seed for this video
        video_type: Type of pattern ('gradient', 'noise', 'shapes')

    Returns:
        NumPy array of shape (height, width, 3) with RGB values
    """
    # Create base frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    if video_type == 'gradient':
        # Animated gradient
        progress = frame_idx / total_frames
        for y in range(height):
            for x in range(width):
                r = int((x / width) * 255)
                g = int((y / height) * 255)
                b = int(progress * 255)
                frame[y, x] = [r, g, b]

    elif video_type == 'noise':
        # Deterministic noise pattern
        local_seed = seed + frame_idx
        rng = np.random.RandomState(local_seed)
        noise = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        # Blend with smooth gradient for visual interest
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(gradient, (height, 1))
        for c in range(3):
            frame[:, :, c] = (noise[:, :, c] * 0.3 + gradient * 0.7).astype(np.uint8)

    elif video_type == 'shapes':
        # Animated geometric shapes
        progress = frame_idx / total_frames
        # Convert to PIL for drawing
        pil_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_frame)

        # Draw circles that move across the frame
        circle_x = int(width * progress)
        circle_y = height // 2
        radius = 30
        draw.ellipse(
            [circle_x - radius, circle_y - radius,
             circle_x + radius, circle_y + radius],
            fill=(255, 100, 100)
        )

        # Draw rectangle
        rect_y = int(height * progress)
        draw.rectangle(
            [width // 4, rect_y, width * 3 // 4, rect_y + 20],
            fill=(100, 255, 100)
        )

        frame = np.array(pil_frame)

    return frame


def generate_video(output_path: Path, duration: float = 3.0, fps: int = 10,
                  width: int = 256, height: int = 256, seed: int = 42,
                  video_type: str = 'gradient'):
    """
    Generate a single synthetic video file.

    Args:
        output_path: Path to save the video file
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
        seed: Random seed for reproducibility
        video_type: Type of synthetic pattern
    """
    total_frames = int(duration * fps)
    logger.info(f"Generating {duration}s video at {width}×{height}, {fps} fps ({total_frames} frames)")

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error(f"Failed to open video writer for {output_path}")
        return

    # Generate and write frames
    for frame_idx in range(total_frames):
        frame_rgb = generate_synthetic_frame(
            frame_idx, total_frames, width, height, seed, video_type
        )
        # OpenCV uses BGR format
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
            logger.info(f"  Frame {frame_idx + 1}/{total_frames}")

    writer.release()
    logger.info(f"Saved video: {output_path}")


def generate_videos(output_dir: Path, count: int = 2, seed: int = 42,
                   duration: float = 3.0, fps: int = 10,
                   width: int = 256, height: int = 256):
    """
    Generate multiple deterministic test videos.

    Args:
        output_dir: Directory to save generated videos
        count: Number of videos to generate
        seed: Random seed for reproducibility
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Video types to cycle through
    video_types = ['gradient', 'noise', 'shapes']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Log model/package information (placeholders for future diffusion models)
    logger.info("=" * 60)
    logger.info("VIDEO GENERATION MODEL INFO (Placeholder)")
    logger.info("=" * 60)
    logger.info("Current method: Procedural frame-based generation")
    logger.info("Model package: opencv-python (cv2)")
    logger.info("Model version: N/A (procedural generation)")
    logger.info("")
    logger.info("For production, consider:")
    logger.info("  - Stable Video Diffusion (Stability AI, preprint)")
    logger.info("  - AnimateDiff or similar video diffusion models")
    logger.info("  - Image-to-video models (check peer-review status)")
    logger.info("=" * 60)

    for i in range(count):
        video_seed = seed + i
        video_type = video_types[i % len(video_types)]

        logger.info(f"\n[{i+1}/{count}] Generating video with seed {video_seed}, type '{video_type}'")

        filename = f"video_{i:03d}_seed{video_seed}_{timestamp}.mp4"
        output_path = output_dir / filename

        generate_video(
            output_path=output_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            seed=video_seed,
            video_type=video_type
        )

        # Save metadata
        import json
        metadata = {
            "filename": filename,
            "seed": video_seed,
            "video_type": video_type,
            "duration": duration,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_frames": int(duration * fps),
            "model_package": "opencv-python",
            "model_version": "procedural_v1",
            "timestamp": timestamp,
            "note": "Procedural generation; replace with diffusion model for production"
        }

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"\nSuccessfully generated {count} videos in {output_dir}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic test videos for C2PA testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_videos"),
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Video duration in seconds"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Video width in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Video height in pixels"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    # Generate videos
    generate_videos(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()
