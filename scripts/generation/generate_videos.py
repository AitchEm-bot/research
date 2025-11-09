#!/usr/bin/env python3
"""
Generate AI videos using Stable Video Diffusion (SVD).

This script generates short, high-quality videos using Stable Video Diffusion,
a latent video diffusion model that converts static images into realistic video sequences.

IMPORTANT: This uses a PREPRINT (not peer-reviewed) model. Annotate in thesis/reports.

Citation:
  Stable Video Diffusion - Blattmann et al., "Stable Video Diffusion: Scaling
  Latent Video Diffusion Models to Large Datasets," arXiv 2311.15127, November 2023
  (PREPRINT - not yet peer-reviewed)
  https://arxiv.org/abs/2311.15127

Architecture:
  - Image-to-video diffusion model
  - Native resolution: 1024×576 (landscape)
  - Output: 25 frames downscaled to 512×512 for this project
  - Requires pre-generated conditioning images (1024×576)

Usage:
  python generate_videos.py --seed 100 --count 2 --output-dir data/raw_videos/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Log environment info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_environment():
    """Log Python, torch, and CUDA environment information."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU VRAM: {vram_gb:.2f} GB")


def set_seed(seed: int):
    """
    Set random seed for reproducibility across numpy, torch, and Python random.

    Args:
        seed: Integer seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


def load_conditioning_images(images_dir: Path, count: int) -> list:
    """
    Load pre-generated conditioning images for SVD.

    Args:
        images_dir: Directory containing 1024×576 images
        count: Number of images to load

    Returns:
        List of PIL Image objects
    """
    # Find all PNG images in the directory
    image_files = sorted(images_dir.glob("vidimg_*.png"))

    if len(image_files) < count:
        logger.error(f"Found only {len(image_files)} images, but need {count}")
        logger.error(f"Run generate_video_images.py first to create conditioning images")
        sys.exit(1)

    images = []
    for i in range(count):
        img_path = image_files[i]
        logger.info(f"Loading conditioning image: {img_path.name}")
        img = Image.open(img_path).convert("RGB")

        # Verify resolution
        if img.size != (1024, 576):
            logger.warning(f"Image {img_path.name} has size {img.size}, expected (1024, 576)")
            logger.info(f"Resizing to 1024×576")
            img = img.resize((1024, 576), Image.Resampling.LANCZOS)

        images.append(img)

    return images


def generate_videos(output_dir: Path, images_dir: Path, count: int = 2,
                   seed: int = 100, target_resolution: int = 512):
    """
    Generate videos using Stable Video Diffusion.

    Args:
        output_dir: Directory to save generated videos
        images_dir: Directory containing conditioning images (1024×576)
        count: Number of videos to generate
        seed: Random seed for reproducibility
        target_resolution: Target output resolution (default: 512×512)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Import required libraries
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        logger.error("diffusers library not found or outdated")
        logger.error("Install with: pip install diffusers>=0.25.0 imageio-ffmpeg")
        sys.exit(1)

    # Load conditioning images
    logger.info(f"Loading {count} conditioning images from {images_dir}")
    conditioning_images = load_conditioning_images(images_dir, count)

    # Load Stable Video Diffusion pipeline
    model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
    logger.info("=" * 70)
    logger.info("VIDEO GENERATION MODEL INFO")
    logger.info("=" * 70)
    logger.info(f"Model: {model_id}")
    logger.info("Citation: Blattmann et al., arXiv 2311.15127, Nov 2023")
    logger.info("Status: PREPRINT (not yet peer-reviewed) - annotate in thesis")
    logger.info("Native resolution: 1024×576 (25 frames)")
    logger.info(f"Output resolution: {target_resolution}×{target_resolution} (downscaled)")
    logger.info("=" * 70)
    logger.info("")

    logger.info(f"Loading Stable Video Diffusion model: {model_id}")

    try:
        # Determine if we should use aggressive memory optimizations
        use_cpu_offload = False
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            use_cpu_offload = vram_gb <= 8.5
            if use_cpu_offload:
                logger.info(f"Enabling CPU offloading for {vram_gb:.2f}GB VRAM GPU")
                logger.info("Note: This will slow generation but prevent OOM errors")

        # Load pipeline with fp16 precision
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        )

        # Memory optimizations for RTX 4060 8GB
        if use_cpu_offload:
            # CPU offloading: moves model components between CPU/GPU as needed
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model CPU offloading (reduces VRAM to <8GB)")
        else:
            # If VRAM > 8GB, keep everything on GPU for speed
            pipe = pipe.to("cuda")

        # Additional memory optimizations (optional)
        try:
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing for memory efficiency")
        except Exception:
            logger.info("VAE slicing not available (skipping)")

    except Exception as e:
        logger.error(f"Failed to load SVD model: {e}")
        logger.error("Ensure you have sufficient disk space (~14GB for model download)")
        logger.error("and network connectivity to HuggingFace")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(count):
        current_seed = seed + i
        conditioning_image = conditioning_images[i]

        logger.info(f"\n[{i+1}/{count}] Generating video with seed {current_seed}")
        logger.info(f"  Conditioning image: {conditioning_image.size}")
        logger.info("  This may take 5-10 minutes on RTX 4060 8GB...")

        # Set per-video seed
        generator = torch.Generator(device="cuda" if not use_cpu_offload else "cpu")
        generator.manual_seed(current_seed)

        # Generate video frames using SVD
        # decode_chunk_size controls VAE memory usage (lower = less VRAM, slower)
        try:
            frames = pipe(
                conditioning_image,
                decode_chunk_size=8,  # Process 8 frames at a time (memory optimization)
                generator=generator,
                num_frames=25,  # SVD-XT generates 25 frames
                fps=7  # Frame rate for export
            ).frames[0]
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            logger.error("Try reducing decode_chunk_size or ensure GPU has enough memory")
            continue

        logger.info(f"  Generated {len(frames)} frames at 1024×576")

        # Downscale frames to target resolution (512×512)
        if target_resolution != 1024:
            logger.info(f"  Downscaling to {target_resolution}×{target_resolution}")
            resized_frames = []
            for frame in frames:
                # Convert to PIL for high-quality resize
                resized_frame = frame.resize(
                    (target_resolution, target_resolution),
                    Image.Resampling.LANCZOS
                )
                resized_frames.append(resized_frame)
            frames = resized_frames

        # Export to MP4
        filename_base = f"video_{i:03d}_seed{current_seed}_{timestamp}"
        video_path = output_dir / f"{filename_base}.mp4"

        export_to_video(frames, str(video_path), fps=7)
        logger.info(f"  Saved: {video_path.name}")

        # Save metadata
        metadata = {
            "filename": video_path.name,
            "seed": current_seed,
            "model": model_id,
            "model_status": "PREPRINT (arXiv 2311.15127, not peer-reviewed)",
            "native_resolution": "1024x576",
            "output_resolution": f"{target_resolution}x{target_resolution}",
            "num_frames": len(frames),
            "fps": 7,
            "duration_seconds": len(frames) / 7,
            "conditioning_image": conditioning_images[i].filename if hasattr(conditioning_images[i], 'filename') else "N/A",
            "generation_method": "stable_video_diffusion_img2vid",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        metadata_path = output_dir / f"{filename_base}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"\nSuccessfully generated {count} videos in {output_dir}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate videos using Stable Video Diffusion (PREPRINT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_videos"),
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/raw_images_for_video"),
        help="Directory containing conditioning images (1024×576)"
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
        default=100,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Target output resolution (square)"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    # Verify conditioning images directory exists
    if not args.images_dir.exists():
        logger.error(f"Conditioning images directory not found: {args.images_dir}")
        logger.error("Run generate_video_images.py first to create conditioning images")
        sys.exit(1)

    # Generate videos
    generate_videos(
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        count=args.count,
        seed=args.seed,
        target_resolution=args.resolution
    )


if __name__ == "__main__":
    main()
