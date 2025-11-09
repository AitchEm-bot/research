#!/usr/bin/env python3
"""
Generate high-resolution images for Stable Video Diffusion input.

This script generates 1024×576 images using Stable Diffusion specifically for use
as conditioning images in Stable Video Diffusion (SVD) image-to-video generation.

The images are generated at SVD's native resolution (1024×576 landscape) to avoid
quality degradation from upscaling. These images are separate from the main image
dataset and serve solely as SVD inputs.

Citation:
  Stable Diffusion - Rombach et al., "High-Resolution Image Synthesis with
  Latent Diffusion Models," CVPR 2022 (peer-reviewed).
  https://arxiv.org/abs/2112.10752

Usage:
  python generate_video_images.py --seed 100 --count 2 --output-dir data/raw_images_for_video/
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


def generate_video_images(output_dir: Path, count: int = 2, seed: int = 100):
    """
    Generate high-resolution images for Stable Video Diffusion input.

    Args:
        output_dir: Directory to save generated images
        count: Number of images to generate (default: 2)
        seed: Random seed for reproducibility (default: 100)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Import diffusers
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        logger.error("diffusers library not found. Install with: pip install diffusers")
        logger.error("Also required: pip install transformers accelerate")
        sys.exit(1)

    # Load Stable Diffusion model
    model_id = "CompVis/stable-diffusion-v1-4"
    logger.info(f"Loading Stable Diffusion model: {model_id}")

    try:
        # Determine if we should use memory optimizations (for GPUs with ≤8GB VRAM)
        use_memory_optimizations = False
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            use_memory_optimizations = vram_gb <= 8.5
            if use_memory_optimizations:
                logger.info(f"Enabling memory optimizations for {vram_gb:.2f}GB VRAM GPU")

        # Load pipeline with fp16 precision for memory efficiency
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,  # Disable NSFW filter for research
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

            # Memory optimizations for 8GB VRAM GPUs
            if use_memory_optimizations:
                pipe.enable_attention_slicing()
                logger.info("Enabled attention slicing for memory efficiency")

                # Try to enable xformers if available (optional acceleration)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory-efficient attention")
                except Exception:
                    logger.info("xformers not available, skipping (not required)")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Ensure you have sufficient disk space and network connectivity")
        sys.exit(1)

    # Prompts for video conditioning images
    # These prompts are designed to create visually interesting scenes with motion potential
    prompts = [
        "A serene ocean wave crashing on a sandy beach at sunset, photorealistic, 4k",
        "Gentle rain falling on a tranquil forest pond with ripples, cinematic lighting",
        "Flowing waterfall in a lush tropical jungle, mist rising, high detail",
        "Clouds drifting across a bright blue sky over rolling hills, natural lighting",
        "Smoke swirling from incense in a peaceful zen garden, soft focus",
        "Wind blowing through golden wheat fields at golden hour, pastoral scene",
        "Steam rising from a hot cup of coffee on a wooden table, warm tones",
        "Autumn leaves gently falling in a park with soft sunlight filtering through trees",
    ]

    logger.info(f"Generating {count} images at 1024×576 for SVD input")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(count):
        current_seed = seed + i
        prompt = prompts[i % len(prompts)]

        logger.info(f"\n[{i+1}/{count}] Generating image with seed {current_seed}")
        logger.info(f"  Prompt: {prompt}")

        # Set per-image seed
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(current_seed)

        # Generate image at 1024×576 (SVD native landscape resolution)
        # Note: We generate at native resolution to avoid upscaling artifacts
        image = pipe(
            prompt,
            num_inference_steps=50,  # Default quality
            guidance_scale=7.5,
            generator=generator,
            height=576,  # SVD native height (landscape)
            width=1024,  # SVD native width (landscape)
        ).images[0]

        # Save image
        filename_base = f"vidimg_{i:03d}_seed{current_seed}_{timestamp}"
        image_path = output_dir / f"{filename_base}.png"
        image.save(image_path, format="PNG")

        logger.info(f"  Saved: {image_path.name}")

        # Save metadata
        metadata = {
            "filename": image_path.name,
            "prompt": prompt,
            "seed": current_seed,
            "model": model_id,
            "resolution": "1024x576",
            "inference_steps": 50,
            "guidance_scale": 7.5,
            "purpose": "SVD_conditioning_image",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        metadata_path = output_dir / f"{filename_base}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"\nSuccessfully generated {count} images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-resolution images for Stable Video Diffusion input"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for reproducibility (default: 100)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=2,
        help="Number of images to generate (default: 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_images_for_video",
        help="Output directory for images (default: data/raw_images_for_video)"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    # Generate images
    output_dir = Path(args.output_dir)
    generate_video_images(output_dir, args.count, args.seed)


if __name__ == "__main__":
    main()
