#!/usr/bin/env python3
"""
Generate deterministic AI images using Stable Diffusion (diffusers library).

This script generates a specified number of 512×512 images using Stable Diffusion
with deterministic seeding for reproducibility. Outputs are saved to the specified
directory with metadata logged.

Citation:
  Stable Diffusion - Rombach et al., "High-Resolution Image Synthesis with
  Latent Diffusion Models," CVPR 2022 (peer-reviewed).
  https://arxiv.org/abs/2112.10752

Usage:
  python generate_images.py --seed 42 --count 10 --output-dir data/raw_images/
"""

import argparse
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


def set_seed(seed: int):
    """
    Set random seed for reproducibility across numpy, torch, and Python random.

    Args:
        seed: Integer seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: For full determinism, also set PYTHONHASHSEED and torch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


def generate_images(output_dir: Path, count: int = 10, seed: int = 42,
                   resolution: int = 512):
    """
    Generate deterministic images using Stable Diffusion.

    Args:
        output_dir: Directory to save generated images
        count: Number of images to generate
        seed: Random seed for reproducibility
        resolution: Image resolution (default 512×512)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Import diffusers here to provide helpful error message if not installed
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        logger.error("diffusers library not found. Install with: pip install diffusers")
        logger.error("Also required: pip install transformers accelerate")
        sys.exit(1)

    # Load Stable Diffusion model
    # Using CompVis/stable-diffusion-v1-4 as a well-established checkpoint
    model_id = "CompVis/stable-diffusion-v1-4"
    logger.info(f"Loading Stable Diffusion model: {model_id}")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,  # Disable for research use (document in ethics)
        )

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        else:
            logger.warning("CUDA not available, using CPU (will be slow)")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("You may need to accept the model license on HuggingFace Hub")
        logger.error(f"Visit: https://huggingface.co/{model_id}")
        sys.exit(1)

    # Define prompts for diverse test images
    prompts = [
        "a photograph of a mountain landscape at sunset",
        "a close-up photo of a red rose with water droplets",
        "an abstract painting with geometric shapes",
        "a portrait of a robot with glowing eyes",
        "a scenic view of a tropical beach with palm trees",
        "a photograph of a modern city skyline at night",
        "a still life with fruits on a wooden table",
        "a digital artwork of a futuristic spaceship",
        "a photograph of autumn leaves on the ground",
        "a minimalist architectural interior design",
    ]

    # Generate images
    logger.info(f"Generating {count} images at {resolution}×{resolution}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(count):
        prompt_idx = i % len(prompts)
        prompt = prompts[prompt_idx]

        # Generate with fixed seed for this image
        image_seed = seed + i
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(image_seed)

        logger.info(f"[{i+1}/{count}] Generating image with seed {image_seed}")
        logger.info(f"  Prompt: {prompt}")

        # Generate image
        with torch.no_grad():
            image = pipe(
                prompt,
                height=resolution,
                width=resolution,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator
            ).images[0]

        # Save image with metadata in filename
        filename = f"img_{i:03d}_seed{image_seed}_{timestamp}.png"
        output_path = output_dir / filename
        image.save(output_path)
        logger.info(f"  Saved: {output_path}")

        # Save metadata to log
        metadata = {
            "filename": filename,
            "seed": image_seed,
            "prompt": prompt,
            "model": model_id,
            "resolution": f"{resolution}x{resolution}",
            "timestamp": timestamp,
        }

        # Optionally save metadata to JSON (for future use)
        import json
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    logger.info(f"Successfully generated {count} images in {output_dir}")
    logger.info(f"Model version: {model_id}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic AI images using Stable Diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_images"),
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution (width and height)"
    )

    args = parser.parse_args()

    # Log environment
    log_environment()

    # Generate images
    generate_images(
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        resolution=args.resolution
    )


if __name__ == "__main__":
    main()
