"""
Asset Editing Script for C2PA Robustness Testing
================================================

This script applies various editing transformations to signed assets
(both images and videos) to test C2PA manifest robustness.

Transformations:
- Crop: Center crop to [90%, 75%, 50%] of original size
- Resize: Multiple resolutions with different interpolation methods
- Rotate: [90°, 180°, 270°]
- Color adjustments: Brightness/contrast/saturation [+20%, -20%]

Research Context:
- All editing operations modify pixel data and will likely INVALIDATE C2PA manifests
- This tests whether manifest STRUCTURE survives even when content validation fails
- For videos, we use ffmpeg with `-map_metadata 0` to preserve C2PA boxes

Usage:
    python scripts/transformations/edit_assets.py [--test] [--images-only] [--videos-only]

    --test: Run on single asset only (smoke test)
    --images-only: Process only images
    --videos-only: Process only videos
"""

import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Tuple, Literal
import json

from PIL import Image, ImageEnhance
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/transformed/transform_edits.log')
    ]
)

# Configuration
INPUT_DIR = Path("data/manifests")
OUTPUT_BASE_DIR = Path("data/transformed/editing")

CROP_PERCENTAGES = [90, 75, 50]  # Percentage of original size to keep
ROTATE_ANGLES = [90, 180, 270]
COLOR_ADJUSTMENTS = [0.8, 1.2]  # 0.8 = -20%, 1.2 = +20%

# Image-specific
IMAGE_RESIZE_TARGETS = [(256, 256), (1024, 1024)]
IMAGE_INTERPOLATIONS = ['bicubic', 'lanczos']

# Video-specific
VIDEO_RESIZE_TARGETS = [(256, 256), (768, 768)]

# Helper functions for ultra-granular directory structure
def get_image_crop_dir(crop_percent: int) -> Path:
    return OUTPUT_BASE_DIR / "images" / "crop" / f"{crop_percent}pct"

def get_image_resize_dir(interpolation: str, size: tuple) -> Path:
    return OUTPUT_BASE_DIR / "images" / "resize" / interpolation / f"{size[0]}x{size[1]}"

def get_image_rotate_dir(angle: int) -> Path:
    return OUTPUT_BASE_DIR / "images" / "rotate" / f"{angle}deg"

def get_image_color_dir(adjustment_type: str, factor: float) -> Path:
    sign = 'plus' if factor > 1.0 else 'minus'
    percent = int(abs(factor - 1.0) * 100)
    return OUTPUT_BASE_DIR / "images" / "color" / adjustment_type / f"{sign}{percent}"

def get_video_crop_dir(crop_percent: int) -> Path:
    return OUTPUT_BASE_DIR / "videos" / "crop" / f"{crop_percent}pct"

def get_video_resize_dir(size: tuple) -> Path:
    return OUTPUT_BASE_DIR / "videos" / "resize" / f"{size[0]}x{size[1]}"

def get_video_rotate_dir(angle: int) -> Path:
    return OUTPUT_BASE_DIR / "videos" / "rotate" / f"{angle}deg"

def get_video_color_dir(adjustment_type: str, factor: float) -> Path:
    sign = 'plus' if factor > 1.0 else 'minus'
    percent = int(abs(factor - 1.0) * 100)
    return OUTPUT_BASE_DIR / "videos" / "color" / adjustment_type / f"{sign}{percent}"


def save_metadata(output_path: Path, source_path: Path, transform_type: str, params: dict):
    """Save transformation metadata as JSON sidecar."""
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


# ===== IMAGE TRANSFORMATIONS =====

def crop_image(image_path: Path, crop_percent: int, output_dir: Path) -> Tuple[Path, bool]:
    """Center crop image to specified percentage of original size."""
    try:
        stem = image_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_crop{crop_percent}.png"

        img = Image.open(image_path)
        width, height = img.size

        # Calculate crop box (center crop)
        new_width = int(width * crop_percent / 100)
        new_height = int(height * crop_percent / 100)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        cropped = img.crop((left, top, right, bottom))
        cropped.save(output_path, 'PNG')

        return output_path, True

    except Exception as e:
        logging.error(f"Failed to crop {image_path.name} to {crop_percent}%: {e}")
        return None, False


def resize_image(image_path: Path, target_size: tuple, interpolation: str, output_dir: Path) -> Tuple[Path, bool]:
    """Resize image to target size with specified interpolation."""
    try:
        stem = image_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_resize{target_size[0]}x{target_size[1]}_{interpolation}.png"

        img = Image.open(image_path)

        # Map interpolation string to Pillow constant
        interp_map = {
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'lanczos': Image.Resampling.LANCZOS
        }
        resample = interp_map.get(interpolation, Image.Resampling.BICUBIC)

        resized = img.resize(target_size, resample)
        resized.save(output_path, 'PNG')

        return output_path, True

    except Exception as e:
        logging.error(f"Failed to resize {image_path.name} to {target_size} ({interpolation}): {e}")
        return None, False


def rotate_image(image_path: Path, angle: int, output_dir: Path) -> Tuple[Path, bool]:
    """Rotate image by specified angle."""
    try:
        stem = image_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_rotate{angle}.png"

        img = Image.open(image_path)

        # For 90/180/270 rotations, use transpose (lossless, no re-sampling)
        if angle == 90:
            rotated = img.transpose(Image.Transpose.ROTATE_270)  # PIL rotates counter-clockwise
        elif angle == 180:
            rotated = img.transpose(Image.Transpose.ROTATE_180)
        elif angle == 270:
            rotated = img.transpose(Image.Transpose.ROTATE_90)
        else:
            rotated = img.rotate(angle, expand=True)

        rotated.save(output_path, 'PNG')

        return output_path, True

    except Exception as e:
        logging.error(f"Failed to rotate {image_path.name} by {angle}°: {e}")
        return None, False


def adjust_color_image(image_path: Path, adjustment_type: Literal['brightness', 'contrast', 'saturation'],
                       factor: float, output_dir: Path) -> Tuple[Path, bool]:
    """Adjust image brightness, contrast, or saturation."""
    try:
        stem = image_path.stem.replace('_signed', '')
        sign = 'plus' if factor > 1.0 else 'minus'
        percent = int(abs(factor - 1.0) * 100)
        output_path = output_dir / f"{stem}_{adjustment_type}_{sign}{percent}.png"

        img = Image.open(image_path)

        if adjustment_type == 'brightness':
            enhancer = ImageEnhance.Brightness(img)
        elif adjustment_type == 'contrast':
            enhancer = ImageEnhance.Contrast(img)
        elif adjustment_type == 'saturation':
            enhancer = ImageEnhance.Color(img)
        else:
            raise ValueError(f"Unknown adjustment type: {adjustment_type}")

        adjusted = enhancer.enhance(factor)
        adjusted.save(output_path, 'PNG')

        return output_path, True

    except Exception as e:
        logging.error(f"Failed to adjust {adjustment_type} for {image_path.name} by factor {factor}: {e}")
        return None, False


# ===== VIDEO TRANSFORMATIONS =====

def crop_video(video_path: Path, crop_percent: int, output_dir: Path) -> Tuple[Path, bool]:
    """Center crop video to specified percentage of original size."""
    try:
        stem = video_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_crop{crop_percent}.mp4"

        # Get video dimensions
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
        width = video_stream['width']
        height = video_stream['height']

        # Calculate crop dimensions
        new_width = int(width * crop_percent / 100)
        new_height = int(height * crop_percent / 100)

        # Ensure even dimensions (required for H.264)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        # Center crop offset
        x_offset = (width - new_width) // 2
        y_offset = (height - new_height) // 2

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'crop={new_width}:{new_height}:{x_offset}:{y_offset}',
            '-c:a', 'copy',  # Copy audio without re-encoding
            '-map_metadata', '0',
            '-y', str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return output_path, True

    except Exception as e:
        logging.error(f"Failed to crop video {video_path.name} to {crop_percent}%: {e}")
        return None, False


def resize_video(video_path: Path, target_size: tuple, output_dir: Path) -> Tuple[Path, bool]:
    """Resize video to target size."""
    try:
        stem = video_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_resize{target_size[0]}x{target_size[1]}.mp4"

        # Ensure even dimensions
        width = target_size[0] - (target_size[0] % 2)
        height = target_size[1] - (target_size[1] % 2)

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'scale={width}:{height}',
            '-c:a', 'copy',
            '-map_metadata', '0',
            '-y', str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return output_path, True

    except Exception as e:
        logging.error(f"Failed to resize video {video_path.name} to {target_size}: {e}")
        return None, False


def rotate_video(video_path: Path, angle: int, output_dir: Path) -> Tuple[Path, bool]:
    """Rotate video by specified angle."""
    try:
        stem = video_path.stem.replace('_signed', '')
        output_path = output_dir / f"{stem}_rotate{angle}.mp4"

        # ffmpeg rotation transposes (0=90CCW, 1=90CW, 2=90CW+vflip=180, etc.)
        if angle == 90:
            transpose = '1'  # 90 degrees clockwise
        elif angle == 180:
            transpose = '2,transpose=2'  # Two 90-degree rotations
        elif angle == 270:
            transpose = '2'  # 90 degrees counter-clockwise
        else:
            raise ValueError(f"Unsupported rotation angle: {angle}")

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f'transpose={transpose}',
            '-c:a', 'copy',
            '-map_metadata', '0',
            '-y', str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return output_path, True

    except Exception as e:
        logging.error(f"Failed to rotate video {video_path.name} by {angle}°: {e}")
        return None, False


def adjust_color_video(video_path: Path, adjustment_type: Literal['brightness', 'contrast', 'saturation'],
                       factor: float, output_dir: Path) -> Tuple[Path, bool]:
    """Adjust video brightness, contrast, or saturation."""
    try:
        stem = video_path.stem.replace('_signed', '')
        sign = 'plus' if factor > 1.0 else 'minus'
        percent = int(abs(factor - 1.0) * 100)
        output_path = output_dir / f"{stem}_{adjustment_type}_{sign}{percent}.mp4"

        # ffmpeg eq filter parameters
        if adjustment_type == 'brightness':
            vf_filter = f'eq=brightness={(factor - 1.0) * 0.5}'  # Scale to ffmpeg range
        elif adjustment_type == 'contrast':
            vf_filter = f'eq=contrast={factor}'
        elif adjustment_type == 'saturation':
            vf_filter = f'eq=saturation={factor}'
        else:
            raise ValueError(f"Unknown adjustment type: {adjustment_type}")

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', vf_filter,
            '-c:a', 'copy',
            '-map_metadata', '0',
            '-y', str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        return output_path, True

    except Exception as e:
        logging.error(f"Failed to adjust {adjustment_type} for video {video_path.name}: {e}")
        return None, False


# ===== MAIN PROCESSING =====

def process_images(test_mode: bool = False):
    """Process all signed images with editing transformations."""
    signed_images = sorted(INPUT_DIR.glob("*_signed.png"))

    if not signed_images:
        logging.info("No signed images found")
        return {'total': 0}

    if test_mode:
        signed_images = signed_images[:1]

    logging.info(f"Processing {len(signed_images)} image(s)")

    # Create all output directories
    for crop_pct in CROP_PERCENTAGES:
        get_image_crop_dir(crop_pct).mkdir(parents=True, exist_ok=True)
    for target_size in IMAGE_RESIZE_TARGETS:
        for interp in IMAGE_INTERPOLATIONS:
            get_image_resize_dir(interp, target_size).mkdir(parents=True, exist_ok=True)
    for angle in ROTATE_ANGLES:
        get_image_rotate_dir(angle).mkdir(parents=True, exist_ok=True)
    for adj_type in ['brightness', 'contrast', 'saturation']:
        for factor in COLOR_ADJUSTMENTS:
            get_image_color_dir(adj_type, factor).mkdir(parents=True, exist_ok=True)

    stats = {'total': 0, 'success': 0, 'failed': 0}

    for img_path in tqdm(signed_images, desc="Processing images", unit="img"):
        # Crop transformations
        for crop_pct in CROP_PERCENTAGES:
            output_dir = get_image_crop_dir(crop_pct)
            output_path, success = crop_image(img_path, crop_pct, output_dir)
            if success:
                stats['success'] += 1
                save_metadata(output_path, img_path, 'crop', {'crop_percent': crop_pct})
            else:
                stats['failed'] += 1

        # Resize transformations
        for target_size in IMAGE_RESIZE_TARGETS:
            for interp in IMAGE_INTERPOLATIONS:
                output_dir = get_image_resize_dir(interp, target_size)
                output_path, success = resize_image(img_path, target_size, interp, output_dir)
                if success:
                    stats['success'] += 1
                    save_metadata(output_path, img_path, 'resize',
                                {'target_size': target_size, 'interpolation': interp})
                else:
                    stats['failed'] += 1

        # Rotation transformations
        for angle in ROTATE_ANGLES:
            output_dir = get_image_rotate_dir(angle)
            output_path, success = rotate_image(img_path, angle, output_dir)
            if success:
                stats['success'] += 1
                save_metadata(output_path, img_path, 'rotate', {'angle': angle})
            else:
                stats['failed'] += 1

        # Color adjustments
        for adj_type in ['brightness', 'contrast', 'saturation']:
            for factor in COLOR_ADJUSTMENTS:
                output_dir = get_image_color_dir(adj_type, factor)
                output_path, success = adjust_color_image(img_path, adj_type, factor, output_dir)
                if success:
                    stats['success'] += 1
                    save_metadata(output_path, img_path, f'{adj_type}_adjustment', {'factor': factor})
                else:
                    stats['failed'] += 1

    stats['total'] = stats['success'] + stats['failed']
    return stats


def process_videos(test_mode: bool = False):
    """Process all signed videos with editing transformations."""
    signed_videos = sorted(INPUT_DIR.glob("*_signed.mp4"))

    if not signed_videos:
        logging.info("No signed videos found")
        return {'total': 0}

    if test_mode:
        # Prefer legacy video for speed
        legacy = [v for v in signed_videos if 'seed4' in v.name]
        signed_videos = (legacy[:1] if legacy else signed_videos[:1])

    logging.info(f"Processing {len(signed_videos)} video(s)")

    # Create all output directories
    for crop_pct in CROP_PERCENTAGES:
        get_video_crop_dir(crop_pct).mkdir(parents=True, exist_ok=True)
    for target_size in VIDEO_RESIZE_TARGETS:
        get_video_resize_dir(target_size).mkdir(parents=True, exist_ok=True)
    for angle in ROTATE_ANGLES:
        get_video_rotate_dir(angle).mkdir(parents=True, exist_ok=True)
    for adj_type in ['brightness', 'contrast', 'saturation']:
        for factor in COLOR_ADJUSTMENTS:
            get_video_color_dir(adj_type, factor).mkdir(parents=True, exist_ok=True)

    stats = {'total': 0, 'success': 0, 'failed': 0}

    for video_path in tqdm(signed_videos, desc="Processing videos", unit="vid"):
        # Crop transformations
        for crop_pct in CROP_PERCENTAGES:
            output_dir = get_video_crop_dir(crop_pct)
            output_path, success = crop_video(video_path, crop_pct, output_dir)
            if success:
                stats['success'] += 1
                save_metadata(output_path, video_path, 'crop', {'crop_percent': crop_pct})
            else:
                stats['failed'] += 1

        # Resize transformations
        for target_size in VIDEO_RESIZE_TARGETS:
            output_dir = get_video_resize_dir(target_size)
            output_path, success = resize_video(video_path, target_size, output_dir)
            if success:
                stats['success'] += 1
                save_metadata(output_path, video_path, 'resize', {'target_size': target_size})
            else:
                stats['failed'] += 1

        # Rotation transformations
        for angle in ROTATE_ANGLES:
            output_dir = get_video_rotate_dir(angle)
            output_path, success = rotate_video(video_path, angle, output_dir)
            if success:
                stats['success'] += 1
                save_metadata(output_path, video_path, 'rotate', {'angle': angle})
            else:
                stats['failed'] += 1

        # Color adjustments
        for adj_type in ['brightness', 'contrast', 'saturation']:
            for factor in COLOR_ADJUSTMENTS:
                output_dir = get_video_color_dir(adj_type, factor)
                output_path, success = adjust_color_video(video_path, adj_type, factor, output_dir)
                if success:
                    stats['success'] += 1
                    save_metadata(output_path, video_path, f'{adj_type}_adjustment', {'factor': factor})
                else:
                    stats['failed'] += 1

    stats['total'] = stats['success'] + stats['failed']
    return stats


def main():
    """Main entry point."""
    logging.info("=" * 60)
    logging.info("Asset Editing Script - C2PA Robustness Testing")
    logging.info(f"Python version: {sys.version}")

    # Check for ffmpeg (needed for videos)
    ffmpeg_available = True
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        ffmpeg_available = False
        logging.warning("ffmpeg not found - video processing will be skipped")

    logging.info("=" * 60)

    # Parse arguments
    test_mode = '--test' in sys.argv
    images_only = '--images-only' in sys.argv
    videos_only = '--videos-only' in sys.argv

    if test_mode:
        logging.info("TEST MODE: Processing only one asset of each type")

    # Process assets (directories will be created within process functions)
    image_stats = {'total': 0}
    video_stats = {'total': 0}

    if not videos_only:
        image_stats = process_images(test_mode=test_mode)

    if not images_only and ffmpeg_available:
        video_stats = process_videos(test_mode=test_mode)

    # Print summary
    logging.info("=" * 60)
    logging.info("Asset Editing Complete")
    logging.info(f"  Images: {image_stats.get('success', 0)} succeeded, {image_stats.get('failed', 0)} failed")
    logging.info(f"  Videos: {video_stats.get('success', 0)} succeeded, {video_stats.get('failed', 0)} failed")
    logging.info(f"  Total output files: {image_stats.get('success', 0) + video_stats.get('success', 0)}")
    logging.info(f"  Output base directory: {OUTPUT_BASE_DIR.absolute()}")
    logging.info(f"    Image folders:")
    logging.info(f"      Crop: 90pct/, 75pct/, 50pct/")
    logging.info(f"      Resize: bicubic/256x256/, bicubic/1024x1024/, lanczos/256x256/, lanczos/1024x1024/")
    logging.info(f"      Rotate: 90deg/, 180deg/, 270deg/")
    logging.info(f"      Color: brightness/plus20/, brightness/minus20/, contrast/plus20/, contrast/minus20/, saturation/plus20/, saturation/minus20/")
    logging.info(f"    Video folders:")
    logging.info(f"      Crop: 90pct/, 75pct/, 50pct/")
    logging.info(f"      Resize: 256x256/, 768x768/")
    logging.info(f"      Rotate: 90deg/, 180deg/, 270deg/")
    logging.info(f"      Color: brightness/plus20/, brightness/minus20/, contrast/plus20/, contrast/minus20/, saturation/plus20/, saturation/minus20/")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
