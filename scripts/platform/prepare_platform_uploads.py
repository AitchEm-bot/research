"""
Platform Upload Preparation Script for C2PA Robustness Testing
==============================================================

This script helps prepare assets for manual upload to social media platforms
as part of Phase 2.5 platform round-trip testing. It provides interactive
selection of assets, platform configuration, and C2PA verification before upload.

Workflow:
1. Interactive asset selection from transformed folders
2. Platform and mode selection
3. C2PA verification to confirm signature before upload
4. Copy to platform-specific uploads folder
5. Generate upload instructions and tracking information

Research Context:
- Tests real-world C2PA manifest persistence through social media platforms
- Manual upload/download required due to API restrictions and app-only features
- Tracks pre-upload signature status for comparison with post-download status

Usage:
    python scripts/platform/prepare_platform_uploads.py

    Interactive prompts will guide through selection process

Supported Platforms:
- Instagram: video, image, story, reel
- Twitter: video, image
- Facebook: video, image
- YouTube Shorts: video
- TikTok: video
- WhatsApp: video, image, status
"""

import logging
import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/platform_tests/preparation.log')
    ]
)

# Configuration
TRANSFORMED_BASE = Path("data/transformed")
PLATFORM_TESTS_BASE = Path("data/platform_tests")

# Platform configurations
PLATFORM_CONFIGS = {
    'instagram': {
        'modes': ['video', 'image', 'story', 'reel'],
        'video_formats': ['.mp4'],
        'image_formats': ['.jpg', '.png'],
        'max_video_duration': 60,  # seconds (reel limit)
        'notes': 'Use mobile app for stories/reels. Feed posts can use web.'
    },
    'twitter': {
        'modes': ['video', 'image'],
        'video_formats': ['.mp4'],
        'image_formats': ['.jpg', '.png'],
        'max_video_duration': 140,  # seconds
        'notes': 'Can upload via web or mobile app.'
    },
    'facebook': {
        'modes': ['video', 'image'],
        'video_formats': ['.mp4'],
        'image_formats': ['.jpg', '.png'],
        'max_video_duration': 240,  # seconds for optimal quality
        'notes': 'Can upload via web or mobile app.'
    },
    'youtube_shorts': {
        'modes': ['video'],
        'video_formats': ['.mp4'],
        'image_formats': [],
        'max_video_duration': 60,  # seconds (Shorts limit)
        'notes': 'Use YouTube mobile app. Vertical format preferred.'
    },
    'tiktok': {
        'modes': ['video'],
        'video_formats': ['.mp4'],
        'image_formats': [],
        'max_video_duration': 180,  # seconds
        'notes': 'Use TikTok mobile app. Vertical format preferred.'
    },
    'whatsapp': {
        'modes': ['video', 'image', 'status'],
        'video_formats': ['.mp4'],
        'image_formats': ['.jpg', '.png'],
        'max_video_duration': 90,  # seconds for status
        'notes': 'Use WhatsApp mobile app. Status = 24hr temporary post.'
    }
}


def check_c2patool() -> bool:
    """Check if c2patool is available."""
    try:
        subprocess.run(['c2patool', '--version'], capture_output=True, check=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def verify_c2pa_signature(file_path: Path) -> Tuple[bool, Dict]:
    """
    Verify C2PA signature on file before upload.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (is_signed, verification_details)
    """
    try:
        cmd = [
            'c2patool',
            str(file_path),
            '--output', 'json'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            try:
                manifest_data = json.loads(result.stdout)

                # Extract validation status
                active_manifest = manifest_data.get('active_manifest')
                if active_manifest:
                    claim_signature = active_manifest.get('claim_signature', {})
                    validated = claim_signature.get('validated', False)

                    return True, {
                        'signed': True,
                        'validated': validated,
                        'manifest_label': active_manifest.get('label', 'unknown')
                    }
            except json.JSONDecodeError:
                pass

        return False, {'signed': False, 'error': 'No manifest found'}

    except Exception as e:
        logging.error(f"C2PA verification failed for {file_path.name}: {e}")
        return False, {'signed': False, 'error': str(e)}


def find_transformed_assets() -> Dict[str, List[Path]]:
    """
    Scan transformed folders for available assets.

    Returns:
        Dict mapping asset type to list of paths
    """
    assets = {
        'images': [],
        'videos': []
    }

    # Scan image compression folders
    image_compression_base = TRANSFORMED_BASE / "compression" / "images"
    if image_compression_base.exists():
        for subfolder in image_compression_base.rglob("*"):
            if subfolder.is_file() and subfolder.suffix.lower() in ['.jpg', '.png']:
                assets['images'].append(subfolder)

    # Scan image editing folders
    image_editing_base = TRANSFORMED_BASE / "editing" / "images"
    if image_editing_base.exists():
        for subfolder in image_editing_base.rglob("*"):
            if subfolder.is_file() and subfolder.suffix.lower() in ['.jpg', '.png']:
                assets['images'].append(subfolder)

    # Scan video compression folders
    video_compression_base = TRANSFORMED_BASE / "compression" / "videos"
    if video_compression_base.exists():
        for subfolder in video_compression_base.rglob("*"):
            if subfolder.is_file() and subfolder.suffix.lower() == '.mp4':
                assets['videos'].append(subfolder)

    # Scan video editing folders
    video_editing_base = TRANSFORMED_BASE / "editing" / "videos"
    if video_editing_base.exists():
        for subfolder in video_editing_base.rglob("*"):
            if subfolder.is_file() and subfolder.suffix.lower() == '.mp4':
                assets['videos'].append(subfolder)

    logging.info(f"Found {len(assets['images'])} transformed images")
    logging.info(f"Found {len(assets['videos'])} transformed videos")

    return assets


def display_asset_menu(assets: List[Path], asset_type: str) -> Optional[Path]:
    """
    Display interactive menu for asset selection.

    Args:
        assets: List of asset paths
        asset_type: 'images' or 'videos'

    Returns:
        Selected asset path or None
    """
    if not assets:
        print(f"\nNo {asset_type} found in transformed folders.")
        return None

    print(f"\n{'='*60}")
    print(f"Available {asset_type.upper()} ({len(assets)} total)")
    print(f"{'='*60}")

    # Group by transformation type for easier browsing
    grouped = {}
    for asset in assets:
        # Extract transform type from path
        # e.g., transformed/compression/images/jpeg/q95/file.jpg -> "jpeg/q95"
        rel_path = asset.relative_to(TRANSFORMED_BASE)
        transform_key = f"{rel_path.parts[0]}/{rel_path.parts[1]}"
        if len(rel_path.parts) > 2:
            transform_key += f"/{rel_path.parts[2]}"

        if transform_key not in grouped:
            grouped[transform_key] = []
        grouped[transform_key].append(asset)

    # Display grouped assets
    idx = 1
    asset_map = {}
    for transform_key, files in sorted(grouped.items()):
        print(f"\n{transform_key}:")
        for file in sorted(files)[:5]:  # Show first 5 from each group
            print(f"  [{idx}] {file.name}")
            asset_map[idx] = file
            idx += 1
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")

    print(f"\n[0] Cancel")

    # Get user selection
    while True:
        try:
            choice = input("\nSelect asset number (or 0 to cancel): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                return None
            elif choice_num in asset_map:
                return asset_map[choice_num]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None


def select_platform() -> Optional[str]:
    """
    Interactive platform selection.

    Returns:
        Platform name or None
    """
    print(f"\n{'='*60}")
    print("SELECT PLATFORM")
    print(f"{'='*60}")

    platforms = list(PLATFORM_CONFIGS.keys())
    for idx, platform in enumerate(platforms, 1):
        modes = PLATFORM_CONFIGS[platform]['modes']
        print(f"  [{idx}] {platform.replace('_', ' ').title()} - Modes: {', '.join(modes)}")

    print(f"\n[0] Cancel")

    while True:
        try:
            choice = input("\nSelect platform number: ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(platforms):
                return platforms[choice_num - 1]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None


def select_mode(platform: str, asset_type: str) -> Optional[str]:
    """
    Interactive mode selection for platform.

    Args:
        platform: Platform name
        asset_type: 'images' or 'videos'

    Returns:
        Mode name or None
    """
    config = PLATFORM_CONFIGS[platform]

    # Filter modes based on asset type
    if asset_type == 'images':
        valid_modes = [m for m in config['modes'] if m in ['image', 'story']]
    else:
        valid_modes = [m for m in config['modes'] if m in ['video', 'story', 'reel', 'status']]

    if not valid_modes:
        print(f"\n{platform} does not support {asset_type}.")
        return None

    print(f"\n{'='*60}")
    print(f"SELECT MODE for {platform.upper()}")
    print(f"{'='*60}")

    for idx, mode in enumerate(valid_modes, 1):
        print(f"  [{idx}] {mode.title()}")

    print(f"\n[0] Cancel")

    while True:
        try:
            choice = input("\nSelect mode number: ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(valid_modes):
                return valid_modes[choice_num - 1]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return None


def prepare_upload(asset_path: Path, platform: str, mode: str) -> bool:
    """
    Prepare asset for platform upload.

    Args:
        asset_path: Path to asset file
        platform: Platform name
        mode: Upload mode

    Returns:
        Success status
    """
    try:
        # Create platform upload directory
        upload_dir = PLATFORM_TESTS_BASE / platform / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename (preserve original for tracking)
        output_path = upload_dir / asset_path.name

        # Check if file already exists
        if output_path.exists():
            print(f"\nFile already exists in uploads folder: {output_path.name}")
            overwrite = input("Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Cancelled.")
                return False

        # Verify C2PA signature
        print(f"\nVerifying C2PA signature...")
        is_signed, details = verify_c2pa_signature(asset_path)

        if not is_signed:
            print(f"WARNING: Asset is NOT signed with C2PA manifest!")
            print(f"  Error: {details.get('error', 'Unknown')}")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Cancelled.")
                return False
        else:
            validated = details.get('validated', False)
            status = "VALID" if validated else "INVALID"
            print(f"C2PA Status: SIGNED ({status})")
            print(f"  Manifest: {details.get('manifest_label', 'unknown')}")

        # Copy to upload folder
        shutil.copy2(asset_path, output_path)
        print(f"\nFile copied to: {output_path.relative_to(Path.cwd())}")

        # Generate upload instructions
        print(f"\n{'='*60}")
        print("UPLOAD INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"Platform: {platform.replace('_', ' ').title()}")
        print(f"Mode: {mode.title()}")
        print(f"File: {output_path.name}")
        print(f"\nPlatform Notes:")
        print(f"  {PLATFORM_CONFIGS[platform]['notes']}")
        print(f"\nSteps:")
        print(f"  1. Transfer file to mobile device (if needed)")
        print(f"  2. Upload to {platform} as {mode}")
        print(f"  3. Download the file from {platform}")
        print(f"  4. Save to: data/platform_tests/{platform}/returned/")
        print(f"  5. Rename using convention:")
        print(f"     {{original}}__{platform}__{mode}__{{timestamp}}.{{ext}}")
        print(f"  6. Log upload/download in platform_manifest.csv")
        print(f"{'='*60}")

        # Save metadata
        metadata = {
            'original_file': str(asset_path),
            'platform': platform,
            'mode': mode,
            'prepared_timestamp': datetime.now().isoformat(),
            'c2pa_signed': is_signed,
            'c2pa_validated': details.get('validated', False),
            'upload_path': str(output_path)
        }

        metadata_path = output_path.with_suffix('.upload_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved to: {metadata_path.name}")

        return True

    except Exception as e:
        logging.error(f"Failed to prepare upload: {e}")
        print(f"\nERROR: {e}")
        return False


def main():
    """Main entry point."""
    print("="*60)
    print("Platform Upload Preparation - C2PA Robustness Testing")
    print("Phase 2.5: Social Media Round-Trip Testing")
    print("="*60)

    # Check c2patool
    if not check_c2patool():
        print("\nERROR: c2patool not found!")
        print("Please install c2patool: cargo install c2patool")
        sys.exit(1)

    # Find transformed assets
    print("\nScanning transformed assets...")
    assets = find_transformed_assets()

    if not assets['images'] and not assets['videos']:
        print("\nNo transformed assets found!")
        print("Please run transformation scripts first:")
        print("  - scripts/transformations/compress_images.py")
        print("  - scripts/transformations/compress_videos.py")
        print("  - scripts/transformations/edit_assets.py")
        sys.exit(1)

    # Main loop
    while True:
        print(f"\n{'='*60}")
        print("MAIN MENU")
        print(f"{'='*60}")
        print(f"  [1] Prepare Image for Upload ({len(assets['images'])} available)")
        print(f"  [2] Prepare Video for Upload ({len(assets['videos'])} available)")
        print(f"  [0] Exit")

        try:
            choice = input("\nSelect option: ").strip()

            if choice == '0':
                print("\nExiting. Goodbye!")
                break
            elif choice == '1':
                # Image workflow
                asset = display_asset_menu(assets['images'], 'images')
                if not asset:
                    continue

                platform = select_platform()
                if not platform:
                    continue

                mode = select_mode(platform, 'images')
                if not mode:
                    continue

                success = prepare_upload(asset, platform, mode)
                if success:
                    print("\nUpload preparation complete!")
                    input("\nPress Enter to continue...")

            elif choice == '2':
                # Video workflow
                asset = display_asset_menu(assets['videos'], 'videos')
                if not asset:
                    continue

                platform = select_platform()
                if not platform:
                    continue

                mode = select_mode(platform, 'videos')
                if not mode:
                    continue

                success = prepare_upload(asset, platform, mode)
                if success:
                    print("\nUpload preparation complete!")
                    input("\nPress Enter to continue...")

            else:
                print("Invalid selection. Please try again.")

        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            print(f"\nERROR: {e}")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
