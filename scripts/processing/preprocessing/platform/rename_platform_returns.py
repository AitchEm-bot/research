#!/usr/bin/env python3
"""
Rename Platform Returns - C2PA Robustness Testing Pipeline
============================================================

Renames downloaded platform files to standardized format:
{original}__{platform}__{mode}__{timestamp}.{ext}

Usage:
    python rename_platform_returns.py

Author: Hani Moustafa (C2PA Robustness Research Project)
"""

import os
import csv
from pathlib import Path
from datetime import datetime

# Base directories - script is in scripts/processing/preprocessing/platform/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PLATFORM_TESTS_BASE = PROJECT_ROOT / "data" / "prepared_assets" / "platform_tests"

# Platform mode mappings (from prepare_platform_uploads.py)
PLATFORM_MODES = {
    'instagram': 'post',
    'twitter': 'upload',
    'facebook': 'post',
    'youtube': 'upload',
    'tiktok': 'upload',
    'whatsapp': 'compressed'
}

# Current timestamp for all renames (as requested)
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

def load_upload_tracking():
    """
    Load the auto_sample_tracking.csv to map platform uploads.

    Returns:
        dict: Mapping of {platform: [uploaded_filenames]}
    """
    tracking_file = PLATFORM_TESTS_BASE / "auto_sample_tracking.csv"
    platform_uploads = {platform: [] for platform in PLATFORM_MODES.keys()}

    if not tracking_file.exists():
        print(f"WARNING: Tracking file not found: {tracking_file}")
        return platform_uploads

    with open(tracking_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            platform = row['platform']

            # Handle legacy youtube_shorts -> youtube mapping
            if platform == 'youtube_shorts':
                platform = 'youtube'

            if platform in platform_uploads:
                upload_path = Path(row['upload_path'])
                filename = upload_path.name
                platform_uploads[platform].append(filename)

    return platform_uploads

def rename_platform_files():
    """
    Rename all downloaded platform files to standardized format.

    Renames files in data/prepared_assets/platform_tests/{platform}/returned/ to:
    {original}__{platform}__{mode}__{timestamp}.{ext}
    """
    platforms = list(PLATFORM_MODES.keys())

    print("="*80)
    print("PLATFORM RETURNS RENAMING SCRIPT")
    print("="*80)
    print(f"\nTimestamp for all files: {TIMESTAMP}")
    print()

    # Load upload tracking
    platform_uploads = load_upload_tracking()

    rename_log = []
    total_renamed = 0
    total_skipped = 0

    for platform in platforms:
        returned_dir = PLATFORM_TESTS_BASE / platform / "returned"

        if not returned_dir.exists():
            print(f"[WARNING] Platform '{platform}': No returned folder found")
            continue

        files = list(returned_dir.iterdir())
        if not files:
            print(f"[WARNING] Platform '{platform}': No files in returned folder")
            continue

        print(f"\n[*] Platform: {platform.upper()}")
        print(f"    Mode: {PLATFORM_MODES[platform]}")
        print(f"    Files found: {len(files)}")
        print("-" * 80)

        platform_renamed = 0
        platform_skipped = 0

        for file_path in files:
            if file_path.is_dir():
                continue

            original_name = file_path.name
            stem = file_path.stem
            ext = file_path.suffix

            # Check if already renamed (contains double underscores)
            if '__' in stem:
                print(f"    [SKIP] Already renamed: {original_name}")
                platform_skipped += 1
                total_skipped += 1
                continue

            # Build new filename
            mode = PLATFORM_MODES[platform]
            new_name = f"{stem}__{platform}__{mode}__{TIMESTAMP}{ext}"
            new_path = returned_dir / new_name

            # Rename file
            try:
                file_path.rename(new_path)
                print(f"    [OK] {original_name}")
                print(f"         -> {new_name}")

                rename_log.append({
                    'platform': platform,
                    'original_name': original_name,
                    'new_name': new_name,
                    'mode': mode,
                    'timestamp': TIMESTAMP
                })

                platform_renamed += 1
                total_renamed += 1

            except Exception as e:
                print(f"    [ERROR] renaming {original_name}: {e}")

        print(f"\n   Platform summary: {platform_renamed} renamed, {platform_skipped} skipped")

    # Save rename log
    log_file = PLATFORM_TESTS_BASE / "rename_log.csv"
    if rename_log:
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['platform', 'original_name', 'new_name', 'mode', 'timestamp'])
            writer.writeheader()
            writer.writerows(rename_log)
        print(f"\n[OK] Rename log saved to: {log_file}")

    # Final summary
    print("\n" + "="*80)
    print("RENAMING COMPLETE")
    print("="*80)
    print(f"Total files renamed: {total_renamed}")
    print(f"Total files skipped: {total_skipped}")
    print(f"Rename log: {log_file}")
    print()

if __name__ == "__main__":
    rename_platform_files()
