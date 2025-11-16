#!/usr/bin/env python3
"""
Rename WhatsApp Platform Returns - Special handling for compressed/file_mode structure
"""

import os
import csv
from pathlib import Path
from datetime import datetime

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLATFORM_TESTS_BASE = PROJECT_ROOT / "data" / "platform_tests"

# Current timestamp for all renames
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

def rename_whatsapp_files():
    """Rename WhatsApp files in compressed/ and file_mode/ subdirectories."""

    print("="*80)
    print("WHATSAPP PLATFORM RETURNS RENAMING SCRIPT")
    print("="*80)
    print(f"\nTimestamp: {TIMESTAMP}\n")

    whatsapp_base = PLATFORM_TESTS_BASE / "whatsapp"
    rename_log = []
    total_renamed = 0

    # Process both modes
    modes = {
        'compressed': 'compressed/returned',
        'file_mode': 'file_mode/returned'
    }

    for mode, subpath in modes.items():
        returned_dir = whatsapp_base / subpath

        if not returned_dir.exists():
            print(f"[WARNING] {mode}: No returned folder found")
            continue

        files = [f for f in returned_dir.iterdir() if f.is_file()]
        if not files:
            print(f"[WARNING] {mode}: No files found")
            continue

        print(f"\n[*] Mode: {mode.upper()}")
        print(f"    Files found: {len(files)}")
        print("-" * 80)

        mode_renamed = 0
        mode_skipped = 0

        for file_path in files:
            original_name = file_path.name
            stem = file_path.stem
            ext = file_path.suffix

            # Check if already renamed
            if '__' in stem:
                print(f"    [SKIP] Already renamed: {original_name}")
                mode_skipped += 1
                continue

            # Build new filename
            new_name = f"{stem}__whatsapp__{mode}__{TIMESTAMP}{ext}"
            new_path = returned_dir / new_name

            # Rename file
            try:
                file_path.rename(new_path)
                print(f"    [OK] {original_name}")
                print(f"         -> {new_name}")

                rename_log.append({
                    'platform': 'whatsapp',
                    'mode': mode,
                    'original_name': original_name,
                    'new_name': new_name,
                    'timestamp': TIMESTAMP
                })

                mode_renamed += 1
                total_renamed += 1

            except Exception as e:
                print(f"    [ERROR] renaming {original_name}: {e}")

        print(f"\n   Mode summary: {mode_renamed} renamed, {mode_skipped} skipped")

    # Save rename log
    log_file = PLATFORM_TESTS_BASE / "rename_log_whatsapp.csv"
    if rename_log:
        with open(log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['platform', 'mode', 'original_name', 'new_name', 'timestamp'])
            writer.writeheader()
            writer.writerows(rename_log)
        print(f"\n[OK] Rename log saved to: {log_file}")

    # Final summary
    print("\n" + "="*80)
    print("WHATSAPP RENAMING COMPLETE")
    print("="*80)
    print(f"Total files renamed: {total_renamed}")
    print(f"Rename log: {log_file}")
    print()

if __name__ == "__main__":
    rename_whatsapp_files()
