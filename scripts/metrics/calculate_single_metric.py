#!/usr/bin/env python3
"""
Calculate Quality Metrics for Single Asset
==========================================

Calculates quality metrics for a single transformed asset and appends to quality_metrics.csv.

Usage:
    python scripts/metrics/calculate_single_metric.py <asset_path>

Example:
    python scripts/metrics/calculate_single_metric.py data/transformed/editing/videos/video_55_saturation_plus19.mp4
"""

import sys
import csv
import logging
from pathlib import Path

# Import from main calculate_quality_metrics.py
sys.path.insert(0, str(Path(__file__).parent))
from calculate_quality_metrics import process_single_asset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_single_metric.py <asset_path>")
        sys.exit(1)

    asset_path = Path(sys.argv[1])

    if not asset_path.exists():
        print(f"Error: File not found: {asset_path}")
        sys.exit(1)

    print(f"Calculating metrics for: {asset_path.name}")

    # Calculate metrics
    result = process_single_asset(asset_path)

    if not result:
        print("Error: Failed to calculate metrics")
        sys.exit(1)

    # Print result
    print("\nMetrics calculated:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    # Append to quality_metrics.csv
    csv_path = Path("data/metrics/quality_metrics.csv")

    if csv_path.exists():
        # Append to existing CSV
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result)
        print(f"\n[OK] Appended to {csv_path}")
    else:
        # Create new CSV with header
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(result)
        print(f"\n[OK] Created new {csv_path}")

if __name__ == "__main__":
    main()
