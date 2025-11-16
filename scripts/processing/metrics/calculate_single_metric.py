#!/usr/bin/env python3
"""
Calculate Quality Metrics for Single Asset
==========================================

Calculates quality metrics for a single transformed asset and appends to quality_metrics.csv.

Usage:
    python scripts/metrics/calculate_single_metric.py <asset_path>

Example:
    python scripts/metrics/calculate_single_metric.py data/prepared_assets/transformed/editing/videos/video_55_saturation_plus19.mp4
"""

import sys
from pathlib import Path

# Add common utilities to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'common'))
import utils

# Import from main calculate_quality_metrics.py
sys.path.insert(0, str(Path(__file__).parent))
from calculate_quality_metrics import process_single_asset

# Configure logging using shared utility
logger = utils.setup_logging()

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

    # Append to quality_metrics.csv using shared utility
    csv_path = Path("data/results/quality_metrics.csv")

    if not csv_path.exists():
        utils.write_csv_header(csv_path, 'quality_metrics')

    utils.append_csv_row(csv_path, result, 'quality_metrics')
    print(f"\n[OK] Appended to {csv_path}")

if __name__ == "__main__":
    main()
