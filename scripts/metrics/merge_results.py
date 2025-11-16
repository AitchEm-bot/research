"""
Results Merge Script for C2PA Robustness Testing
=================================================

This script combines transformed asset metrics with optional platform round-trip
results into a single comprehensive dataset for analysis.

Input CSVs:
- data/metrics/c2pa_validation.csv (C2PA verification results for transformed assets)
- data/metrics/quality_metrics.csv (Quality metrics: PSNR/SSIM/VMAF for transformed assets)
- data/metrics/platform_results.csv (Platform round-trip results, Phase 2.5, optional)

Output CSV:
- data/metrics/final_metrics.csv (Complete dataset matching CLAUDE.md schema)

Workflow:
1. Load c2pa_validation.csv (C2PA verification: manifest_present, verified, etc.)
2. Load quality_metrics.csv (Quality metrics: PSNR, SSIM, VMAF, lossless flags)
3. Merge on filename (inner join - both C2PA and quality must exist)
4. Optionally append platform_results.csv (platform round-trip results)
5. Reorder columns to match final schema
6. Validate data types and completeness
7. Save to final_metrics.csv

Features:
- Merges C2PA verification with quality metrics on filename
- Appends platform results if available (Phase 2.5)
- Preserves all metadata columns (seed, model_version, transform details)
- Validates column types and completeness
- Reports missing or mismatched rows

Usage:
    python scripts/metrics/merge_results.py

Output:
    data/metrics/final_metrics.csv (~3,460 transformed + ~195 platform rows)
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Ensure log directory exists
Path("data/metrics").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/metrics/merge_results.log')
    ]
)

# Configuration
C2PA_CSV = Path("data/metrics/c2pa_validation.csv")
QUALITY_CSV = Path("data/metrics/quality_metrics.csv")
PLATFORM_CSV = Path("data/metrics/platform_results.csv")  # Phase 2.5 optional input
OUTPUT_CSV = Path("data/metrics/final_metrics.csv")

# Final CSV column schema (matches CLAUDE.md specification + Phase 2.5/4 enhancements)
# Combines C2PA verification, quality metrics, and optional platform testing data
FINAL_COLUMNS = [
    # Core metadata (from both C2PA and quality CSVs)
    'filename',                  # Transformed asset filename
    'asset_type',                # 'image' or 'video'
    'transform_type',            # Transform category (e.g., 'jpeg_compression', 'platform_roundtrip')
    'transform_level',           # Transform parameter (e.g., 'q95', 'platform_roundtrip')
    'seed',                      # Generation seed (empty for external media)
    'model_version',             # 'SD1.4', 'SVD', or 'Veo3.1'

    # C2PA verification metrics (from c2pa_validation.csv)
    'manifest_present',          # Boolean (0/1): C2PA manifest exists
    'verified',                  # Boolean (0/1): INTEGRITY validation passed (claimSignature.validated)
    'signature_valid',           # Boolean (0/1): Cryptographic signature valid
    'hash_match',                # Boolean (0/1): Hash consistency (dataHash or bmffHash)
    'assertion_uris_match',      # Boolean (0/1): All assertion URI hashes match
    'trust_verified',            # Boolean (0/1): Certificate trust chain (informational, NOT failure metric)
    'validation_state',          # String: c2patool validation_state field
    'failure_reason',            # String: Human-readable failure description

    # Quality metrics (from quality_metrics.csv)
    'psnr',                      # Peak Signal-to-Noise Ratio (dB, 'inf' for lossless, 'NA' for videos) - stretched (scales distorted to reference)
    'psnr_aligned',              # PSNR with aspect ratio alignment (crops reference if aspect changed, isolates content distortion)
    'ssim',                      # Structural Similarity Index (0-1, 'NA' for videos) - stretched (scales distorted to reference)
    'ssim_aligned',              # SSIM with aspect ratio alignment (crops reference if aspect changed, isolates content distortion)
    'vmaf',                      # Video Multimethod Assessment Fusion (0-100, traditional method, scales distorted to reference)
    'vmaf_aligned',              # VMAF with aspect ratio alignment (crops reference if aspect changed, more accurate for platform transforms)
    'alignment_method',          # Alignment method used for aligned metrics: 'same_aspect_ratio', 'crop_reference_center_square', 'scale_both_to_minimum'
    'lossless_match',            # Boolean (0/1): pixels identical to original (PSNR >= 100 dB)
    'lossless_transform',        # Boolean (0/1): mathematically lossless operation (png_c0, png_c9)
    'processing_time_ms',        # Quality metric calculation time (milliseconds)
    'calculation_error',         # Error message if quality metric calculation failed (empty if successful)
    'timestamp',                 # ISO 8601 timestamp when metrics calculated

    # Phase 2.5 optional columns (only if platform testing performed)
    'platform',                  # Platform name (instagram, twitter, facebook, youtube_shorts, tiktok, whatsapp)
    'platform_mode',             # Upload mode (video, image, story, reel, short, status)
    'media_source',              # Media origin ("internal" for SD1.4/SVD, "external" for Veo3.1/Sora/etc.)
    'upload_timestamp',          # ISO 8601 timestamp (optional, from manual CSV)
    'download_timestamp'         # ISO 8601 timestamp
]


def load_c2pa_data() -> pd.DataFrame:
    """
    Load C2PA verification data (transformed assets only).

    Returns:
        DataFrame with C2PA verification results
    """
    if not C2PA_CSV.exists():
        logging.error(f"C2PA validation CSV not found: {C2PA_CSV}")
        sys.exit(1)

    df = pd.read_csv(C2PA_CSV)
    logging.info(f"Loaded C2PA validation data: {len(df)} rows")

    return df


def load_quality_data() -> pd.DataFrame:
    """
    Load quality metrics data (transformed assets only).

    Returns:
        DataFrame with quality metrics
    """
    if not QUALITY_CSV.exists():
        logging.error(f"Quality metrics CSV not found: {QUALITY_CSV}")
        sys.exit(1)

    df = pd.read_csv(QUALITY_CSV)
    logging.info(f"Loaded transformed assets quality metrics: {len(df)} rows")

    # Drop calculation_error column if present (errors logged elsewhere)
    if 'calculation_error' in df.columns:
        df.drop(columns=['calculation_error'], inplace=True)

    return df


def load_platform_data() -> pd.DataFrame:
    """
    Load Phase 2.5 platform results data (optional).

    Returns:
        DataFrame with platform results, or empty DataFrame if not available
    """
    if not PLATFORM_CSV.exists():
        logging.info("Platform results CSV not found - skipping platform data (Phase 2.5 not performed)")
        return pd.DataFrame()

    df = pd.read_csv(PLATFORM_CSV)
    logging.info(f"Loaded platform results data: {len(df)} rows")

    return df


def merge_datasets(c2pa_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge C2PA validation with quality metrics on filename (inner join).

    Args:
        c2pa_df: C2PA verification DataFrame
        quality_df: Quality metrics DataFrame

    Returns:
        Merged DataFrame with both C2PA and quality data
    """
    logging.info("Merging C2PA validation with quality metrics...")

    # Perform inner join on filename
    merged_df = pd.merge(
        c2pa_df,
        quality_df,
        on='filename',
        how='inner',
        suffixes=('_c2pa', '_quality')
    )

    logging.info(f"Merged dataset: {len(merged_df)} rows")

    # Report any mismatches
    c2pa_only = set(c2pa_df['filename']) - set(quality_df['filename'])
    quality_only = set(quality_df['filename']) - set(c2pa_df['filename'])

    if c2pa_only:
        logging.warning(f"Files in C2PA validation but not in quality metrics: {len(c2pa_only)}")
        for fname in list(c2pa_only)[:5]:
            logging.warning(f"  - {fname}")

    if quality_only:
        logging.warning(f"Files in quality metrics but not in C2PA validation: {len(quality_only)}")
        for fname in list(quality_only)[:5]:
            logging.warning(f"  - {fname}")

    # Resolve duplicate columns (prefer quality_df for metadata like asset_type, transform_type, etc.)
    # C2PA verification columns should come from c2pa_df
    # Quality metrics should come from quality_df

    # Drop duplicate metadata columns from C2PA (keep from quality)
    cols_to_drop = []
    for col in merged_df.columns:
        if col.endswith('_c2pa'):
            base_col = col.replace('_c2pa', '')
            quality_col = f"{base_col}_quality"
            if quality_col in merged_df.columns:
                # Keep the quality version, drop the C2PA version
                merged_df[base_col] = merged_df[quality_col]
                cols_to_drop.append(col)
                cols_to_drop.append(quality_col)
            else:
                # No quality version, rename C2PA version to base name
                merged_df[base_col] = merged_df[col]
                cols_to_drop.append(col)
        elif col.endswith('_quality') and col.replace('_quality', '') not in [c.replace('_c2pa', '') for c in merged_df.columns if c.endswith('_c2pa')]:
            # Quality column with no C2PA counterpart
            base_col = col.replace('_quality', '')
            merged_df[base_col] = merged_df[col]
            cols_to_drop.append(col)

    # Drop resolved columns
    merged_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return merged_df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns to match final schema.

    Args:
        df: Merged DataFrame

    Returns:
        DataFrame with reordered columns
    """
    # Check which columns are present
    available_cols = [col for col in FINAL_COLUMNS if col in df.columns]
    missing_cols = [col for col in FINAL_COLUMNS if col not in df.columns]

    if missing_cols:
        logging.info(f"Columns not present in dataset: {missing_cols}")

    # Reorder available columns
    df = df[available_cols]

    return df


def validate_dataset(df: pd.DataFrame):
    """
    Validate final dataset for correctness.

    Args:
        df: Final merged DataFrame
    """
    logging.info("=" * 60)
    logging.info("Dataset Validation")
    logging.info("=" * 60)

    # Row count
    logging.info(f"Total rows: {len(df)}")
    expected_rows = 304  # 180 images + 84 videos + 40 additional
    if len(df) != expected_rows:
        logging.warning(f"Expected {expected_rows} rows, got {len(df)}")

    # Column types
    logging.info("Column data types:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        logging.info(f"  {col}: {dtype} ({null_count} null values)")

    # Check for duplicates
    duplicates = df[df.duplicated(subset=['filename'], keep=False)]
    if not duplicates.empty:
        logging.error(f"Found {len(duplicates)} duplicate filenames!")
        for fname in duplicates['filename'].unique()[:5]:
            logging.error(f"  - {fname}")

    # Verify integer columns
    int_cols = ['lossless_match', 'lossless_transform']
    for col in int_cols:
        if col in df.columns:
            if df[col].dtype not in ['int64', 'Int64']:
                logging.warning(f"Column {col} should be integer, got {df[col].dtype}")

    # Verify metric columns (may be string due to 'inf' or 'NA' values)
    metric_cols = ['psnr', 'ssim', 'vmaf', 'vmaf_aligned', 'processing_time_ms']
    for col in metric_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            logging.info(f"  {col}: {non_null} non-null values")

    # Asset type distribution
    if 'asset_type' in df.columns:
        logging.info("Asset type distribution:")
        for asset_type, count in df['asset_type'].value_counts().items():
            logging.info(f"  {asset_type}: {count}")

    # Failure reason distribution
    if 'failure_reason' in df.columns:
        logging.info("Failure reason distribution:")
        for reason, count in df['failure_reason'].value_counts().items():
            logging.info(f"  {reason}: {count}")

    # Transform type distribution
    if 'transform_type' in df.columns:
        logging.info("Transform type distribution:")
        for transform, count in df['transform_type'].value_counts().items():
            logging.info(f"  {transform}: {count}")

    logging.info("=" * 60)


def main():
    """Main entry point."""
    logging.info("=" * 60)
    logging.info("Results Merge Script - C2PA Robustness Testing")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Pandas version: {pd.__version__}")
    logging.info("=" * 60)

    # Load datasets
    c2pa_df = load_c2pa_data()        # C2PA verification results
    quality_df = load_quality_data()  # Quality metrics
    platform_df = load_platform_data()  # Phase 2.5 optional platform round-trip results

    # Merge C2PA validation with quality metrics (inner join on filename)
    merged_df = merge_datasets(c2pa_df, quality_df)

    # Start with merged C2PA + quality data
    final_df = merged_df.copy()

    # Append platform results if available (Phase 2.5)
    if not platform_df.empty:
        logging.info("Appending platform results to dataset...")

        # Add platform-specific columns to merged_df if not present
        for col in ['platform', 'platform_mode', 'media_source', 'upload_timestamp', 'download_timestamp']:
            if col not in final_df.columns:
                final_df[col] = ''

        # Ensure platform_df has all columns from merged_df
        for col in final_df.columns:
            if col not in platform_df.columns:
                if col not in ['platform', 'platform_mode', 'media_source', 'upload_timestamp', 'download_timestamp']:
                    # Column should exist in platform results
                    logging.warning(f"Column '{col}' missing from platform results")

        # Append platform results
        combined_df = pd.concat([final_df, platform_df], ignore_index=True)
        logging.info(f"Combined dataset: {len(combined_df)} rows ({len(final_df)} transformed + {len(platform_df)} platform)")
        final_df = combined_df

    # Reorder columns
    final_df = reorder_columns(final_df)

    # Validate
    validate_dataset(final_df)

    # Save final dataset
    final_df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Final dataset saved: {OUTPUT_CSV.absolute()}")

    # Summary
    logging.info("=" * 60)
    logging.info("Merge Complete")
    if platform_df.empty:
        logging.info(f"  Input: {len(c2pa_df)} C2PA rows + {len(quality_df)} quality rows → {len(merged_df)} merged")
    else:
        logging.info(f"  Input: {len(c2pa_df)} C2PA rows + {len(quality_df)} quality rows → {len(merged_df)} merged + {len(platform_df)} platform")
    logging.info(f"  Output: {len(final_df)} total rows")
    logging.info(f"  Columns: {len(final_df.columns)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
