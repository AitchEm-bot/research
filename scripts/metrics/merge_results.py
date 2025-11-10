"""
Results Merge Script for C2PA Robustness Testing
=================================================

This script merges C2PA verification results and quality metrics into a
single comprehensive dataset for analysis.

Input CSVs:
- data/metrics/c2pa_validation.csv (C2PA verification flags and metadata)
- data/metrics/quality_metrics.csv (PSNR/SSIM/VMAF scores)

Output CSV:
- data/metrics/final_metrics.csv (complete dataset matching CLAUDE.md schema)

Features:
- Merges on filename (primary key)
- Preserves all metadata columns (seed, model_version, transform details)
- Validates column types and completeness
- Reports missing or mismatched rows

Usage:
    python scripts/metrics/merge_results.py

Output:
    data/metrics/final_metrics.csv (304 rows expected)
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
OUTPUT_CSV = Path("data/metrics/final_metrics.csv")

# Final CSV column schema (matches CLAUDE.md specification)
FINAL_COLUMNS = [
    'filename',
    'asset_type',
    'transform_type',
    'transform_level',
    'seed',
    'model_version',
    'manifest_present',
    'verified',
    'signature_valid',
    'hash_match',
    'assertion_uris_match',
    'trust_verified',
    'validation_state',
    'failure_reason',
    'psnr',
    'ssim',
    'vmaf',
    'c2pa_processing_time_ms',
    'quality_processing_time_ms',
    'timestamp'
]


def load_c2pa_data() -> pd.DataFrame:
    """
    Load C2PA verification data.

    Returns:
        DataFrame with C2PA validation results
    """
    if not C2PA_CSV.exists():
        logging.error(f"C2PA validation CSV not found: {C2PA_CSV}")
        sys.exit(1)

    df = pd.read_csv(C2PA_CSV)
    logging.info(f"Loaded C2PA data: {len(df)} rows")

    # Rename processing_time_ms to c2pa_processing_time_ms for clarity
    if 'processing_time_ms' in df.columns:
        df.rename(columns={'processing_time_ms': 'c2pa_processing_time_ms'}, inplace=True)

    # Drop timestamp from C2PA data (we'll use quality metrics timestamp)
    if 'timestamp' in df.columns:
        df.drop(columns=['timestamp'], inplace=True)

    return df


def load_quality_data() -> pd.DataFrame:
    """
    Load quality metrics data.

    Returns:
        DataFrame with quality metrics
    """
    if not QUALITY_CSV.exists():
        logging.error(f"Quality metrics CSV not found: {QUALITY_CSV}")
        sys.exit(1)

    df = pd.read_csv(QUALITY_CSV)
    logging.info(f"Loaded quality metrics data: {len(df)} rows")

    # Rename processing_time_ms to quality_processing_time_ms for clarity
    if 'processing_time_ms' in df.columns:
        df.rename(columns={'processing_time_ms': 'quality_processing_time_ms'}, inplace=True)

    # Drop calculation_error column (errors logged elsewhere)
    if 'calculation_error' in df.columns:
        df.drop(columns=['calculation_error'], inplace=True)

    return df


def merge_datasets(c2pa_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge C2PA and quality metrics datasets.

    Args:
        c2pa_df: C2PA verification DataFrame
        quality_df: Quality metrics DataFrame

    Returns:
        Merged DataFrame
    """
    # Merge on filename (inner join - only rows present in both)
    merged = pd.merge(
        c2pa_df,
        quality_df,
        on='filename',
        how='inner',
        suffixes=('_c2pa', '_quality')
    )

    logging.info(f"Merged dataset: {len(merged)} rows")

    # Resolve asset_type conflict (should be same in both, but prefer c2pa)
    if 'asset_type_c2pa' in merged.columns and 'asset_type_quality' in merged.columns:
        merged['asset_type'] = merged['asset_type_c2pa']
        merged.drop(columns=['asset_type_c2pa', 'asset_type_quality'], inplace=True)
    elif 'asset_type_c2pa' in merged.columns:
        merged.rename(columns={'asset_type_c2pa': 'asset_type'}, inplace=True)
    elif 'asset_type_quality' in merged.columns:
        merged.rename(columns={'asset_type_quality': 'asset_type'}, inplace=True)

    # Check for missing rows
    c2pa_only = set(c2pa_df['filename']) - set(quality_df['filename'])
    quality_only = set(quality_df['filename']) - set(c2pa_df['filename'])

    if c2pa_only:
        logging.warning(f"Files in C2PA but not quality metrics: {len(c2pa_only)}")
        for fname in list(c2pa_only)[:5]:  # Show first 5
            logging.warning(f"  - {fname}")

    if quality_only:
        logging.warning(f"Files in quality metrics but not C2PA: {len(quality_only)}")
        for fname in list(quality_only)[:5]:  # Show first 5
            logging.warning(f"  - {fname}")

    return merged


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
        logging.warning(f"Missing columns in final dataset: {missing_cols}")

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
    int_cols = ['manifest_present', 'verified', 'signature_valid', 'hash_match',
                'assertion_uris_match', 'trust_verified']
    for col in int_cols:
        if col in df.columns:
            if df[col].dtype not in ['int64', 'Int64']:
                logging.warning(f"Column {col} should be integer, got {df[col].dtype}")

    # Verify float columns
    float_cols = ['psnr', 'ssim', 'vmaf', 'c2pa_processing_time_ms',
                  'quality_processing_time_ms']
    for col in float_cols:
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
    c2pa_df = load_c2pa_data()
    quality_df = load_quality_data()

    # Merge
    merged_df = merge_datasets(c2pa_df, quality_df)

    # Reorder columns
    final_df = reorder_columns(merged_df)

    # Validate
    validate_dataset(final_df)

    # Save final dataset
    final_df.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"Final dataset saved: {OUTPUT_CSV.absolute()}")

    # Summary
    logging.info("=" * 60)
    logging.info("Merge Complete")
    logging.info(f"  Input: {len(c2pa_df)} C2PA rows + {len(quality_df)} quality rows")
    logging.info(f"  Output: {len(final_df)} merged rows")
    logging.info(f"  Columns: {len(final_df.columns)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
