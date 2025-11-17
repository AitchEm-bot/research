"""
Data Loading and Preparation Module for Phase 4 Analysis
=========================================================

This module handles loading and normalizing data from CSV files for analysis.

Features:
- Load final_metrics.csv with column name normalization
- Load baseline validation data
- Split data by asset type and platform
- Handle missing values and data type conversions
- Column mappings for PHASE4.md compatibility

Usage:
    from analysis.data_analysis import load_and_prepare
    df = load_and_prepare.load_final_metrics()

Output:
    Returns normalized DataFrames ready for analysis
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.common import utils

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_analysis.log')


def load_final_metrics(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load final_metrics.csv and normalize column names for PHASE4.md compatibility.

    Column mappings:
    - vmaf → vmaf_stretched (backward compatibility)
    - alignment_method → vmaf_method
    - media_source → platform_source
    - trust_verified → trust_valid

    Args:
        csv_path: Optional path to CSV. Defaults to standard location.

    Returns:
        DataFrame with normalized columns and cleaned data types
    """
    if csv_path is None:
        csv_path = utils.DIRS['results_csv'] / "final_metrics.csv"

    logger.info(f"Loading final metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Log initial data stats
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Asset types: {df['asset_type'].value_counts().to_dict()}")
    logger.info(f"Transform types: {df['transform_type'].value_counts().to_dict()}")

    # Column name normalization for PHASE4.md compatibility
    column_mappings = {
        'vmaf': 'vmaf_stretched',  # Keep original vmaf as stretched
        'alignment_method': 'vmaf_method',
        'media_source': 'platform_source',
        'trust_verified': 'trust_valid'
    }

    # Apply mappings
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
            logger.info(f"Mapped column: {old_name} -> {new_name}")

    # Handle numeric columns with 'inf' and 'NA' strings
    numeric_columns = ['psnr', 'psnr_aligned', 'ssim', 'ssim_aligned',
                      'vmaf_stretched', 'vmaf_aligned']

    for col in numeric_columns:
        if col in df.columns:
            # Replace string representations
            df[col] = df[col].replace('inf', np.inf)
            df[col] = df[col].replace('NA', np.nan)
            df[col] = df[col].replace('N/A', np.nan)
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure integer columns
    int_columns = ['manifest_present', 'verified', 'signature_valid',
                   'hash_match', 'assertion_uris_match', 'trust_valid',
                   'lossless_match', 'lossless_transform']

    for col in int_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')

    # Log data quality stats
    logger.info(f"Manifest retention: {df['manifest_present'].mean():.1%}")
    logger.info(f"Verification rate: {df['verified'].mean():.1%}")

    return df


def load_baseline_validation(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load baseline validation data (original signed assets before transformation).

    Args:
        csv_path: Optional path to CSV. Defaults to standard location.

    Returns:
        DataFrame with baseline validation results
    """
    if csv_path is None:
        csv_path = utils.DIRS['results_csv'] / "c2pa_validation_baseline.csv"

    if not csv_path.exists():
        logger.warning(f"Baseline validation file not found: {csv_path}")
        return pd.DataFrame()

    logger.info(f"Loading baseline validation from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Apply same column mappings as final_metrics
    column_mappings = {
        'media_source': 'platform_source',
        'trust_verified': 'trust_valid'
    }

    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    logger.info(f"Loaded {len(df)} baseline samples")
    logger.info(f"Baseline manifest present: {df['manifest_present'].mean():.1%}")
    logger.info(f"Baseline verified: {df['verified'].mean():.1%}")

    return df


def split_by_asset_type(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into image and video subsets.

    Args:
        df: Input DataFrame with 'asset_type' column

    Returns:
        Tuple of (df_images, df_videos)
    """
    df_images = df[df['asset_type'] == 'image'].copy()
    df_videos = df[df['asset_type'] == 'video'].copy()

    logger.info(f"Split data: {len(df_images)} images, {len(df_videos)} videos")

    return df_images, df_videos


def split_by_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract platform testing samples (transform_type == 'platform_roundtrip').

    Args:
        df: Input DataFrame with 'transform_type' column

    Returns:
        DataFrame containing only platform testing samples
    """
    df_platform = df[df['transform_type'] == 'platform_roundtrip'].copy()

    if len(df_platform) > 0:
        logger.info(f"Extracted {len(df_platform)} platform samples")
        # Log platform distribution
        if 'platform' in df_platform.columns:
            platform_counts = df_platform['platform'].value_counts()
            logger.info(f"Platform distribution: {platform_counts.to_dict()}")
    else:
        logger.warning("No platform_roundtrip samples found")

    return df_platform


def prepare_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare quality metric columns for analysis.

    Handles:
    - Convert "inf" to np.inf for lossless transforms
    - Convert "NA" to np.nan for non-applicable metrics
    - Ensure proper numeric types
    - Validate lossless transform flags

    Args:
        df: Input DataFrame with quality metric columns

    Returns:
        DataFrame with cleaned quality metrics
    """
    df = df.copy()

    # Quality metric columns
    quality_cols = ['psnr', 'psnr_aligned', 'ssim', 'ssim_aligned',
                    'vmaf_stretched', 'vmaf_aligned']

    for col in quality_cols:
        if col not in df.columns:
            continue

        # Count special values before cleaning
        inf_count = (df[col] == np.inf).sum()
        nan_count = df[col].isna().sum()

        if inf_count > 0:
            logger.info(f"{col}: {inf_count} lossless values (inf)")
        if nan_count > 0:
            logger.info(f"{col}: {nan_count} N/A values")

    # Validate lossless transforms
    if 'lossless_transform' in df.columns and 'transform_type' in df.columns:
        lossless_types = df[df['lossless_transform'] == 1]['transform_type'].unique()
        if len(lossless_types) > 0:
            logger.info(f"Lossless transform types: {list(lossless_types)}")

    return df


def get_transform_categories(df: pd.DataFrame) -> dict:
    """
    Categorize transforms into groups for analysis.

    Categories:
    - compression: jpeg, png, h264, h265, fps
    - editing: resize, crop, rotate, brightness, contrast, saturation
    - platform: platform_roundtrip

    Args:
        df: Input DataFrame with 'transform_type' column

    Returns:
        Dictionary mapping category to list of transform types
    """
    categories = {
        'compression': ['jpeg_compression', 'png_compression',
                       'h264_compression', 'h265_compression', 'fps_adjustment'],
        'editing': ['resize', 'crop', 'rotation', 'brightness_adjustment',
                   'contrast_adjustment', 'saturation_adjustment'],
        'platform': ['platform_roundtrip']
    }

    # Create reverse mapping
    type_to_category = {}
    for category, types in categories.items():
        for t in types:
            type_to_category[t] = category

    # Add category column if not exists
    if 'transform_category' not in df.columns and 'transform_type' in df.columns:
        df['transform_category'] = df['transform_type'].map(type_to_category)

        # Log category distribution
        category_counts = df['transform_category'].value_counts()
        logger.info(f"Transform categories: {category_counts.to_dict()}")

    return categories


def load_all_data() -> dict:
    """
    Load all required data for Phase 4 analysis.

    Returns:
        Dictionary containing:
        - 'final_metrics': Full dataset
        - 'baseline': Baseline validation data
        - 'images': Image subset
        - 'videos': Video subset
        - 'platform': Platform testing subset
    """
    logger.info("=" * 60)
    logger.info("Loading all data for Phase 4 analysis")
    logger.info("=" * 60)

    # Load main dataset
    df = load_final_metrics()

    # Load baseline
    df_baseline = load_baseline_validation()

    # Split by type
    df_images, df_videos = split_by_asset_type(df)

    # Extract platform samples
    df_platform = split_by_platform(df)

    # Clean quality metrics
    df = prepare_quality_metrics(df)

    # Add categories
    get_transform_categories(df)

    data = {
        'final_metrics': df,
        'baseline': df_baseline,
        'images': df_images,
        'videos': df_videos,
        'platform': df_platform
    }

    logger.info("=" * 60)
    logger.info("Data loading complete")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Baseline samples: {len(df_baseline)}")
    logger.info(f"Image samples: {len(df_images)}")
    logger.info(f"Video samples: {len(df_videos)}")
    logger.info(f"Platform samples: {len(df_platform)}")
    logger.info("=" * 60)

    return data


if __name__ == "__main__":
    # Test loading
    data = load_all_data()

    # Display summary
    df = data['final_metrics']
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nManifest retention rate: {df['manifest_present'].mean():.1%}")
    print(f"Verification rate: {df['verified'].mean():.1%}")

    # Check for expected columns
    expected_cols = ['vmaf_stretched', 'vmaf_method', 'platform_source', 'trust_valid']
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        print(f"\nWarning: Missing expected columns: {missing}")
    else:
        print(f"\nAll expected columns present - OK")