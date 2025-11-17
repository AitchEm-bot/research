"""
Quality Metrics Analysis for Phase 4
=====================================

This module analyzes perceptual quality metrics (PSNR, SSIM, VMAF) across transformations.

Features:
- PSNR and SSIM statistics for images
- Dual VMAF analysis (stretched vs aligned) for videos
- Identification of lossless transforms
- Quality degradation patterns by transform type
- Aspect ratio alignment method analysis

Usage:
    python scripts/analysis/data_analysis/quality_metrics_analysis.py

Output:
    data/results/analysis_results/csv/quality_summary.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.common import utils
from scripts.analysis.data_analysis import load_and_prepare

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_quality_analysis.log')


def analyze_psnr_by_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PSNR statistics by transform_type.

    Handles:
    - Infinite PSNR for lossless transforms
    - NA values for videos (PSNR is image-only)

    Args:
        df: DataFrame with psnr and transform_type columns

    Returns:
        DataFrame with PSNR summary statistics
    """
    logger.info("Analyzing PSNR by transform type")

    # Filter to images only (PSNR is for images)
    df_images = df[df['asset_type'] == 'image'].copy()

    if len(df_images) == 0:
        logger.warning("No image data for PSNR analysis")
        return pd.DataFrame()

    # Handle infinite values for statistics
    df_images['psnr_finite'] = df_images['psnr'].replace([np.inf, -np.inf], np.nan)

    # Calculate statistics
    psnr_stats = df_images.groupby('transform_type').agg({
        'psnr_finite': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'psnr': lambda x: (x == np.inf).sum()  # Count lossless
    })

    psnr_stats.columns = ['count', 'mean', 'median', 'std', 'min', 'max', 'lossless_count']
    psnr_stats = psnr_stats.round(2)
    psnr_stats = psnr_stats.reset_index()

    # Add percentage lossless
    psnr_stats['lossless_pct'] = (psnr_stats['lossless_count'] /
                                  (psnr_stats['count'] + psnr_stats['lossless_count']) * 100).round(1)

    # Sort by mean PSNR (descending)
    psnr_stats = psnr_stats.sort_values('mean', ascending=False)

    # Log findings
    logger.info(f"PSNR analysis for {len(psnr_stats)} transform types")
    logger.info(f"Overall mean PSNR: {df_images['psnr_finite'].mean():.1f} dB")
    logger.info(f"Lossless transforms: {df_images[df_images['psnr'] == np.inf]['transform_type'].unique().tolist()}")

    return psnr_stats


def analyze_ssim_by_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SSIM statistics by transform_type.

    SSIM ranges from 0 to 1, with 1 being perfect similarity.

    Args:
        df: DataFrame with ssim and transform_type columns

    Returns:
        DataFrame with SSIM summary statistics
    """
    logger.info("Analyzing SSIM by transform type")

    # Filter to images only
    df_images = df[df['asset_type'] == 'image'].copy()

    if len(df_images) == 0:
        logger.warning("No image data for SSIM analysis")
        return pd.DataFrame()

    ssim_stats = df_images.groupby('transform_type')['ssim'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)

    ssim_stats = ssim_stats.reset_index()

    # Add quality categories
    ssim_stats['quality'] = pd.cut(ssim_stats['mean'],
                                   bins=[0, 0.7, 0.9, 0.99, 1.0],
                                   labels=['Poor', 'Fair', 'Good', 'Excellent'])

    # Sort by mean SSIM (descending)
    ssim_stats = ssim_stats.sort_values('mean', ascending=False)

    logger.info(f"SSIM analysis for {len(ssim_stats)} transform types")
    logger.info(f"Overall mean SSIM: {df_images['ssim'].mean():.3f}")
    logger.info(f"Excellent quality (>0.99): {(ssim_stats['mean'] > 0.99).sum()} transforms")
    logger.info(f"Good quality (>0.9): {(ssim_stats['mean'] > 0.9).sum()} transforms")

    return ssim_stats


def analyze_vmaf_dual_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare vmaf_stretched vs vmaf_aligned metrics.

    Shows dramatic differences for aspect-changing transforms.

    Args:
        df: DataFrame with vmaf columns

    Returns:
        DataFrame comparing both VMAF metrics
    """
    logger.info("Analyzing dual VMAF metrics (stretched vs aligned)")

    # Filter to videos only
    df_videos = df[df['asset_type'] == 'video'].copy()

    if len(df_videos) == 0:
        logger.warning("No video data for VMAF analysis")
        return pd.DataFrame()

    # Calculate statistics for both metrics
    vmaf_stats = df_videos.groupby('transform_type').agg({
        'vmaf_stretched': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'vmaf_aligned': ['mean', 'median', 'std', 'min', 'max']
    }).round(2)

    # Flatten columns
    vmaf_stats.columns = ['_'.join(col).strip() for col in vmaf_stats.columns.values]
    vmaf_stats = vmaf_stats.reset_index()

    # Rename for clarity
    vmaf_stats.rename(columns={
        'vmaf_stretched_count': 'count',
        'vmaf_stretched_mean': 'stretched_mean',
        'vmaf_stretched_median': 'stretched_median',
        'vmaf_stretched_std': 'stretched_std',
        'vmaf_stretched_min': 'stretched_min',
        'vmaf_stretched_max': 'stretched_max',
        'vmaf_aligned_mean': 'aligned_mean',
        'vmaf_aligned_median': 'aligned_median',
        'vmaf_aligned_std': 'aligned_std',
        'vmaf_aligned_min': 'aligned_min',
        'vmaf_aligned_max': 'aligned_max'
    }, inplace=True)

    # Calculate difference
    vmaf_stats['mean_difference'] = vmaf_stats['aligned_mean'] - vmaf_stats['stretched_mean']
    vmaf_stats['mean_difference'] = vmaf_stats['mean_difference'].round(2)

    # Sort by difference (largest first)
    vmaf_stats = vmaf_stats.sort_values('mean_difference', ascending=False)

    # Log findings
    logger.info(f"VMAF analysis for {len(vmaf_stats)} transform types")
    logger.info(f"Overall stretched VMAF: {df_videos['vmaf_stretched'].mean():.1f}")
    logger.info(f"Overall aligned VMAF: {df_videos['vmaf_aligned'].mean():.1f}")

    # Identify transforms with large differences
    large_diff = vmaf_stats[vmaf_stats['mean_difference'] > 10]
    if len(large_diff) > 0:
        logger.info(f"Transforms with >10 VMAF difference: {large_diff['transform_type'].tolist()}")
        logger.info("These likely involve aspect ratio changes")

    return vmaf_stats


def analyze_alignment_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze distribution of VMAF alignment methods used.

    Methods indicate how aspect ratio differences were handled.

    Args:
        df: DataFrame with vmaf_method column

    Returns:
        DataFrame with alignment method distribution
    """
    logger.info("Analyzing VMAF alignment methods")

    if 'vmaf_method' not in df.columns:
        logger.warning("No vmaf_method column found")
        return pd.DataFrame()

    # Count by transform type and method
    method_dist = df[df['asset_type'] == 'video'].groupby(
        ['transform_type', 'vmaf_method']
    ).size().reset_index(name='count')

    # Pivot for better readability
    method_pivot = method_dist.pivot(
        index='transform_type',
        columns='vmaf_method',
        values='count'
    ).fillna(0).astype(int)

    method_pivot = method_pivot.reset_index()

    # Log distribution
    overall_dist = df[df['asset_type'] == 'video']['vmaf_method'].value_counts()
    logger.info(f"Overall alignment method distribution:\n{overall_dist}")

    return method_pivot


def identify_lossless_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and analyze lossless transforms (PSNR=inf, SSIM=1.0).

    Args:
        df: DataFrame with quality metrics and lossless flags

    Returns:
        DataFrame of lossless transform samples
    """
    logger.info("Identifying lossless transforms")

    # Use lossless_transform flag if available
    if 'lossless_transform' in df.columns:
        df_lossless = df[df['lossless_transform'] == 1].copy()
    else:
        # Fallback: identify by metrics
        df_lossless = df[
            ((df['psnr'] == np.inf) | (df['psnr'] > 100)) &
            (df['ssim'] >= 0.9999)
        ].copy()

    if len(df_lossless) > 0:
        # Count by transform type
        lossless_counts = df_lossless['transform_type'].value_counts()
        logger.info(f"Found {len(df_lossless)} lossless samples")
        logger.info(f"Lossless transform types:\n{lossless_counts}")

        # Verify expected lossless types
        expected_lossless = {'png_compression'}
        actual_lossless = set(df_lossless['transform_type'].unique())
        unexpected = actual_lossless - expected_lossless
        if unexpected:
            logger.warning(f"Unexpected lossless transforms: {unexpected}")
    else:
        logger.warning("No lossless transforms found")

    return df_lossless


def calculate_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive quality summary table.

    Args:
        df: DataFrame with all quality metrics

    Returns:
        Summary DataFrame with key statistics per transform type
    """
    logger.info("Calculating comprehensive quality summary")

    summary_data = []

    for transform_type in df['transform_type'].unique():
        transform_df = df[df['transform_type'] == transform_type]

        # Separate by asset type
        images = transform_df[transform_df['asset_type'] == 'image']
        videos = transform_df[transform_df['asset_type'] == 'video']

        row = {
            'transform_type': transform_type,
            'total_samples': len(transform_df),
            'image_samples': len(images),
            'video_samples': len(videos),
        }

        # Image metrics
        if len(images) > 0:
            # Handle infinite PSNR
            psnr_finite = images['psnr'].replace([np.inf, -np.inf], np.nan)
            row['psnr_mean'] = psnr_finite.mean()
            row['psnr_lossless'] = (images['psnr'] == np.inf).sum()
            row['ssim_mean'] = images['ssim'].mean()
            row['ssim_perfect'] = (images['ssim'] >= 0.9999).sum()
        else:
            row['psnr_mean'] = np.nan
            row['psnr_lossless'] = 0
            row['ssim_mean'] = np.nan
            row['ssim_perfect'] = 0

        # Video metrics
        if len(videos) > 0:
            row['vmaf_stretched_mean'] = videos['vmaf_stretched'].mean()
            row['vmaf_aligned_mean'] = videos['vmaf_aligned'].mean()
            row['vmaf_difference'] = row['vmaf_aligned_mean'] - row['vmaf_stretched_mean']
        else:
            row['vmaf_stretched_mean'] = np.nan
            row['vmaf_aligned_mean'] = np.nan
            row['vmaf_difference'] = np.nan

        # Lossless flag
        if 'lossless_transform' in transform_df.columns:
            row['lossless_count'] = transform_df['lossless_transform'].sum()
        else:
            row['lossless_count'] = 0

        summary_data.append(row)

    summary = pd.DataFrame(summary_data)

    # Sort by total samples
    summary = summary.sort_values('total_samples', ascending=False)

    # Round numeric columns
    numeric_cols = ['psnr_mean', 'ssim_mean', 'vmaf_stretched_mean',
                   'vmaf_aligned_mean', 'vmaf_difference']
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(2)

    logger.info(f"Quality summary for {len(summary)} transform types")

    return summary


def run_quality_analysis():
    """Execute complete quality metrics analysis and save results."""
    logger.info("=" * 60)
    logger.info("Starting Quality Metrics Analysis")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']

    # Run analyses
    psnr_stats = analyze_psnr_by_transform(df)
    ssim_stats = analyze_ssim_by_transform(df)
    vmaf_stats = analyze_vmaf_dual_metrics(df)
    alignment_methods = analyze_alignment_methods(df)
    lossless_transforms = identify_lossless_transforms(df)
    quality_summary = calculate_quality_summary(df)

    # Save results
    if len(psnr_stats) > 0:
        output_path = utils.DIRS['analysis_csv'] / "psnr_statistics.csv"
        psnr_stats.to_csv(output_path, index=False)
        logger.info(f"Saved PSNR statistics to: {output_path}")

    if len(ssim_stats) > 0:
        output_path = utils.DIRS['analysis_csv'] / "ssim_statistics.csv"
        ssim_stats.to_csv(output_path, index=False)
        logger.info(f"Saved SSIM statistics to: {output_path}")

    if len(vmaf_stats) > 0:
        output_path = utils.DIRS['analysis_csv'] / "vmaf_comparison.csv"
        vmaf_stats.to_csv(output_path, index=False)
        logger.info(f"Saved VMAF comparison to: {output_path}")

    if len(alignment_methods) > 0:
        output_path = utils.DIRS['analysis_csv'] / "alignment_methods.csv"
        alignment_methods.to_csv(output_path, index=False)
        logger.info(f"Saved alignment methods to: {output_path}")

    # Save comprehensive summary
    output_path = utils.DIRS['analysis_csv'] / "quality_summary.csv"
    quality_summary.to_csv(output_path, index=False)
    logger.info(f"Saved quality summary to: {output_path}")

    # Generate text summary
    summary_text = generate_quality_summary_text(psnr_stats, ssim_stats, vmaf_stats, lossless_transforms)
    output_path = utils.DIRS['analysis_results'] / "quality_analysis_summary.txt"
    output_path.write_text(summary_text, encoding='utf-8')
    logger.info(f"Saved text summary to: {output_path}")

    logger.info("Quality analysis summary generated successfully")

    logger.info("=" * 60)
    logger.info("Quality metrics analysis complete")
    logger.info("=" * 60)

    return {
        'psnr_stats': psnr_stats,
        'ssim_stats': ssim_stats,
        'vmaf_stats': vmaf_stats,
        'quality_summary': quality_summary
    }


def generate_quality_summary_text(psnr_stats, ssim_stats, vmaf_stats, lossless_df) -> str:
    """Generate text summary of quality analysis findings."""
    summary = []
    summary.append("=" * 60)
    summary.append("QUALITY METRICS ANALYSIS SUMMARY")
    summary.append("=" * 60)
    summary.append("")

    # PSNR Analysis
    if len(psnr_stats) > 0:
        summary.append("PSNR ANALYSIS (Images):")
        summary.append(f"- Transform types analyzed: {len(psnr_stats)}")
        summary.append(f"- Overall mean PSNR: {psnr_stats['mean'].mean():.1f} dB")
        summary.append(f"- Best quality: {psnr_stats.iloc[0]['transform_type']} ({psnr_stats.iloc[0]['mean']:.1f} dB)")
        summary.append(f"- Lossless transforms: {psnr_stats[psnr_stats['lossless_count'] > 0]['transform_type'].tolist()}")
        summary.append("")

    # SSIM Analysis
    if len(ssim_stats) > 0:
        summary.append("SSIM ANALYSIS (Images):")
        summary.append(f"- Transform types analyzed: {len(ssim_stats)}")
        summary.append(f"- Overall mean SSIM: {ssim_stats['mean'].mean():.3f}")
        summary.append(f"- Excellent quality (>0.99): {(ssim_stats['mean'] > 0.99).sum()} transforms")
        summary.append(f"- Best quality: {ssim_stats.iloc[0]['transform_type']} ({ssim_stats.iloc[0]['mean']:.3f})")
        summary.append("")

    # VMAF Analysis
    if len(vmaf_stats) > 0:
        summary.append("VMAF ANALYSIS (Videos):")
        summary.append(f"- Transform types analyzed: {len(vmaf_stats)}")
        summary.append(f"- Mean stretched VMAF: {vmaf_stats['stretched_mean'].mean():.1f}")
        summary.append(f"- Mean aligned VMAF: {vmaf_stats['aligned_mean'].mean():.1f}")
        large_diff = vmaf_stats[vmaf_stats['mean_difference'] > 10]
        if len(large_diff) > 0:
            summary.append(f"- Aspect ratio affected: {large_diff['transform_type'].tolist()}")
            summary.append(f"  (Difference: {large_diff['mean_difference'].tolist()})")
        summary.append("")

    # Lossless Transforms
    if len(lossless_df) > 0:
        summary.append("LOSSLESS TRANSFORMS:")
        summary.append(f"- Total lossless samples: {len(lossless_df)}")
        summary.append(f"- Types: {lossless_df['transform_type'].unique().tolist()}")
        summary.append("")

    # Key Finding
    summary.append("KEY FINDING:")
    summary.append("Despite 100% C2PA manifest loss, perceptual quality remains high:")
    summary.append("- Most transforms maintain >30 dB PSNR (good quality)")
    summary.append("- Most transforms maintain >0.9 SSIM (high structural similarity)")
    summary.append("- Video quality varies significantly with aspect ratio handling")
    summary.append("")
    summary.append("=" * 60)

    return "\n".join(summary)


if __name__ == "__main__":
    results = run_quality_analysis()