"""
Correlation Heatmap Visualization for Phase 4
=============================================

This module creates correlation heatmap visualizations.

Usage:
    python scripts/analysis/visualization/plot_correlation_heatmap.py

Output:
    data/results/analysis_results/plots/correlation_heatmap.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.common import utils
from scripts.analysis.data_analysis import load_and_prepare

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_visualization.log')

# Setup publication-quality plot defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (10, 8),
    'font.size': 12
})


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """Create correlation heatmap of key metrics."""
    logger.info(f"Creating correlation heatmap: {output_path}")

    # Select relevant columns for correlation
    cols_manifest = ['manifest_present', 'verified', 'signature_valid', 'hash_match']
    cols_quality_img = ['psnr', 'ssim']
    cols_quality_vid = ['vmaf_stretched', 'vmaf_aligned']

    # Create subplots for different asset types
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Overall correlation (limited metrics due to mixed types)
    overall_cols = ['manifest_present', 'verified', 'lossless_match']
    corr_overall = df[overall_cols].corr()
    sns.heatmap(corr_overall, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[0, 0], vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'})
    axes[0, 0].set_title('Overall Correlation (All Samples)', fontweight='bold')

    # Image correlation
    df_img = df[df['asset_type'] == 'image']
    img_cols = cols_manifest + cols_quality_img
    img_cols = [col for col in img_cols if col in df_img.columns]

    # Handle infinite PSNR
    df_img_clean = df_img[img_cols].copy()
    if 'psnr' in df_img_clean.columns:
        df_img_clean['psnr'] = df_img_clean['psnr'].replace([np.inf, -np.inf], np.nan)

    corr_img = df_img_clean.corr()
    sns.heatmap(corr_img, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[0, 1], vmin=-1, vmax=1)
    axes[0, 1].set_title('Image Metrics Correlation', fontweight='bold')

    # Video correlation
    df_vid = df[df['asset_type'] == 'video']
    vid_cols = cols_manifest + cols_quality_vid
    vid_cols = [col for col in vid_cols if col in df_vid.columns]
    corr_vid = df_vid[vid_cols].corr()
    sns.heatmap(corr_vid, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[1, 0], vmin=-1, vmax=1)
    axes[1, 0].set_title('Video Metrics Correlation', fontweight='bold')

    # Transform type analysis
    transform_summary = df.groupby('transform_type').agg({
        'manifest_present': 'mean',
        'verified': 'mean'
    })

    # Add quality metrics with proper handling
    df_img_grp = df[df['asset_type'] == 'image'].copy()
    df_img_grp['psnr_clean'] = df_img_grp['psnr'].replace([np.inf], np.nan)
    img_quality = df_img_grp.groupby('transform_type').agg({
        'psnr_clean': 'mean',
        'ssim': 'mean'
    })

    vid_quality = df[df['asset_type'] == 'video'].groupby('transform_type').agg({
        'vmaf_aligned': 'mean'
    })

    # Combine summaries
    transform_combined = transform_summary.join(img_quality, how='outer').join(vid_quality, how='outer')
    transform_corr = transform_combined.corr()

    sns.heatmap(transform_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=axes[1, 1], vmin=-1, vmax=1)
    axes[1, 1].set_title('Transform-Level Metrics Correlation', fontweight='bold')

    # Add main title
    fig.suptitle('Correlation Analysis: C2PA Integrity vs Quality Metrics',
                 fontsize=16, fontweight='bold', y=1.02)

    # Add note about constant values
    fig.text(0.5, -0.02,
             'Note: Manifest metrics show no variance (all 0), resulting in undefined correlations',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Correlation heatmap saved: {output_path}")


def create_correlation_plots():
    """Generate correlation visualizations."""
    logger.info("Creating Correlation Visualizations")

    data = load_and_prepare.load_all_data()
    df = data['final_metrics']

    output_path = utils.DIRS['analysis_plots'] / "correlation_heatmap.png"
    plot_correlation_heatmap(df, output_path)

    logger.info("Correlation visualizations complete")


if __name__ == "__main__":
    create_correlation_plots()