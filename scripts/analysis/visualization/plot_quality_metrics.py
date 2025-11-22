"""
Quality Metrics Visualization for Phase 4
=========================================

This module creates visualizations for perceptual quality metrics.

Plots:
- PSNR boxplot by transform type
- SSIM boxplot by transform type
- Combined quality metrics comparison

Usage:
    python scripts/analysis/visualization/plot_quality_metrics.py

Output:
    data/results/analysis_results/plots/psnr_boxplot.png
    data/results/analysis_results/plots/ssim_boxplot.png
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
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Color palette
COLORS = sns.color_palette("husl", 12)


def plot_psnr_boxplot(df: pd.DataFrame, output_path: Path):
    """
    Create PSNR boxplot by transform type for images.

    Args:
        df: DataFrame with PSNR values and transform types
        output_path: Path to save the plot
    """
    logger.info(f"Creating PSNR boxplot: {output_path}")

    # Filter to images only and drop NaN values
    df_images = df[df['asset_type'] == 'image'].copy()
    df_images = df_images.dropna(subset=['psnr'])

    if len(df_images) == 0:
        logger.warning("No image data available for PSNR plotting")
        return

    # Handle infinite values (lossless transforms)
    df_images['psnr_plot'] = df_images['psnr'].replace([np.inf], 100)  # Cap at 100 for visualization

    # Count lossless transforms
    lossless_counts = df_images.groupby('transform_type').apply(
        lambda x: (x['psnr'] == np.inf).sum()
    ).to_dict()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort transforms by median PSNR (only includes categories with valid data)
    transform_order = df_images.groupby('transform_type')['psnr_plot'].median().sort_values(ascending=False).index.tolist()

    # Create boxplot
    sns.boxplot(data=df_images, x='transform_type', y='psnr_plot',
                order=transform_order, ax=ax)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add lossless annotations
    for i, transform in enumerate(transform_order):
        if transform in lossless_counts and lossless_counts[transform] > 0:
            ax.text(i, 102, f'Lossless\n({lossless_counts[transform]})',
                   ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

    # Customize plot
    ax.set_xlabel('Transform Type', fontsize=14)
    ax.set_ylabel('PSNR (dB)', fontsize=14)
    ax.set_title('Peak Signal-to-Noise Ratio by Transform Type (Images)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 110)

    # Add reference lines
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Good quality (30 dB)')
    ax.axhline(y=40, color='green', linestyle='--', alpha=0.5, label='Excellent quality (40 dB)')

    # Add legend
    ax.legend(loc='lower left')

    # Add sample counts
    for i, transform in enumerate(transform_order):
        count = len(df_images[df_images['transform_type'] == transform])
        ax.text(i, -5, f'n={count}', ha='center', va='top', fontsize=8)

    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"PSNR boxplot saved: {output_path}")


def plot_ssim_boxplot(df: pd.DataFrame, output_path: Path):
    """
    Create SSIM boxplot by transform type for images.

    Args:
        df: DataFrame with SSIM values and transform types
        output_path: Path to save the plot
    """
    logger.info(f"Creating SSIM boxplot: {output_path}")

    # Filter to images only and drop NaN values
    df_images = df[df['asset_type'] == 'image'].copy()
    df_images = df_images.dropna(subset=['ssim'])

    if len(df_images) == 0:
        logger.warning("No image data available for SSIM plotting")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort transforms by median SSIM (only includes categories with valid data)
    transform_order = df_images.groupby('transform_type')['ssim'].median().sort_values(ascending=False).index.tolist()

    # Create boxplot
    sns.boxplot(data=df_images, x='transform_type', y='ssim',
                order=transform_order, ax=ax)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Customize plot
    ax.set_xlabel('Transform Type', fontsize=14)
    ax.set_ylabel('SSIM', fontsize=14)
    ax.set_title('Structural Similarity Index by Transform Type (Images)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.05)

    # Add reference lines
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent quality (>0.9)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good quality (>0.7)')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Fair quality (>0.5)')

    # Add legend
    ax.legend(loc='lower left')

    # Add sample counts and perfect scores
    for i, transform in enumerate(transform_order):
        transform_data = df_images[df_images['transform_type'] == transform]['ssim']
        count = len(transform_data)
        perfect = (transform_data >= 0.999).sum()

        ax.text(i, -0.05, f'n={count}', ha='center', va='top', fontsize=8)
        if perfect > 0:
            ax.text(i, 1.02, f'Perfect: {perfect}', ha='center', va='bottom',
                   fontsize=9, color='green', fontweight='bold')

    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"SSIM boxplot saved: {output_path}")


def plot_quality_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create combined quality metrics comparison plot.

    Args:
        df: DataFrame with quality metrics
        output_path: Path to save the plot
    """
    logger.info(f"Creating quality comparison plot: {output_path}")

    # Filter to images only
    df_images = df[df['asset_type'] == 'image'].copy()

    # Handle infinite PSNR
    df_images['psnr_norm'] = df_images['psnr'].replace([np.inf], 50).clip(0, 50) / 50  # Normalize to 0-1

    # Calculate means by transform type
    quality_summary = df_images.groupby('transform_type').agg({
        'psnr_norm': 'mean',
        'ssim': 'mean'
    }).reset_index()

    # Sort by average quality
    quality_summary['avg_quality'] = (quality_summary['psnr_norm'] + quality_summary['ssim']) / 2
    quality_summary = quality_summary.sort_values('avg_quality', ascending=False)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Bar comparison
    x = np.arange(len(quality_summary))
    width = 0.35

    bars1 = ax1.bar(x - width/2, quality_summary['psnr_norm'], width,
                    label='PSNR (normalized)', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, quality_summary['ssim'], width,
                    label='SSIM', color='coral', alpha=0.8)

    ax1.set_xlabel('Transform Type', fontsize=12)
    ax1.set_ylabel('Quality Score (0-1)', fontsize=12)
    ax1.set_title('Quality Metrics Comparison by Transform Type', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(quality_summary['transform_type'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Scatter plot (PSNR vs SSIM)
    for transform in quality_summary['transform_type'].unique():
        transform_data = df_images[df_images['transform_type'] == transform]
        # Handle infinite PSNR for display
        psnr_display = transform_data['psnr'].replace([np.inf], 60).clip(0, 60)
        ax2.scatter(psnr_display, transform_data['ssim'],
                   label=transform, alpha=0.6, s=30)

    ax2.set_xlabel('PSNR (dB)', fontsize=12)
    ax2.set_ylabel('SSIM', fontsize=12)
    ax2.set_title('PSNR vs SSIM Correlation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 65)
    ax2.set_ylim(0, 1.05)

    # Add reference lines
    ax2.axvline(x=30, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3)

    # Legend outside plot
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Quality comparison plot saved: {output_path}")


def create_all_quality_plots():
    """Generate all quality metrics visualizations."""
    logger.info("=" * 60)
    logger.info("Creating Quality Metrics Visualizations")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']

    # Create plots
    output_path = utils.DIRS['analysis_plots'] / "psnr_boxplot.png"
    plot_psnr_boxplot(df, output_path)

    output_path = utils.DIRS['analysis_plots'] / "ssim_boxplot.png"
    plot_ssim_boxplot(df, output_path)

    output_path = utils.DIRS['analysis_plots'] / "quality_comparison.png"
    plot_quality_comparison(df, output_path)

    logger.info("=" * 60)
    logger.info("Quality metrics visualizations complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_all_quality_plots()