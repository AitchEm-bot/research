"""
VMAF Comparison Visualization for Phase 4
=========================================

This module creates visualizations for VMAF metrics comparison.

Plots:
- VMAF stretched boxplot by transform type
- VMAF aligned boxplot by transform type
- VMAF method distribution
- Stretched vs aligned comparison

Usage:
    python scripts/analysis/visualization/plot_vmaf_comparison.py

Output:
    data/results/analysis_results/plots/vmaf_stretched_boxplot.png
    data/results/analysis_results/plots/vmaf_aligned_boxplot.png
    data/results/analysis_results/plots/vmaf_method_distribution.png
    data/results/analysis_results/plots/vmaf_comparison.png
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
COLORS = sns.color_palette("viridis", 12)


def plot_vmaf_stretched_boxplot(df: pd.DataFrame, output_path: Path):
    """
    Create VMAF stretched boxplot by transform type for videos.

    Args:
        df: DataFrame with VMAF stretched values
        output_path: Path to save the plot
    """
    logger.info(f"Creating VMAF stretched boxplot: {output_path}")

    # Filter to videos only
    df_videos = df[df['asset_type'] == 'video'].copy()

    if len(df_videos) == 0:
        logger.warning("No video data available for VMAF plotting")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort transforms by median VMAF
    transform_order = df_videos.groupby('transform_type')['vmaf_stretched'].median().sort_values(ascending=False).index

    # Create boxplot
    sns.boxplot(data=df_videos, x='transform_type', y='vmaf_stretched',
                order=transform_order, palette=COLORS, ax=ax)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Customize plot
    ax.set_xlabel('Transform Type', fontsize=14)
    ax.set_ylabel('VMAF Score', fontsize=14)
    ax.set_title('VMAF Scores by Transform Type (Stretched Method)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)

    # Add reference lines
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent quality (>90)')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good quality (>70)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Fair quality (>50)')

    # Add legend
    ax.legend(loc='lower left')

    # Add sample counts
    for i, transform in enumerate(transform_order):
        count = len(df_videos[df_videos['transform_type'] == transform])
        ax.text(i, -5, f'n={count}', ha='center', va='top', fontsize=8)

    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"VMAF stretched boxplot saved: {output_path}")


def plot_vmaf_aligned_boxplot(df: pd.DataFrame, output_path: Path):
    """
    Create VMAF aligned boxplot by transform type for videos.

    Args:
        df: DataFrame with VMAF aligned values
        output_path: Path to save the plot
    """
    logger.info(f"Creating VMAF aligned boxplot: {output_path}")

    # Filter to videos only
    df_videos = df[df['asset_type'] == 'video'].copy()

    if len(df_videos) == 0:
        logger.warning("No video data available for VMAF plotting")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    # Sort transforms by median VMAF
    transform_order = df_videos.groupby('transform_type')['vmaf_aligned'].median().sort_values(ascending=False).index

    # Create boxplot
    sns.boxplot(data=df_videos, x='transform_type', y='vmaf_aligned',
                order=transform_order, palette=COLORS, ax=ax)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Customize plot
    ax.set_xlabel('Transform Type', fontsize=14)
    ax.set_ylabel('VMAF Score', fontsize=14)
    ax.set_title('VMAF Scores by Transform Type (Aligned Method)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 105)

    # Add reference lines
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent quality (>90)')
    ax.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good quality (>70)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Fair quality (>50)')

    # Add legend
    ax.legend(loc='lower left')

    # Add sample counts
    for i, transform in enumerate(transform_order):
        count = len(df_videos[df_videos['transform_type'] == transform])
        ax.text(i, -5, f'n={count}', ha='center', va='top', fontsize=8)

    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"VMAF aligned boxplot saved: {output_path}")


def plot_vmaf_method_distribution(df: pd.DataFrame, output_path: Path):
    """
    Create VMAF alignment method distribution plot.

    Args:
        df: DataFrame with vmaf_method column
        output_path: Path to save the plot
    """
    logger.info(f"Creating VMAF method distribution plot: {output_path}")

    # Filter to videos only
    df_videos = df[df['asset_type'] == 'video'].copy()

    if 'vmaf_method' not in df_videos.columns:
        logger.warning("No vmaf_method column found")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Overall distribution
    method_counts = df_videos['vmaf_method'].value_counts()
    colors = sns.color_palette("Set2", len(method_counts))

    wedges, texts, autotexts = ax1.pie(method_counts.values, labels=method_counts.index,
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('VMAF Alignment Method Distribution', fontsize=14, fontweight='bold')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Plot 2: Method by transform type
    method_by_transform = df_videos.groupby(['transform_type', 'vmaf_method']).size().unstack(fill_value=0)

    # Create stacked bar chart
    method_by_transform.plot(kind='bar', stacked=True, ax=ax2, color=colors)
    ax2.set_xlabel('Transform Type', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Alignment Method by Transform Type', fontsize=14, fontweight='bold')
    ax2.legend(title='Alignment Method', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Rotate x-axis labels
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"VMAF method distribution plot saved: {output_path}")


def plot_vmaf_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create side-by-side comparison of stretched vs aligned VMAF.

    Args:
        df: DataFrame with both VMAF metrics
        output_path: Path to save the plot
    """
    logger.info(f"Creating VMAF comparison plot: {output_path}")

    # Filter to videos only
    df_videos = df[df['asset_type'] == 'video'].copy()

    if len(df_videos) == 0:
        logger.warning("No video data available for VMAF plotting")
        return

    # Calculate differences
    df_videos['vmaf_difference'] = df_videos['vmaf_aligned'] - df_videos['vmaf_stretched']

    # Group by transform type
    vmaf_summary = df_videos.groupby('transform_type').agg({
        'vmaf_stretched': 'mean',
        'vmaf_aligned': 'mean',
        'vmaf_difference': 'mean'
    }).reset_index()

    # Sort by difference
    vmaf_summary = vmaf_summary.sort_values('vmaf_difference', ascending=False)

    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Side-by-side bar comparison
    x = np.arange(len(vmaf_summary))
    width = 0.35

    bars1 = ax1.bar(x - width/2, vmaf_summary['vmaf_stretched'], width,
                    label='Stretched', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, vmaf_summary['vmaf_aligned'], width,
                    label='Aligned', color='coral', alpha=0.8)

    ax1.set_xlabel('Transform Type', fontsize=12)
    ax1.set_ylabel('Mean VMAF Score', fontsize=12)
    ax1.set_title('VMAF Scores: Stretched vs Aligned Methods', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(vmaf_summary['transform_type'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Plot 2: Difference plot
    colors = ['green' if d > 0 else 'red' for d in vmaf_summary['vmaf_difference']]
    bars3 = ax2.bar(x, vmaf_summary['vmaf_difference'], color=colors, alpha=0.7)
    ax2.set_xlabel('Transform Type', fontsize=12)
    ax2.set_ylabel('VMAF Difference (Aligned - Stretched)', fontsize=12)
    ax2.set_title('VMAF Score Difference by Transform Type', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(vmaf_summary['transform_type'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, axis='y', alpha=0.3)

    # Plot 3: Scatter plot (Stretched vs Aligned)
    ax3.scatter(df_videos['vmaf_stretched'], df_videos['vmaf_aligned'],
               alpha=0.5, s=20)
    ax3.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Equal scores')
    ax3.set_xlabel('VMAF Stretched', fontsize=12)
    ax3.set_ylabel('VMAF Aligned', fontsize=12)
    ax3.set_title('VMAF Score Correlation', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 105)
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distribution of differences
    ax4.hist(df_videos['vmaf_difference'], bins=30, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('VMAF Difference (Aligned - Stretched)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of VMAF Score Differences', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=df_videos['vmaf_difference'].mean(), color='blue', linestyle='--',
               alpha=0.5, label=f'Mean: {df_videos["vmaf_difference"].mean():.1f}')
    ax4.legend()
    ax4.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"VMAF comparison plot saved: {output_path}")


def create_all_vmaf_plots():
    """Generate all VMAF visualizations."""
    logger.info("=" * 60)
    logger.info("Creating VMAF Visualizations")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']

    # Create plots
    output_path = utils.DIRS['analysis_plots'] / "vmaf_stretched_boxplot.png"
    plot_vmaf_stretched_boxplot(df, output_path)

    output_path = utils.DIRS['analysis_plots'] / "vmaf_aligned_boxplot.png"
    plot_vmaf_aligned_boxplot(df, output_path)

    output_path = utils.DIRS['analysis_plots'] / "vmaf_method_distribution.png"
    plot_vmaf_method_distribution(df, output_path)

    output_path = utils.DIRS['analysis_plots'] / "vmaf_comparison.png"
    plot_vmaf_comparison(df, output_path)

    logger.info("=" * 60)
    logger.info("VMAF visualizations complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_all_vmaf_plots()