"""
Manifest Retention Visualization for Phase 4
============================================

This module creates visualizations for C2PA manifest retention analysis.

Plots:
- Bar chart of manifest retention by transform type
- Grouped bar chart for baseline vs transformed
- Platform-specific retention chart

Usage:
    python scripts/analysis/visualization/plot_manifest_retention.py

Output:
    data/results/analysis_results/plots/manifest_retention.png
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
from scripts.analysis.data_analysis import load_and_prepare, manifest_retention_analysis

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_visualization.log')

# Setup publication-quality plot defaults
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (10, 6),
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

# Set color palette
COLORS = {
    'retained': '#2ecc71',  # Green
    'lost': '#e74c3c',      # Red
    'baseline': '#3498db',   # Blue
    'transformed': '#e74c3c', # Red
    'platform': '#9b59b6'    # Purple
}


def plot_manifest_retention_by_transform(retention_df: pd.DataFrame, output_path: Path):
    """
    Create bar chart showing manifest retention percentage by transform type.

    All bars expected to be at 0% (complete manifest stripping).

    Args:
        retention_df: DataFrame with transform_type, total_samples, retention_pct
        output_path: Path to save the plot
    """
    logger.info(f"Creating manifest retention plot: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by sample count for better visibility
    retention_df = retention_df.sort_values('total_samples', ascending=True)

    # Create bar chart
    bars = ax.barh(retention_df['transform_type'],
                   retention_df['retention_pct'],
                   color=COLORS['lost'],
                   edgecolor='black',
                   linewidth=0.5)

    # Add value labels on bars
    for i, (bar, samples) in enumerate(zip(bars, retention_df['total_samples'])):
        width = bar.get_width()
        # Show retention percentage and sample count
        label = f'{width:.1f}% (n={samples})'
        ax.text(1, bar.get_y() + bar.get_height()/2,
               label,
               ha='left', va='center', fontsize=9)

    # Customize plot
    ax.set_xlabel('Manifest Retention Rate (%)', fontsize=14)
    ax.set_ylabel('Transform Type', fontsize=14)
    ax.set_title('C2PA Manifest Retention After Transformations', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 100)

    # Add reference line at 100%
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Expected (baseline)')

    # Add annotation for critical finding
    if retention_df['retention_pct'].max() == 0:
        ax.text(50, ax.get_ylim()[1] * 0.9,
               'WARNING: 100% Manifest Loss\nAcross All Transforms',
               ha='center', fontsize=14, fontweight='bold',
               color=COLORS['lost'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['lost']))

    # Grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Manifest retention plot saved: {output_path}")


def plot_baseline_vs_transformed(baseline_stats: dict, transformed_stats: dict, output_path: Path):
    """
    Create grouped bar chart comparing baseline vs transformed retention.

    Args:
        baseline_stats: Dictionary with baseline statistics
        transformed_stats: Dictionary with transformed statistics
        output_path: Path to save the plot
    """
    logger.info(f"Creating baseline vs transformed plot: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Manifest Present', 'Verified', 'Signature Valid']
    baseline_values = [
        baseline_stats.get('manifest_present', 100),
        baseline_stats.get('verified', 100),
        baseline_stats.get('signature_valid', 100)
    ]
    transformed_values = [
        transformed_stats.get('manifest_present', 0),
        transformed_stats.get('verified', 0),
        transformed_stats.get('signature_valid', 0)
    ]

    x = np.arange(len(categories))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline',
                   color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, transformed_values, width, label='After Transformation',
                   color=COLORS['transformed'], edgecolor='black', linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%',
                   ha='center', va='bottom', fontsize=10)

    # Customize plot
    ax.set_ylabel('Percentage (%)', fontsize=14)
    ax.set_title('C2PA Integrity: Baseline vs Post-Transformation', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    # Add annotation for drop
    drop_pct = baseline_values[0] - transformed_values[0]
    ax.annotate(f'{drop_pct:.0f}% Drop',
               xy=(0, transformed_values[0]),
               xytext=(0, 50),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, fontweight='bold', color='red',
               ha='center')

    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Baseline vs transformed plot saved: {output_path}")


def plot_platform_retention(platform_df: pd.DataFrame, output_path: Path):
    """
    Create bar chart for platform-specific manifest retention.

    Args:
        platform_df: DataFrame with platform retention data
        output_path: Path to save the plot
    """
    if len(platform_df) == 0:
        logger.warning("No platform data available for plotting")
        return

    logger.info(f"Creating platform retention plot: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort platforms alphabetically
    platform_df = platform_df.sort_values('platform')

    # Create bar chart
    bars = ax.bar(platform_df['platform'],
                  platform_df['retention_pct'],
                  color=COLORS['platform'],
                  edgecolor='black',
                  linewidth=0.5)

    # Add value labels and sample counts
    for bar, samples in zip(bars, platform_df['total_samples']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.0f}%\n(n={samples})',
               ha='center', va='bottom', fontsize=10)

    # Customize plot
    ax.set_xlabel('Platform', fontsize=14)
    ax.set_ylabel('Manifest Retention Rate (%)', fontsize=14)
    ax.set_title('C2PA Manifest Retention Across Social Media Platforms', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 10)  # Set to 10% since all are 0%

    # Rotate x labels if needed
    plt.xticks(rotation=45, ha='right')

    # Add annotation
    if platform_df['retention_pct'].max() == 0:
        ax.text(len(platform_df)/2 - 0.5, 5,
               'All Platforms Strip C2PA Metadata',
               ha='center', fontsize=14, fontweight='bold',
               color=COLORS['lost'],
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['lost']))

    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Platform retention plot saved: {output_path}")


def plot_retention_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Create heatmap showing retention across transform types and levels.

    Args:
        df: DataFrame with detailed retention data
        output_path: Path to save the plot
    """
    logger.info(f"Creating retention heatmap: {output_path}")

    # Create pivot table
    pivot = df.pivot_table(
        values='retention_pct',
        index='transform_type',
        columns='transform_level',
        aggfunc='mean',
        fill_value=0
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use diverging colormap centered at 0
    sns.heatmap(pivot,
                annot=True,
                fmt='.0f',
                cmap='RdYlGn',
                vmin=0, vmax=100,
                cbar_kws={'label': 'Retention (%)'},
                linewidths=0.5,
                linecolor='gray',
                ax=ax)

    ax.set_title('C2PA Manifest Retention Heatmap by Transform Type and Level',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Transform Level', fontsize=14)
    ax.set_ylabel('Transform Type', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Retention heatmap saved: {output_path}")


def create_all_retention_plots():
    """Generate all manifest retention visualizations."""
    logger.info("=" * 60)
    logger.info("Creating Manifest Retention Visualizations")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']
    df_platform = data['platform']

    # Calculate retention metrics
    retention_by_type = manifest_retention_analysis.calculate_retention_by_transform(df)
    retention_by_level = manifest_retention_analysis.calculate_retention_by_level(df)

    # Platform retention
    if len(df_platform) > 0:
        platform_retention = manifest_retention_analysis.analyze_platform_retention(df_platform)
    else:
        platform_retention = pd.DataFrame()

    # Get baseline comparison
    df_baseline = data['baseline']
    comparison = manifest_retention_analysis.compare_baseline_vs_transformed(df_baseline, df)

    # Create plots
    # 1. Main retention plot
    output_path = utils.DIRS['analysis_plots'] / "manifest_retention.png"
    plot_manifest_retention_by_transform(retention_by_type, output_path)

    # 2. Baseline vs transformed
    output_path = utils.DIRS['analysis_plots'] / "baseline_vs_transformed.png"
    baseline_stats = {
        'manifest_present': comparison.get('baseline_manifest_present', 1.0) * 100,
        'verified': comparison.get('baseline_verified', 1.0) * 100,
        'signature_valid': comparison.get('baseline_signature_valid', 1.0) * 100
    }
    transformed_stats = {
        'manifest_present': comparison.get('transformed_manifest_present', 0) * 100,
        'verified': comparison.get('transformed_verified', 0) * 100,
        'signature_valid': comparison.get('transformed_signature_valid', 0) * 100
    }
    plot_baseline_vs_transformed(baseline_stats, transformed_stats, output_path)

    # 3. Platform retention
    if len(platform_retention) > 0:
        output_path = utils.DIRS['analysis_plots'] / "platform_retention.png"
        plot_platform_retention(platform_retention, output_path)

    # 4. Retention heatmap
    if len(retention_by_level) > 0:
        output_path = utils.DIRS['analysis_plots'] / "retention_heatmap.png"
        plot_retention_heatmap(retention_by_level, output_path)

    logger.info("=" * 60)
    logger.info("Manifest retention visualizations complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_all_retention_plots()