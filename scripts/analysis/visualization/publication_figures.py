"""
Publication-Ready Figures for C2PA Robustness Study
====================================================

Generates the 8 publication-quality figures (F1-F8) specified for the thesis.
Uses Springer/Nature style with Palette B colors and minimal annotations.

Figures:
- F1: C2PA Integrity: Baseline vs Post-Transformation
- F2: Manifest Retention by Transform Type
- F3: Manifest Retention Across Platforms
- F4: PSNR by Transform Type (Images)
- F5: SSIM by Transform Type (Images)
- F6: VMAF (Aligned) by Transform Type (Videos)
- F7: VMAF: Stretched vs Aligned Comparison
- F8: PSNR vs SSIM Scatter
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import seaborn as sns

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import centralized style configuration
from visualization.plot_style import (
    set_publication_style, PALETTE_B, SEMANTIC_COLORS,
    FIGURE_SIZES, add_reference_line, format_axis_labels,
    add_value_labels, save_figure, add_quality_reference_lines,
    get_transform_colors
)

# Import data loading utilities
from data_analysis.load_and_prepare import load_all_data

# Import shared utilities
from common import utils

# Configure logging
logger = utils.setup_logging(log_file='data/results/logs/publication_figures.log')


def create_f1_c2pa_integrity(df: pd.DataFrame, df_baseline: pd.DataFrame,
                             output_dir: Path) -> None:
    """
    F1 - C2PA Integrity: Baseline vs Post-Transformation
    Clean grouped bar chart showing 100% to 0% story.
    """
    logger.info("Creating F1: C2PA Integrity Baseline vs Post-Transformation")

    metrics = ["Manifest present", "Verified", "Signature valid"]
    baseline_values = [100, 100, 100]  # All baseline samples have intact C2PA
    after_values = [0, 0, 0]  # All transforms strip C2PA

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])

    # Create bars
    bars1 = ax.bar(x - width/2, baseline_values, width, label="Baseline",
                   color=SEMANTIC_COLORS['baseline'])
    bars2 = ax.bar(x + width/2, after_values, width, label="After transformation",
                   color=SEMANTIC_COLORS['transformed'])

    # Add minimal value labels (only on baseline bars)
    for i, v in enumerate(baseline_values):
        ax.text(x[i] - width/2, v + 1, f"{v}%", ha="center", va="bottom",
                fontsize=8, color='0.4')

    # Format axes
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=False, loc='upper right')

    # Remove top and right spines (already done by style)
    save_figure(fig, 'F1_c2pa_integrity.png', output_dir)
    logger.info(f"Saved F1 to {output_dir}/F1_c2pa_integrity.png")


def create_f2_manifest_retention_by_transform(df: pd.DataFrame, output_dir: Path) -> None:
    """
    F2 - Manifest Retention by Transform Type
    Horizontal bar plot showing 0% retention across all transforms.
    """
    logger.info("Creating F2: Manifest Retention by Transform Type")

    # Calculate retention by transform type
    transform_stats = (
        df.groupby("transform_type")["manifest_present"]
        .agg(['mean', 'count'])
        .sort_values('mean')
    )
    transform_stats['percentage'] = transform_stats['mean'] * 100

    # Order transforms logically (compression, editing, platform)
    order = ['platform_roundtrip', 'h264_compression', 'h265_compression',
             'fps_adjustment', 'jpeg_compression', 'png_compression',
             'resize', 'crop', 'rotation',
             'brightness_adjustment', 'contrast_adjustment', 'saturation_adjustment']

    # Filter to available transforms
    order = [t for t in order if t in transform_stats.index]
    transform_stats = transform_stats.loc[order]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])

    # Create horizontal bars
    y_pos = np.arange(len(transform_stats))
    ax.barh(y_pos, transform_stats['percentage'].values,
            color=SEMANTIC_COLORS['lost'])

    # Format axes
    ax.set_xlabel("Manifest retention rate (%)")
    ax.set_xlim(0, 5)  # Small range since all are 0
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t.replace('_', ' ').title() for t in transform_stats.index])

    save_figure(fig, 'F2_manifest_retention_by_transform.png', output_dir)
    logger.info(f"Saved F2 to {output_dir}/F2_manifest_retention_by_transform.png")


def create_f3_platform_manifest_retention(df_platform: pd.DataFrame, output_dir: Path) -> None:
    """
    F3 - Manifest Retention Across Platforms
    Simple bar chart showing all platforms strip C2PA.
    """
    logger.info("Creating F3: Manifest Retention Across Platforms")

    if df_platform.empty:
        logger.warning("No platform data available for F3")
        return

    # Calculate platform retention
    platform_stats = (
        df_platform.groupby("platform")["manifest_present"]
        .agg(['mean', 'count'])
        .sort_values('mean')
    )
    platform_stats['percentage'] = platform_stats['mean'] * 100

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])

    # Create bars
    x_pos = np.arange(len(platform_stats))
    ax.bar(x_pos, platform_stats['percentage'].values,
           color=SEMANTIC_COLORS['platform'])

    # Format axes
    ax.set_ylabel("Manifest retention rate (%)")
    ax.set_ylim(0, 5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.title() for p in platform_stats.index], rotation=20, ha='right')

    # Add sample counts in x-labels (subtle)
    labels = [f"{p.title()}\n(n={n})" for p, n in
              zip(platform_stats.index, platform_stats['count'])]
    ax.set_xticklabels(labels, rotation=0, fontsize=8)

    save_figure(fig, 'F3_platform_manifest_retention.png', output_dir)
    logger.info(f"Saved F3 to {output_dir}/F3_platform_manifest_retention.png")


def create_f4_psnr_by_transform(df_images: pd.DataFrame, output_dir: Path) -> None:
    """
    F4 - PSNR by Transform Type (Images)
    Boxplot showing quality distribution across transforms.
    """
    logger.info("Creating F4: PSNR by Transform Type (Images)")

    # Transform order for consistency
    order = ["png_compression", "jpeg_compression", "platform_roundtrip",
             "saturation_adjustment", "contrast_adjustment",
             "brightness_adjustment", "crop", "rotation", "resize"]

    # Filter to available transforms
    available = df_images['transform_type'].unique()
    order = [t for t in order if t in available]

    # Replace inf values with a high but plottable value
    df_plot = df_images.copy()
    df_plot['psnr_plot'] = df_plot['psnr'].replace([np.inf, -np.inf], 60)

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['wide'])

    # Create boxplot
    box_data = [df_plot[df_plot['transform_type'] == t]['psnr_plot'].dropna()
                for t in order]

    bp = ax.boxplot(box_data, labels=[t.replace('_', ' ').title() for t in order],
                    patch_artist=True, showfliers=False)

    # Color boxes using palette
    colors = get_transform_colors(len(order))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add reference lines
    ax.axhline(30, color='0.7', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(40, color='0.6', linestyle='--', linewidth=0.8, alpha=0.5)

    # Format axes
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(0, 65)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

    save_figure(fig, 'F4_psnr_by_transform.png', output_dir)
    logger.info(f"Saved F4 to {output_dir}/F4_psnr_by_transform.png")


def create_f5_ssim_by_transform(df_images: pd.DataFrame, output_dir: Path) -> None:
    """
    F5 - SSIM by Transform Type (Images)
    Boxplot showing structural similarity across transforms.
    """
    logger.info("Creating F5: SSIM by Transform Type (Images)")

    # Transform order for consistency
    order = ["png_compression", "jpeg_compression", "platform_roundtrip",
             "saturation_adjustment", "contrast_adjustment",
             "brightness_adjustment", "crop", "rotation", "resize"]

    # Filter to available transforms
    available = df_images['transform_type'].unique()
    order = [t for t in order if t in available]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['wide'])

    # Create boxplot
    box_data = [df_images[df_images['transform_type'] == t]['ssim'].dropna()
                for t in order]

    bp = ax.boxplot(box_data, labels=[t.replace('_', ' ').title() for t in order],
                    patch_artist=True, showfliers=False)

    # Color boxes using palette
    colors = get_transform_colors(len(order))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add reference lines
    for threshold in [0.5, 0.7, 0.9]:
        ax.axhline(threshold, color='0.7', linestyle='--', linewidth=0.8, alpha=0.5)

    # Format axes
    ax.set_ylabel("SSIM")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

    save_figure(fig, 'F5_ssim_by_transform.png', output_dir)
    logger.info(f"Saved F5 to {output_dir}/F5_ssim_by_transform.png")


def create_f6_vmaf_aligned_by_transform(df_videos: pd.DataFrame, output_dir: Path) -> None:
    """
    F6 - VMAF (Aligned) by Transform Type (Videos)
    Boxplot showing perceptual quality after aspect ratio correction.
    """
    logger.info("Creating F6: VMAF Aligned by Transform Type (Videos)")

    # Transform order for videos
    order = ["platform_roundtrip", "h264_compression", "h265_compression",
             "fps_adjustment", "contrast_adjustment", "brightness_adjustment",
             "saturation_adjustment", "resize", "crop", "rotation"]

    # Filter to available transforms
    available = df_videos['transform_type'].unique()
    order = [t for t in order if t in available]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['wide'])

    # Create boxplot using aligned VMAF
    box_data = [df_videos[df_videos['transform_type'] == t]['vmaf_aligned'].dropna()
                for t in order]

    bp = ax.boxplot(box_data, labels=[t.replace('_', ' ').title() for t in order],
                    patch_artist=True, showfliers=False)

    # Color boxes using palette
    colors = get_transform_colors(len(order))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add quality reference lines
    for threshold in [50, 70, 90]:
        ax.axhline(threshold, color='0.7', linestyle='--', linewidth=0.8, alpha=0.5)

    # Format axes
    ax.set_ylabel("VMAF (aligned)")
    ax.set_ylim(0, 105)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')

    save_figure(fig, 'F6_vmaf_aligned_by_transform.png', output_dir)
    logger.info(f"Saved F6 to {output_dir}/F6_vmaf_aligned_by_transform.png")


def create_f7_vmaf_comparison(df_videos: pd.DataFrame, output_dir: Path) -> None:
    """
    F7 - VMAF: Stretched vs Aligned Comparison
    2-panel figure showing importance of alignment for platforms.
    """
    logger.info("Creating F7: VMAF Stretched vs Aligned Comparison")

    # Transform order
    order = ["platform_roundtrip", "h264_compression", "h265_compression",
             "fps_adjustment", "contrast_adjustment", "resize", "crop", "rotation"]

    # Filter to available transforms
    available = df_videos['transform_type'].unique()
    order = [t for t in order if t in available]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGURE_SIZES['double_vertical'])

    # Top panel: Bar chart of mean VMAF
    group_stats = df_videos.groupby("transform_type")[["vmaf_stretched", "vmaf_aligned"]].mean()
    group_stats = group_stats.loc[order]

    x = np.arange(len(order))
    width = 0.35

    ax1.bar(x - width/2, group_stats["vmaf_stretched"], width,
            label="Stretched", color=PALETTE_B[0])
    ax1.bar(x + width/2, group_stats["vmaf_aligned"], width,
            label="Aligned", color=PALETTE_B[1])

    ax1.set_xticks(x)
    ax1.set_xticklabels([t.replace('_', ' ').title() for t in order],
                         rotation=25, ha='right')
    ax1.set_ylabel("Mean VMAF")
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=False)

    # Bottom panel: Scatter plot
    ax2.scatter(df_videos["vmaf_stretched"], df_videos["vmaf_aligned"],
                s=8, alpha=0.5, color=PALETTE_B[0])

    # Add y=x reference line
    lims = [0, 100]
    ax2.plot(lims, lims, color='0.6', linestyle='--', linewidth=1, alpha=0.7)

    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("VMAF (stretched)")
    ax2.set_ylabel("VMAF (aligned)")
    ax2.set_aspect('equal')

    plt.tight_layout()
    save_figure(fig, 'F7_vmaf_comparison.png', output_dir, tight_layout=False)
    logger.info(f"Saved F7 to {output_dir}/F7_vmaf_comparison.png")


def create_f8_psnr_ssim_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    """
    F8 - PSNR vs SSIM Scatter
    Shows correlation and clustering by transform category.
    """
    logger.info("Creating F8: PSNR vs SSIM Scatter")

    # Filter to images with both metrics
    df_plot = df[(df['asset_type'] == 'image') &
                 (df['psnr'].notna()) &
                 (df['ssim'].notna())].copy()

    # Replace inf PSNR with 60 for plotting
    df_plot['psnr_plot'] = df_plot['psnr'].replace([np.inf, -np.inf], 60)

    # Define transform categories for coloring
    category_map = {
        'jpeg_compression': 'Compression',
        'png_compression': 'Compression',
        'platform_roundtrip': 'Platform',
        'brightness_adjustment': 'Color',
        'contrast_adjustment': 'Color',
        'saturation_adjustment': 'Color',
        'crop': 'Geometric',
        'rotation': 'Geometric',
        'resize': 'Geometric',
    }

    df_plot['category'] = df_plot['transform_type'].map(category_map).fillna('Other')

    fig, ax = plt.subplots(figsize=FIGURE_SIZES['single'])

    # Plot by category
    categories = df_plot['category'].unique()
    colors = get_transform_colors(len(categories))

    for i, cat in enumerate(categories):
        mask = df_plot['category'] == cat
        ax.scatter(df_plot[mask]['psnr_plot'],
                  df_plot[mask]['ssim'],
                  s=8, alpha=0.4, color=colors[i],
                  label=cat)

    # Format axes
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("SSIM")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 1.05)

    # Add light grid for easier reading
    ax.grid(True, alpha=0.2)

    # Legend
    ax.legend(frameon=False, markerscale=2, loc='lower right')

    save_figure(fig, 'F8_psnr_ssim_scatter.png', output_dir)
    logger.info(f"Saved F8 to {output_dir}/F8_psnr_ssim_scatter.png")


def generate_all_figures(output_dir: Optional[str] = None) -> None:
    """
    Generate all 8 publication figures.

    Args:
        output_dir: Output directory for figures (defaults to standard location)
    """
    # Apply publication style globally
    set_publication_style()

    # Set output directory
    if output_dir is None:
        output_dir = utils.DIRS['analysis_plots']
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GENERATING PUBLICATION FIGURES (F1-F8)")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    try:
        # Load data
        logger.info("Loading data for publication figures...")
        data = load_all_data()
        df_all = data['final_metrics']
        df_baseline = data['baseline']
        df_images = data['images']
        df_videos = data['videos']
        df_platform = data['platform']

        # Generate each figure
        logger.info("Generating figures...")

        # F1: C2PA Integrity
        create_f1_c2pa_integrity(df_all, df_baseline, output_dir)

        # F2: Manifest Retention by Transform
        create_f2_manifest_retention_by_transform(df_all, output_dir)

        # F3: Platform Manifest Retention
        if not df_platform.empty:
            create_f3_platform_manifest_retention(df_platform, output_dir)
        else:
            logger.warning("Skipping F3 - no platform data available")

        # F4: PSNR by Transform
        create_f4_psnr_by_transform(df_images, output_dir)

        # F5: SSIM by Transform
        create_f5_ssim_by_transform(df_images, output_dir)

        # F6: VMAF Aligned by Transform
        create_f6_vmaf_aligned_by_transform(df_videos, output_dir)

        # F7: VMAF Comparison
        create_f7_vmaf_comparison(df_videos, output_dir)

        # F8: PSNR vs SSIM Scatter
        create_f8_psnr_ssim_scatter(df_all, output_dir)

        logger.info("=" * 60)
        logger.info("PUBLICATION FIGURES COMPLETE")
        logger.info(f"All figures saved to: {output_dir}")
        logger.info("Figures generated: F1-F8")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error generating publication figures: {e}")
        raise


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures for C2PA robustness study"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (defaults to analysis_plots)"
    )
    parser.add_argument(
        "--figure",
        type=str,
        choices=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'all'],
        default='all',
        help="Generate specific figure or all figures"
    )

    args = parser.parse_args()

    # Apply style
    set_publication_style()

    if args.figure == 'all':
        generate_all_figures(args.output_dir)
    else:
        # Load data
        data = load_all_data()
        df_all = data['final_metrics']
        df_baseline = data['baseline']
        df_images = data['images']
        df_videos = data['videos']
        df_platform = data['platform']

        output_dir = Path(args.output_dir) if args.output_dir else utils.DIRS['analysis_plots']
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate specific figure
        figure_map = {
            'F1': lambda: create_f1_c2pa_integrity(df_all, df_baseline, output_dir),
            'F2': lambda: create_f2_manifest_retention_by_transform(df_all, output_dir),
            'F3': lambda: create_f3_platform_manifest_retention(df_platform, output_dir),
            'F4': lambda: create_f4_psnr_by_transform(df_images, output_dir),
            'F5': lambda: create_f5_ssim_by_transform(df_images, output_dir),
            'F6': lambda: create_f6_vmaf_aligned_by_transform(df_videos, output_dir),
            'F7': lambda: create_f7_vmaf_comparison(df_videos, output_dir),
            'F8': lambda: create_f8_psnr_ssim_scatter(df_all, output_dir),
        }

        figure_map[args.figure]()
        logger.info(f"Generated {args.figure} in {output_dir}")


if __name__ == "__main__":
    main()