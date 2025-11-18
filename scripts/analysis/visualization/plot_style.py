"""
Publication-Ready Plot Style Configuration
==========================================

Centralized style configuration for Springer/Nature academic publication figures.
Implements Palette B (Tableau-ish muted scientific colors) with minimal annotations.

This module provides:
- Global matplotlib rcParams for consistent styling
- Palette B color scheme in order of preference
- Utility functions for common plot elements
- Style presets for different figure types
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# =============================================================================
# PALETTE B: Tableau-ish Muted Scientific Colors
# =============================================================================
# Use in this order for consistency across all figures
PALETTE_B = [
    "#4E79A7",  # Blue - primary/baseline
    "#F28E2C",  # Orange - secondary/after transformation
    "#59A14F",  # Green - success/good quality
    "#E15759",  # Red - failure/poor quality
    "#76B7B2",  # Teal - alternative 1
    "#EDC948",  # Yellow - alternative 2
    "#B07AA1",  # Purple - alternative 3
    "#FF9DA7",  # Pink - alternative 4
    "#9C755F",  # Brown - alternative 5
    "#BAB0AC",  # Gray - neutral/reference
]

# Semantic color mappings for specific use cases
SEMANTIC_COLORS = {
    'baseline': PALETTE_B[0],      # Blue
    'transformed': PALETTE_B[1],    # Orange
    'retained': PALETTE_B[2],       # Green
    'lost': PALETTE_B[3],          # Red
    'platform': PALETTE_B[6],      # Purple
    'reference': PALETTE_B[9],     # Gray
}

# =============================================================================
# GLOBAL MATPLOTLIB SETTINGS (Springer/Nature Style)
# =============================================================================
def set_publication_style():
    """
    Apply global matplotlib settings for Springer/Nature publication style.
    Call this once at the beginning of any plotting script.
    """
    plt.rcParams.update({
        # Figure settings
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Default figure sizes
        "figure.figsize": (6, 4),        # Single-panel default

        # Font settings (Springer/Nature compliant)
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],

        # Font sizes
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,

        # Axes settings
        "axes.spines.top": False,        # Remove top spine
        "axes.spines.right": False,      # Remove right spine
        "axes.linewidth": 0.8,
        "axes.edgecolor": "0.2",

        # Grid settings (OFF by default for Springer/Nature)
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,

        # Tick settings
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend settings
        "legend.frameon": False,
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,

        # Line settings
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # Patch settings
        "patch.linewidth": 0,

        # Error bar settings
        "errorbar.capsize": 3,
    })


# =============================================================================
# FIGURE SIZE PRESETS
# =============================================================================
FIGURE_SIZES = {
    'single': (6, 4),          # Standard single panel
    'wide': (6.5, 4),          # Wide single panel (many categories)
    'tall': (6, 5),            # Tall single panel
    'double_vertical': (6, 7),  # Two panels stacked
    'double_horizontal': (12, 4), # Two panels side by side
    'square': (5, 5),          # Square plot (correlation, etc.)
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_transform_colors(n_transforms: int) -> List[str]:
    """
    Get colors for transform types, cycling through palette if needed.

    Args:
        n_transforms: Number of transform types

    Returns:
        List of hex color codes
    """
    if n_transforms <= len(PALETTE_B):
        return PALETTE_B[:n_transforms]
    else:
        # Cycle through palette if more transforms than colors
        colors = []
        for i in range(n_transforms):
            colors.append(PALETTE_B[i % len(PALETTE_B)])
        return colors


def add_reference_line(ax, y_value: float, label: Optional[str] = None,
                       style: str = '--', alpha: float = 0.5):
    """
    Add a subtle reference line to a plot.

    Args:
        ax: Matplotlib axis object
        y_value: Y-coordinate for horizontal line
        label: Optional label for the line
        style: Line style (default: dashed)
        alpha: Line transparency (default: 0.5)
    """
    ax.axhline(y_value, color='0.6', linestyle=style, linewidth=0.8,
               alpha=alpha, zorder=1)

    if label:
        # Add subtle label at the right edge
        ax.text(1.01, y_value, label, transform=ax.get_yaxis_transform(),
                va='center', ha='left', fontsize=8, color='0.5')


def format_axis_labels(ax, xlabel: Optional[str] = None,
                      ylabel: Optional[str] = None,
                      rotation: int = 0):
    """
    Apply consistent axis labeling.

    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label
        ylabel: Y-axis label
        rotation: X-tick label rotation (default: 0)
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if rotation > 0:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')


def add_value_labels(ax, bars, format_str: str = '{:.1f}',
                    threshold: Optional[float] = None):
    """
    Add value labels to bar chart (use sparingly!).

    Args:
        ax: Matplotlib axis object
        bars: Bar container from ax.bar()
        format_str: Format string for values
        threshold: Only label values above this threshold
    """
    for bar in bars:
        height = bar.get_height()
        if threshold is None or abs(height) > threshold:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   format_str.format(height),
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8, color='0.3')


def style_boxplot(bp, color_index: int = 0):
    """
    Apply consistent styling to a boxplot.

    Args:
        bp: Boxplot object from ax.boxplot()
        color_index: Index into PALETTE_B for color
    """
    color = PALETTE_B[color_index % len(PALETTE_B)]

    # Style the boxplot elements
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'caps']:
        if element in bp:
            plt.setp(bp[element], color=color, linewidth=1)

    # Make median lines more prominent
    if 'medians' in bp:
        plt.setp(bp['medians'], color=color, linewidth=2)

    # Style outlier points
    if 'fliers' in bp:
        plt.setp(bp['fliers'], markeredgecolor=color, markersize=4, alpha=0.5)


def create_grouped_bars(ax, data_dict: Dict[str, List[float]],
                       group_labels: List[str],
                       colors: Optional[List[str]] = None,
                       width: float = 0.35,
                       show_legend: bool = True):
    """
    Create grouped bar chart with consistent styling.

    Args:
        ax: Matplotlib axis object
        data_dict: Dictionary mapping series names to data values
        group_labels: Labels for x-axis groups
        colors: Optional color list (uses PALETTE_B by default)
        width: Bar width
        show_legend: Whether to show legend
    """
    n_groups = len(group_labels)
    n_series = len(data_dict)
    x = np.arange(n_groups)

    if colors is None:
        colors = PALETTE_B[:n_series]

    # Calculate bar positions
    offsets = np.linspace(-(n_series-1)*width/2, (n_series-1)*width/2, n_series)

    # Create bars
    for i, (label, values) in enumerate(data_dict.items()):
        ax.bar(x + offsets[i], values, width, label=label,
               color=colors[i % len(colors)])

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)

    # Add legend
    if show_legend:
        ax.legend(frameon=False, loc='best')


def save_figure(fig, filename: str, output_dir: str,
               tight_layout: bool = True, **kwargs):
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure object
        filename: Output filename (without path)
        output_dir: Output directory path
        tight_layout: Apply tight layout before saving
        **kwargs: Additional arguments for savefig
    """
    if tight_layout:
        fig.tight_layout()

    from pathlib import Path
    output_path = Path(output_dir) / filename

    # Default save parameters
    save_params = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.05,
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_params.update(kwargs)

    fig.savefig(output_path, **save_params)
    plt.close(fig)


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================
QUALITY_THRESHOLDS = {
    'psnr': {
        'excellent': 40,  # > 40 dB
        'good': 30,       # 30-40 dB
        'fair': 20,       # 20-30 dB
    },
    'ssim': {
        'excellent': 0.99,
        'good': 0.9,
        'fair': 0.7,
        'poor': 0.5,
    },
    'vmaf': {
        'excellent': 90,
        'good': 70,
        'fair': 50,
        'poor': 30,
    }
}


def add_quality_reference_lines(ax, metric: str, subtle: bool = True):
    """
    Add reference lines for quality metrics.

    Args:
        ax: Matplotlib axis object
        metric: One of 'psnr', 'ssim', 'vmaf'
        subtle: If True, use light gray lines
    """
    if metric not in QUALITY_THRESHOLDS:
        return

    thresholds = QUALITY_THRESHOLDS[metric]

    for level, value in thresholds.items():
        if subtle:
            ax.axhline(value, color='0.7', linestyle='--',
                      linewidth=0.8, alpha=0.5, zorder=1)
        else:
            color = {'excellent': PALETTE_B[2], 'good': PALETTE_B[4],
                    'fair': PALETTE_B[5], 'poor': PALETTE_B[3]}.get(level, '0.5')
            ax.axhline(value, color=color, linestyle='--',
                      linewidth=1, alpha=0.3, zorder=1)


# =============================================================================
# ANNOTATION LEVEL SETTINGS
# =============================================================================
class AnnotationLevel:
    """Annotation level presets matching user requirements."""

    NONE = 0      # No annotations at all
    MINIMAL = 1   # Only critical values (1-2 per figure max)
    MODERATE = 2  # Key values and thresholds
    FULL = 3      # All values, arrows, callouts (exploratory mode)


# Default to minimal for publication figures
DEFAULT_ANNOTATION_LEVEL = AnnotationLevel.MINIMAL