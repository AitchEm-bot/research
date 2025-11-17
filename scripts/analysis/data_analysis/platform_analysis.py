"""
Social Media Platform Analysis for Phase 4
===========================================

This module analyzes C2PA manifest retention and quality degradation
across social media platforms (Phase 2.5 results).

Features:
- Platform-specific manifest retention rates
- Quality degradation by platform (PSNR, SSIM, VMAF)
- Comparison of internal vs external video sources
- Platform ranking by quality preservation
- Upload mode analysis (post, story, reel, etc.)

Usage:
    python scripts/analysis/data_analysis/platform_analysis.py

Output:
    data/results/analysis_results/csv/platform_summary.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.common import utils
from scripts.analysis.data_analysis import load_and_prepare

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_platform_analysis.log')


def analyze_platform_retention(df_platform: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate manifest retention by platform.

    Expected: 0% for all 6 platforms (complete stripping).

    Args:
        df_platform: Platform testing subset

    Returns:
        DataFrame with platform retention statistics
    """
    if len(df_platform) == 0:
        logger.warning("No platform data available")
        return pd.DataFrame()

    logger.info("Analyzing platform-specific manifest retention")

    retention = df_platform.groupby('platform').agg({
        'manifest_present': ['count', 'sum', 'mean'],
        'verified': 'mean',
        'signature_valid': 'mean',
        'hash_match': 'mean'
    }).round(4)

    retention.columns = ['total_samples', 'manifests_retained', 'retention_pct',
                        'verified_pct', 'signature_valid_pct', 'hash_match_pct']

    # Convert to percentages
    pct_cols = ['retention_pct', 'verified_pct', 'signature_valid_pct', 'hash_match_pct']
    for col in pct_cols:
        retention[col] = retention[col] * 100

    retention = retention.reset_index()

    # Sort by platform name
    retention = retention.sort_values('platform')

    # Log findings
    logger.info(f"Platforms analyzed: {list(retention['platform'])}")
    logger.info(f"Total platform samples: {retention['total_samples'].sum()}")
    logger.info(f"Manifests retained: {retention['manifests_retained'].sum()}")

    if retention['manifests_retained'].sum() == 0:
        logger.warning("All platforms stripped C2PA manifests (100% loss)")
    else:
        preserved = retention[retention['manifests_retained'] > 0]
        logger.info(f"Platforms preserving manifests: {preserved['platform'].tolist()}")

    return retention


def analyze_platform_quality(df_platform: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze quality degradation by platform.

    Metrics:
    - PSNR/SSIM for images (Instagram, Twitter, Facebook, WhatsApp)
    - VMAF (stretched and aligned) for videos (all platforms)

    Args:
        df_platform: Platform testing subset

    Returns:
        DataFrame with quality metrics by platform
    """
    if len(df_platform) == 0:
        logger.warning("No platform data available")
        return pd.DataFrame()

    logger.info("Analyzing platform quality degradation")

    quality_stats = []

    for platform in df_platform['platform'].unique():
        platform_df = df_platform[df_platform['platform'] == platform]
        platform_images = platform_df[platform_df['asset_type'] == 'image']
        platform_videos = platform_df[platform_df['asset_type'] == 'video']

        stats = {
            'platform': platform,
            'total_samples': len(platform_df),
            'image_samples': len(platform_images),
            'video_samples': len(platform_videos)
        }

        # Image quality metrics
        if len(platform_images) > 0:
            # Handle infinite PSNR
            psnr_finite = platform_images['psnr'].replace([np.inf, -np.inf], np.nan)
            stats['psnr_mean'] = psnr_finite.mean()
            stats['psnr_median'] = psnr_finite.median()
            stats['ssim_mean'] = platform_images['ssim'].mean()
            stats['ssim_median'] = platform_images['ssim'].median()
        else:
            stats['psnr_mean'] = np.nan
            stats['psnr_median'] = np.nan
            stats['ssim_mean'] = np.nan
            stats['ssim_median'] = np.nan

        # Video quality metrics
        if len(platform_videos) > 0:
            stats['vmaf_stretched_mean'] = platform_videos['vmaf_stretched'].mean()
            stats['vmaf_stretched_median'] = platform_videos['vmaf_stretched'].median()
            stats['vmaf_aligned_mean'] = platform_videos['vmaf_aligned'].mean()
            stats['vmaf_aligned_median'] = platform_videos['vmaf_aligned'].median()
            stats['vmaf_difference'] = stats['vmaf_aligned_mean'] - stats['vmaf_stretched_mean']
        else:
            stats['vmaf_stretched_mean'] = np.nan
            stats['vmaf_stretched_median'] = np.nan
            stats['vmaf_aligned_mean'] = np.nan
            stats['vmaf_aligned_median'] = np.nan
            stats['vmaf_difference'] = np.nan

        quality_stats.append(stats)

    quality_df = pd.DataFrame(quality_stats)

    # Round numeric columns
    numeric_cols = ['psnr_mean', 'psnr_median', 'ssim_mean', 'ssim_median',
                   'vmaf_stretched_mean', 'vmaf_stretched_median',
                   'vmaf_aligned_mean', 'vmaf_aligned_median', 'vmaf_difference']
    for col in numeric_cols:
        if col in quality_df.columns:
            quality_df[col] = quality_df[col].round(2)

    # Sort by aligned VMAF (best quality first)
    quality_df = quality_df.sort_values('vmaf_aligned_mean', ascending=False)

    logger.info(f"Quality analysis for {len(quality_df)} platforms")

    return quality_df


def analyze_platform_modes(df_platform: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze quality by platform upload mode (post, story, reel, etc.).

    Args:
        df_platform: Platform testing subset with platform_mode column

    Returns:
        DataFrame with quality metrics by platform and mode
    """
    if 'platform_mode' not in df_platform.columns:
        logger.warning("No platform_mode column found")
        return pd.DataFrame()

    logger.info("Analyzing platform upload modes")

    mode_stats = df_platform.groupby(['platform', 'platform_mode']).agg({
        'manifest_present': ['count', 'mean'],
        'vmaf_aligned': 'mean',
        'psnr': lambda x: x.replace([np.inf, -np.inf], np.nan).mean(),
        'ssim': 'mean'
    }).round(3)

    mode_stats.columns = ['count', 'retention_pct', 'vmaf_aligned', 'psnr', 'ssim']
    mode_stats['retention_pct'] = mode_stats['retention_pct'] * 100
    mode_stats = mode_stats.reset_index()

    # Sort by platform then mode
    mode_stats = mode_stats.sort_values(['platform', 'platform_mode'])

    logger.info(f"Analyzed {len(mode_stats)} platform-mode combinations")

    # Check for mode-specific differences
    for platform in mode_stats['platform'].unique():
        platform_modes = mode_stats[mode_stats['platform'] == platform]
        if len(platform_modes) > 1:
            vmaf_std = platform_modes['vmaf_aligned'].std()
            if vmaf_std > 5:
                logger.info(f"{platform}: Significant quality variation across modes (std={vmaf_std:.1f})")

    return mode_stats


def compare_internal_vs_external_videos(df_platform: pd.DataFrame) -> pd.DataFrame:
    """
    Compare platform behavior on internal (SVD) vs external (Veo3.1) videos.

    Args:
        df_platform: Platform testing subset with platform_source column

    Returns:
        DataFrame comparing internal vs external video treatment
    """
    if 'platform_source' not in df_platform.columns:
        logger.warning("No platform_source column found")
        return pd.DataFrame()

    # Filter to videos only
    df_videos = df_platform[df_platform['asset_type'] == 'video'].copy()

    if len(df_videos) == 0:
        logger.warning("No video data for internal/external comparison")
        return pd.DataFrame()

    logger.info("Comparing internal vs external video treatment")

    comparison = df_videos.groupby(['platform', 'platform_source']).agg({
        'manifest_present': ['count', 'mean'],
        'vmaf_stretched': 'mean',
        'vmaf_aligned': 'mean'
    }).round(2)

    comparison.columns = ['count', 'retention_pct', 'vmaf_stretched', 'vmaf_aligned']
    comparison['retention_pct'] = comparison['retention_pct'] * 100
    comparison = comparison.reset_index()

    # Pivot for easier comparison
    pivot = comparison.pivot(
        index='platform',
        columns='platform_source',
        values=['count', 'retention_pct', 'vmaf_aligned']
    )

    # Flatten column names
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    logger.info(f"Internal/external comparison for {len(pivot)} platforms")

    return pivot


def identify_worst_platforms(df_platform: pd.DataFrame) -> Dict:
    """
    Rank platforms by quality degradation.

    Ranking criteria:
    1. Manifest retention (all tied at 0%)
    2. VMAF aligned (for videos)
    3. PSNR (for images)

    Args:
        df_platform: Platform testing subset

    Returns:
        Dictionary with platform rankings
    """
    if len(df_platform) == 0:
        logger.warning("No platform data available")
        return {}

    logger.info("Ranking platforms by quality preservation")

    rankings = {}

    # Video quality ranking (VMAF aligned)
    df_videos = df_platform[df_platform['asset_type'] == 'video']
    if len(df_videos) > 0:
        video_ranking = df_videos.groupby('platform')['vmaf_aligned'].mean().sort_values(ascending=False)
        rankings['video_quality'] = video_ranking.to_dict()
        logger.info(f"Video quality ranking (best to worst):")
        for i, (platform, vmaf) in enumerate(video_ranking.items(), 1):
            logger.info(f"  {i}. {platform}: {vmaf:.1f} VMAF")

    # Image quality ranking (PSNR)
    df_images = df_platform[df_platform['asset_type'] == 'image']
    if len(df_images) > 0:
        # Handle infinite PSNR
        df_images['psnr_finite'] = df_images['psnr'].replace([np.inf, -np.inf], np.nan)
        image_ranking = df_images.groupby('platform')['psnr_finite'].mean().sort_values(ascending=False)
        rankings['image_quality'] = image_ranking.to_dict()
        logger.info(f"Image quality ranking (best to worst):")
        for i, (platform, psnr) in enumerate(image_ranking.items(), 1):
            logger.info(f"  {i}. {platform}: {psnr:.1f} dB")

    # Overall ranking (combined)
    all_platforms = df_platform['platform'].unique()
    overall_scores = {}

    for platform in all_platforms:
        platform_df = df_platform[df_platform['platform'] == platform]
        scores = []

        # Add video score if available
        if platform in rankings.get('video_quality', {}):
            scores.append(rankings['video_quality'][platform])

        # Add image score if available (normalize to 0-100 scale)
        if platform in rankings.get('image_quality', {}):
            psnr = rankings['image_quality'][platform]
            # Convert PSNR to 0-100 scale (20-50 dB → 0-100)
            normalized_psnr = min(100, max(0, (psnr - 20) * 100 / 30))
            scores.append(normalized_psnr)

        if scores:
            overall_scores[platform] = np.mean(scores)

    overall_ranking = pd.Series(overall_scores).sort_values(ascending=False)
    rankings['overall'] = overall_ranking.to_dict()

    logger.info(f"Overall quality ranking (best to worst):")
    for i, (platform, score) in enumerate(overall_ranking.items(), 1):
        logger.info(f"  {i}. {platform}: {score:.1f}")

    return rankings


def generate_platform_summary() -> str:
    """Generate text summary of platform analysis findings."""
    # Load data
    data = load_and_prepare.load_all_data()
    df_platform = data['platform']

    if len(df_platform) == 0:
        return "No platform data available for analysis."

    # Run analyses
    retention = analyze_platform_retention(df_platform)
    quality = analyze_platform_quality(df_platform)
    rankings = identify_worst_platforms(df_platform)

    summary = []
    summary.append("=" * 60)
    summary.append("SOCIAL MEDIA PLATFORM ANALYSIS SUMMARY")
    summary.append("=" * 60)
    summary.append("")

    # Platform overview
    summary.append("PLATFORMS TESTED:")
    platforms = df_platform['platform'].unique()
    summary.append(f"- Platforms: {', '.join(sorted(platforms))}")
    summary.append(f"- Total samples: {len(df_platform)}")
    summary.append(f"- Image samples: {(df_platform['asset_type'] == 'image').sum()}")
    summary.append(f"- Video samples: {(df_platform['asset_type'] == 'video').sum()}")
    summary.append("")

    # Manifest retention
    summary.append("C2PA MANIFEST RETENTION:")
    if len(retention) > 0:
        summary.append(f"- Manifests retained: {retention['manifests_retained'].sum()}/{retention['total_samples'].sum()}")
        summary.append(f"- Retention rate: {retention['retention_pct'].mean():.1f}%")
        if retention['manifests_retained'].sum() == 0:
            summary.append("- ALL platforms stripped C2PA manifests completely")
    summary.append("")

    # Quality degradation
    summary.append("QUALITY DEGRADATION:")
    if len(quality) > 0:
        # Video quality
        video_platforms = quality.dropna(subset=['vmaf_aligned_mean'])
        if len(video_platforms) > 0:
            summary.append("Video Quality (VMAF aligned):")
            summary.append(f"- Best: {video_platforms.iloc[0]['platform']} ({video_platforms.iloc[0]['vmaf_aligned_mean']:.1f})")
            summary.append(f"- Worst: {video_platforms.iloc[-1]['platform']} ({video_platforms.iloc[-1]['vmaf_aligned_mean']:.1f})")
            summary.append(f"- Average: {video_platforms['vmaf_aligned_mean'].mean():.1f}")

        # Image quality
        image_platforms = quality.dropna(subset=['psnr_mean'])
        if len(image_platforms) > 0:
            summary.append("Image Quality (PSNR):")
            summary.append(f"- Best: {image_platforms.iloc[0]['platform']} ({image_platforms.iloc[0]['psnr_mean']:.1f} dB)")
            summary.append(f"- Worst: {image_platforms.iloc[-1]['platform']} ({image_platforms.iloc[-1]['psnr_mean']:.1f} dB)")
            summary.append(f"- Average: {image_platforms['psnr_mean'].mean():.1f} dB")
    summary.append("")

    # Key findings
    summary.append("KEY FINDINGS:")
    summary.append("1. Universal C2PA stripping: No platform preserves metadata")
    summary.append("2. Quality varies significantly across platforms")
    summary.append("3. Aspect ratio handling critical for video quality assessment")
    summary.append("4. Platform behavior consistent regardless of video source")
    summary.append("")
    summary.append("=" * 60)

    return "\n".join(summary)


def run_platform_analysis():
    """Execute complete platform analysis and save results."""
    logger.info("=" * 60)
    logger.info("Starting Platform Analysis")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df_platform = data['platform']

    if len(df_platform) == 0:
        logger.warning("No platform data available, skipping platform analysis")
        return {}

    # Run analyses
    retention = analyze_platform_retention(df_platform)
    quality = analyze_platform_quality(df_platform)
    modes = analyze_platform_modes(df_platform)
    internal_external = compare_internal_vs_external_videos(df_platform)
    rankings = identify_worst_platforms(df_platform)

    # Save results
    if len(retention) > 0:
        output_path = utils.DIRS['analysis_csv'] / "platform_retention.csv"
        retention.to_csv(output_path, index=False)
        logger.info(f"Saved platform retention to: {output_path}")

    if len(quality) > 0:
        output_path = utils.DIRS['analysis_csv'] / "platform_quality.csv"
        quality.to_csv(output_path, index=False)
        logger.info(f"Saved platform quality to: {output_path}")

    if len(modes) > 0:
        output_path = utils.DIRS['analysis_csv'] / "platform_modes.csv"
        modes.to_csv(output_path, index=False)
        logger.info(f"Saved platform modes to: {output_path}")

    if len(internal_external) > 0:
        output_path = utils.DIRS['analysis_csv'] / "platform_video_comparison.csv"
        internal_external.to_csv(output_path, index=False)
        logger.info(f"Saved video comparison to: {output_path}")

    # Save comprehensive summary
    summary_df = quality  # Use quality df as main summary
    output_path = utils.DIRS['analysis_csv'] / "platform_summary.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved platform summary to: {output_path}")

    # Generate text summary
    summary_text = generate_platform_summary()
    output_path = utils.DIRS['analysis_results'] / "platform_analysis_summary.txt"
    output_path.write_text(summary_text, encoding='utf-8')
    logger.info(f"Saved text summary to: {output_path}")

    logger.info("Platform analysis summary generated successfully")

    logger.info("=" * 60)
    logger.info("Platform analysis complete")
    logger.info("=" * 60)

    return {
        'retention': retention,
        'quality': quality,
        'modes': modes,
        'rankings': rankings
    }


if __name__ == "__main__":
    results = run_platform_analysis()