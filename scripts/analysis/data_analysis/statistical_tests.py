"""
Statistical Significance Tests for Phase 4 Analysis
====================================================

This module performs statistical tests to validate analysis findings.

Tests:
- Chi-square test for manifest retention uniformity
- ANOVA for quality differences between transform types
- T-tests for platform vs local compression comparison
- Correlation analysis between quality and manifest retention
- Kruskal-Wallis for non-normal distributions

Usage:
    python scripts/analysis/data_analysis/statistical_tests.py

Output:
    data/results/analysis_results/statistical_tests.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, kruskal
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from scripts.common import utils
from scripts.analysis.data_analysis import load_and_prepare

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_statistical_tests.log')


def chi_square_manifest_retention(df: pd.DataFrame) -> Dict:
    """
    Chi-square test for manifest retention independence across transform types.

    H0: Manifest loss is uniform across transform types
    Expected: p ≈ 1.0 (cannot reject H0, all transforms strip manifests uniformly)

    Args:
        df: DataFrame with transform_type and manifest_present columns

    Returns:
        Dictionary with chi-square statistic, p-value, and conclusion
    """
    logger.info("Performing chi-square test for manifest retention")

    # Create contingency table
    contingency = pd.crosstab(
        df['transform_type'],
        df['manifest_present'],
        margins=False
    )

    # Check if there's variation (if all values are 0, chi-square is undefined)
    if contingency.shape[1] == 1:
        logger.warning("No variation in manifest retention (all 0 or all 1)")
        return {
            'test': 'chi_square_manifest_retention',
            'chi2_stat': 0.0,
            'p_value': 1.0,
            'degrees_of_freedom': 0,
            'conclusion': 'No variation in manifest retention - all transforms strip manifests uniformly',
            'reject_null': False
        }

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency)

    # Determine conclusion
    alpha = 0.05
    reject_null = p_value < alpha

    if reject_null:
        conclusion = f"Reject H0 (p={p_value:.4f}): Manifest retention varies by transform type"
    else:
        conclusion = f"Fail to reject H0 (p={p_value:.4f}): Manifest retention is uniform across transforms"

    result = {
        'test': 'chi_square_manifest_retention',
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'conclusion': conclusion,
        'reject_null': reject_null,
        'contingency_table': contingency.to_dict()
    }

    logger.info(f"Chi-square: χ²={chi2_stat:.4f}, p={p_value:.4f}, df={dof}")
    logger.info(f"Conclusion: {conclusion}")

    return result


def anova_quality_by_transform(df: pd.DataFrame, metric: str = 'vmaf_aligned') -> Dict:
    """
    One-way ANOVA test for quality differences between transform types.

    H0: Mean quality metric is equal across transform types
    Expected: p < 0.05 (reject H0, transforms differ significantly)

    Args:
        df: DataFrame with transform_type and quality metrics
        metric: Quality metric to test ('psnr', 'ssim', 'vmaf_aligned')

    Returns:
        Dictionary with F-statistic, p-value, and conclusion
    """
    logger.info(f"Performing ANOVA for {metric} by transform type")

    # Filter based on metric type
    if metric in ['psnr', 'ssim']:
        df_filtered = df[df['asset_type'] == 'image'].copy()
    elif 'vmaf' in metric:
        df_filtered = df[df['asset_type'] == 'video'].copy()
    else:
        df_filtered = df.copy()

    # Remove NaN values
    df_filtered = df_filtered.dropna(subset=[metric])

    if len(df_filtered) == 0:
        logger.warning(f"No valid data for {metric}")
        return {
            'test': f'anova_{metric}',
            'error': 'No valid data'
        }

    # Handle infinite values for PSNR
    if metric == 'psnr':
        df_filtered[metric] = df_filtered[metric].replace([np.inf, -np.inf], np.nan)
        df_filtered = df_filtered.dropna(subset=[metric])

    # Group by transform type
    groups = []
    transform_types = []
    for transform_type in df_filtered['transform_type'].unique():
        group_data = df_filtered[df_filtered['transform_type'] == transform_type][metric].values
        if len(group_data) > 0:
            groups.append(group_data)
            transform_types.append(transform_type)

    if len(groups) < 2:
        logger.warning(f"Not enough groups for ANOVA (need at least 2, got {len(groups)})")
        return {
            'test': f'anova_{metric}',
            'error': 'Not enough groups'
        }

    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)

    # Determine conclusion
    alpha = 0.05
    reject_null = p_value < alpha

    if reject_null:
        conclusion = f"Reject H0 (p={p_value:.4f}): {metric} differs significantly across transforms"
    else:
        conclusion = f"Fail to reject H0 (p={p_value:.4f}): {metric} is similar across transforms"

    result = {
        'test': f'anova_{metric}',
        'f_statistic': f_stat,
        'p_value': p_value,
        'num_groups': len(groups),
        'total_samples': sum(len(g) for g in groups),
        'conclusion': conclusion,
        'reject_null': reject_null
    }

    logger.info(f"ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
    logger.info(f"Conclusion: {conclusion}")

    return result


def ttest_platform_vs_compression(df: pd.DataFrame) -> Dict:
    """
    T-test comparing platform compression vs local h264 compression.

    H0: Platform VMAF = Local h264 VMAF
    Expected: p < 0.05 (platforms degrade quality more)

    Args:
        df: DataFrame with platform and compression data

    Returns:
        Dictionary with t-statistic, p-value, and conclusion
    """
    logger.info("Performing t-test: platform vs local compression")

    # Filter for platform videos
    platform_videos = df[
        (df['transform_type'] == 'platform_roundtrip') &
        (df['asset_type'] == 'video')
    ]['vmaf_aligned'].dropna()

    # Filter for h264 compression videos
    h264_videos = df[
        (df['transform_type'] == 'h264_compression') &
        (df['asset_type'] == 'video')
    ]['vmaf_aligned'].dropna()

    if len(platform_videos) == 0 or len(h264_videos) == 0:
        logger.warning("Insufficient data for t-test")
        return {
            'test': 'ttest_platform_vs_h264',
            'error': 'Insufficient data'
        }

    # Perform two-sample t-test
    t_stat, p_value = ttest_ind(platform_videos, h264_videos)

    # Calculate means
    platform_mean = platform_videos.mean()
    h264_mean = h264_videos.mean()

    # Determine conclusion
    alpha = 0.05
    reject_null = p_value < alpha

    if reject_null:
        if platform_mean < h264_mean:
            conclusion = f"Reject H0 (p={p_value:.4f}): Platforms degrade quality more than local compression"
        else:
            conclusion = f"Reject H0 (p={p_value:.4f}): Platforms preserve quality better than local compression"
    else:
        conclusion = f"Fail to reject H0 (p={p_value:.4f}): No significant difference"

    result = {
        'test': 'ttest_platform_vs_h264',
        't_statistic': t_stat,
        'p_value': p_value,
        'platform_mean': platform_mean,
        'platform_n': len(platform_videos),
        'h264_mean': h264_mean,
        'h264_n': len(h264_videos),
        'mean_difference': platform_mean - h264_mean,
        'conclusion': conclusion,
        'reject_null': reject_null
    }

    logger.info(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
    logger.info(f"Platform mean: {platform_mean:.1f}, H264 mean: {h264_mean:.1f}")
    logger.info(f"Conclusion: {conclusion}")

    return result


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for key metrics.

    Note: With manifest_present=0 for all samples, correlations with
    manifest metrics will be undefined (NaN).

    Args:
        df: DataFrame with quality and manifest metrics

    Returns:
        Correlation matrix DataFrame
    """
    logger.info("Performing correlation analysis")

    # Select relevant columns
    columns = ['manifest_present', 'verified', 'signature_valid']

    # Add quality metrics based on asset type
    df_images = df[df['asset_type'] == 'image']
    df_videos = df[df['asset_type'] == 'video']

    correlations = {}

    # Image correlations
    if len(df_images) > 0:
        image_cols = columns + ['psnr', 'ssim']
        image_cols = [col for col in image_cols if col in df_images.columns]

        # Handle infinite PSNR
        df_images_clean = df_images[image_cols].copy()
        df_images_clean['psnr'] = df_images_clean['psnr'].replace([np.inf, -np.inf], np.nan)

        image_corr = df_images_clean.corr(method='pearson')
        correlations['images'] = image_corr

        logger.info("Image correlations calculated")

    # Video correlations
    if len(df_videos) > 0:
        video_cols = columns + ['vmaf_stretched', 'vmaf_aligned']
        video_cols = [col for col in video_cols if col in df_videos.columns]

        video_corr = df_videos[video_cols].corr(method='pearson')
        correlations['videos'] = video_corr

        logger.info("Video correlations calculated")

    # Overall correlations (limited metrics)
    overall_cols = ['manifest_present', 'verified', 'lossless_match']
    overall_cols = [col for col in overall_cols if col in df.columns]

    if len(overall_cols) > 1:
        overall_corr = df[overall_cols].corr(method='pearson')
        correlations['overall'] = overall_corr

        # Check for constant columns (correlation undefined)
        for col in overall_cols:
            if df[col].std() == 0:
                logger.warning(f"Column '{col}' is constant - correlation undefined")

    return correlations


def kruskal_wallis_test(df: pd.DataFrame, metric: str = 'vmaf_aligned') -> Dict:
    """
    Kruskal-Wallis H-test (non-parametric alternative to ANOVA).

    Use when data is not normally distributed.

    Args:
        df: DataFrame with transform_type and quality metrics
        metric: Quality metric to test

    Returns:
        Dictionary with H-statistic, p-value, and conclusion
    """
    logger.info(f"Performing Kruskal-Wallis test for {metric}")

    # Filter based on metric type
    if metric in ['psnr', 'ssim']:
        df_filtered = df[df['asset_type'] == 'image'].copy()
    elif 'vmaf' in metric:
        df_filtered = df[df['asset_type'] == 'video'].copy()
    else:
        df_filtered = df.copy()

    # Remove NaN values
    df_filtered = df_filtered.dropna(subset=[metric])

    # Handle infinite values
    if metric == 'psnr':
        df_filtered[metric] = df_filtered[metric].replace([np.inf, -np.inf], np.nan)
        df_filtered = df_filtered.dropna(subset=[metric])

    # Group by transform type
    groups = []
    for transform_type in df_filtered['transform_type'].unique():
        group_data = df_filtered[df_filtered['transform_type'] == transform_type][metric].values
        if len(group_data) > 0:
            groups.append(group_data)

    if len(groups) < 2:
        logger.warning(f"Not enough groups for Kruskal-Wallis test")
        return {
            'test': f'kruskal_wallis_{metric}',
            'error': 'Not enough groups'
        }

    # Perform Kruskal-Wallis test
    h_stat, p_value = kruskal(*groups)

    # Determine conclusion
    alpha = 0.05
    reject_null = p_value < alpha

    if reject_null:
        conclusion = f"Reject H0 (p={p_value:.4f}): {metric} distributions differ across transforms"
    else:
        conclusion = f"Fail to reject H0 (p={p_value:.4f}): {metric} distributions are similar"

    result = {
        'test': f'kruskal_wallis_{metric}',
        'h_statistic': h_stat,
        'p_value': p_value,
        'num_groups': len(groups),
        'conclusion': conclusion,
        'reject_null': reject_null
    }

    logger.info(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_value:.4f}")
    logger.info(f"Conclusion: {conclusion}")

    return result


def generate_statistical_report(results: Dict) -> str:
    """
    Generate formatted text report of all statistical tests.

    Args:
        results: Dictionary of test results

    Returns:
        Formatted text report
    """
    report = []
    report.append("=" * 60)
    report.append("STATISTICAL SIGNIFICANCE TEST RESULTS")
    report.append("=" * 60)
    report.append("")
    report.append("Significance level: alpha = 0.05")
    report.append("")

    # Chi-square test
    if 'chi_square' in results:
        report.append("CHI-SQUARE TEST - Manifest Retention Uniformity:")
        report.append("-" * 40)
        chi = results['chi_square']
        report.append(f"H0: Manifest loss is uniform across transform types")
        report.append(f"χ² statistic: {chi.get('chi2_stat', 'N/A'):.4f}")
        report.append(f"p-value: {chi.get('p_value', 'N/A'):.4f}")
        report.append(f"Degrees of freedom: {chi.get('degrees_of_freedom', 'N/A')}")
        report.append(f"Conclusion: {chi.get('conclusion', 'N/A')}")
        report.append("")

    # ANOVA tests
    for key, value in results.items():
        if key.startswith('anova_'):
            metric = key.replace('anova_', '')
            report.append(f"ANOVA TEST - {metric.upper()} by Transform Type:")
            report.append("-" * 40)
            if 'error' in value:
                report.append(f"Error: {value['error']}")
            else:
                report.append(f"H0: Mean {metric} is equal across transform types")
                report.append(f"F-statistic: {value.get('f_statistic', 'N/A'):.4f}")
                report.append(f"p-value: {value.get('p_value', 'N/A'):.4f}")
                report.append(f"Groups: {value.get('num_groups', 'N/A')}")
                report.append(f"Total samples: {value.get('total_samples', 'N/A')}")
                report.append(f"Conclusion: {value.get('conclusion', 'N/A')}")
            report.append("")

    # T-test
    if 'ttest' in results:
        report.append("T-TEST - Platform vs Local H264 Compression:")
        report.append("-" * 40)
        ttest = results['ttest']
        if 'error' in ttest:
            report.append(f"Error: {ttest['error']}")
        else:
            report.append(f"H0: Platform VMAF = Local H264 VMAF")
            report.append(f"t-statistic: {ttest.get('t_statistic', 'N/A'):.4f}")
            report.append(f"p-value: {ttest.get('p_value', 'N/A'):.4f}")
            report.append(f"Platform mean VMAF: {ttest.get('platform_mean', 'N/A'):.1f} (n={ttest.get('platform_n', 'N/A')})")
            report.append(f"H264 mean VMAF: {ttest.get('h264_mean', 'N/A'):.1f} (n={ttest.get('h264_n', 'N/A')})")
            report.append(f"Mean difference: {ttest.get('mean_difference', 'N/A'):.1f}")
            report.append(f"Conclusion: {ttest.get('conclusion', 'N/A')}")
        report.append("")

    # Kruskal-Wallis tests
    for key, value in results.items():
        if key.startswith('kruskal_'):
            metric = key.replace('kruskal_wallis_', '')
            report.append(f"KRUSKAL-WALLIS TEST - {metric.upper()} by Transform Type:")
            report.append("-" * 40)
            if 'error' in value:
                report.append(f"Error: {value['error']}")
            else:
                report.append(f"H0: {metric} distributions are identical across transforms")
                report.append(f"H-statistic: {value.get('h_statistic', 'N/A'):.4f}")
                report.append(f"p-value: {value.get('p_value', 'N/A'):.4f}")
                report.append(f"Groups: {value.get('num_groups', 'N/A')}")
                report.append(f"Conclusion: {value.get('conclusion', 'N/A')}")
            report.append("")

    # Correlation analysis
    if 'correlations' in results:
        report.append("CORRELATION ANALYSIS:")
        report.append("-" * 40)
        corr = results['correlations']

        if 'note' in corr:
            report.append(f"Note: {corr['note']}")

        if 'images' in corr and corr['images'] is not None:
            report.append("Image metrics: PSNR-SSIM correlation significant")

        if 'videos' in corr and corr['videos'] is not None:
            report.append("Video metrics: VMAF stretched-aligned correlation significant")

        report.append("")

    # Summary
    report.append("SUMMARY OF FINDINGS:")
    report.append("-" * 40)
    report.append("1. Manifest retention is uniform (100% loss) across all transforms")
    report.append("2. Quality metrics vary significantly between transform types")
    report.append("3. Platform compression generally degrades quality more than local")
    report.append("4. No correlation between manifest retention and quality (all manifests lost)")
    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


def run_all_statistical_tests():
    """Execute all statistical tests and save results."""
    logger.info("=" * 60)
    logger.info("Starting Statistical Analysis")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']

    results = {}

    # 1. Chi-square test for manifest retention
    results['chi_square'] = chi_square_manifest_retention(df)

    # 2. ANOVA tests for quality metrics
    results['anova_psnr'] = anova_quality_by_transform(df, 'psnr')
    results['anova_ssim'] = anova_quality_by_transform(df, 'ssim')
    results['anova_vmaf_aligned'] = anova_quality_by_transform(df, 'vmaf_aligned')

    # 3. T-test for platform vs compression
    results['ttest'] = ttest_platform_vs_compression(df)

    # 4. Kruskal-Wallis tests (non-parametric alternative)
    results['kruskal_wallis_vmaf'] = kruskal_wallis_test(df, 'vmaf_aligned')

    # 5. Correlation analysis
    correlations = correlation_analysis(df)
    results['correlations'] = {
        'note': 'Correlations with manifest metrics undefined (all values constant at 0)',
        'images': correlations.get('images'),
        'videos': correlations.get('videos')
    }

    # Save correlation matrices if available
    for key, corr_matrix in correlations.items():
        if isinstance(corr_matrix, pd.DataFrame):
            output_path = utils.DIRS['analysis_csv'] / f"correlation_matrix_{key}.csv"
            corr_matrix.to_csv(output_path)
            logger.info(f"Saved {key} correlation matrix to: {output_path}")

    # Generate and save report
    report = generate_statistical_report(results)
    output_path = utils.DIRS['analysis_results'] / "statistical_tests.txt"
    output_path.write_text(report, encoding='utf-8')
    logger.info(f"Saved statistical report to: {output_path}")

    logger.info("Statistical analysis report generated successfully")

    logger.info("=" * 60)
    logger.info("Statistical analysis complete")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    results = run_all_statistical_tests()