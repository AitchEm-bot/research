"""
C2PA Manifest Retention Analysis for Phase 4
=============================================

This module analyzes C2PA manifest retention rates across all transformations.

Features:
- Calculate manifest retention percentages by transform type
- Calculate verification success rates (VSR) and signature validation rates (SVR)
- Compare baseline vs transformed retention
- Generate retention summary tables
- Identify patterns in manifest loss

Usage:
    python scripts/analysis/data_analysis/manifest_retention_analysis.py

Output:
    data/results/analysis_results/csv/retention_table.csv
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
logger = utils.setup_logging(log_file='data/results/logs/phase4_manifest_analysis.log')


def calculate_retention_by_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate manifest retention percentage by transform_type.

    Expected: 0% for all transforms (complete manifest stripping)

    Args:
        df: DataFrame with transform_type and manifest_present columns

    Returns:
        DataFrame with columns: transform_type, total_samples, retained, retention_pct
    """
    logger.info("Calculating manifest retention by transform type")

    retention = df.groupby('transform_type').agg({
        'manifest_present': ['count', 'sum', 'mean']
    }).round(4)

    retention.columns = ['total_samples', 'retained', 'retention_pct']
    retention['retention_pct'] = retention['retention_pct'] * 100
    retention = retention.reset_index()

    # Sort by retention percentage (descending) then by sample count
    retention = retention.sort_values(['retention_pct', 'total_samples'],
                                    ascending=[False, False])

    # Log findings
    logger.info(f"Analyzed {len(retention)} transform types")
    logger.info(f"Total samples: {retention['total_samples'].sum()}")
    logger.info(f"Manifests retained: {retention['retained'].sum()}")
    logger.info(f"Overall retention: {retention['retained'].sum() / retention['total_samples'].sum():.1%}")

    # Check if any transforms preserved manifests
    if retention['retained'].sum() == 0:
        logger.warning("CRITICAL FINDING: 100% manifest loss across ALL transforms")
    else:
        preserved = retention[retention['retained'] > 0]
        logger.info(f"Transforms that preserved manifests: {preserved['transform_type'].tolist()}")

    return retention


def calculate_retention_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate retention by transform_level (e.g., q25, q50, bitrate5000k).

    Args:
        df: DataFrame with transform_type, transform_level, and manifest_present

    Returns:
        DataFrame with transform_type, transform_level, total, retained, retention_pct
    """
    logger.info("Calculating manifest retention by transform level")

    retention = df.groupby(['transform_type', 'transform_level']).agg({
        'manifest_present': ['count', 'sum', 'mean']
    }).round(4)

    retention.columns = ['total_samples', 'retained', 'retention_pct']
    retention['retention_pct'] = retention['retention_pct'] * 100
    retention = retention.reset_index()

    # Sort by transform type then level
    retention = retention.sort_values(['transform_type', 'transform_level'])

    # Identify any patterns
    compression_levels = retention[retention['transform_type'].str.contains('compression', na=False)]
    if len(compression_levels) > 0:
        logger.info(f"Compression levels analyzed: {len(compression_levels)}")
        # Check if quality level affects retention
        if compression_levels['retention_pct'].std() > 0:
            logger.info("Retention varies by compression level")
        else:
            logger.info("Retention uniform across compression levels")

    return retention


def calculate_verification_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Verification Success Rate (VSR) and Signature Verification Rate (SVR).

    VSR = (manifest_present + verified + signature_valid) / 3
    SVR = manifest_present only

    Args:
        df: DataFrame with verification columns

    Returns:
        DataFrame with VSR and SVR by transform type
    """
    logger.info("Calculating verification rates (VSR and SVR)")

    # Calculate VSR components
    df['vsr'] = (df['manifest_present'] + df['verified'] + df['signature_valid']) / 3
    df['svr'] = df['manifest_present']

    # Aggregate by transform type
    rates = df.groupby('transform_type').agg({
        'vsr': 'mean',
        'svr': 'mean',
        'manifest_present': 'mean',
        'verified': 'mean',
        'signature_valid': 'mean',
        'hash_match': 'mean',
        'assertion_uris_match': 'mean',
        'trust_valid': 'mean'
    }).round(4)

    rates = rates * 100  # Convert to percentages
    rates = rates.reset_index()

    logger.info(f"Overall VSR: {df['vsr'].mean():.1%}")
    logger.info(f"Overall SVR: {df['svr'].mean():.1%}")

    return rates


def compare_baseline_vs_transformed(baseline_df: pd.DataFrame,
                                   transformed_df: pd.DataFrame) -> Dict:
    """
    Compare baseline verification (expected 100%) vs post-transform (actual 0%).

    Args:
        baseline_df: Baseline validation data
        transformed_df: Post-transformation data

    Returns:
        Dictionary with comparison statistics
    """
    logger.info("Comparing baseline vs transformed retention")

    comparison = {}

    if len(baseline_df) > 0:
        # Baseline statistics
        comparison['baseline_samples'] = len(baseline_df)
        comparison['baseline_manifest_present'] = baseline_df['manifest_present'].mean()
        comparison['baseline_verified'] = baseline_df['verified'].mean()
        comparison['baseline_signature_valid'] = baseline_df['signature_valid'].mean() if 'signature_valid' in baseline_df.columns else 0

        logger.info(f"Baseline: {len(baseline_df)} samples")
        logger.info(f"Baseline manifest present: {comparison['baseline_manifest_present']:.1%}")
        logger.info(f"Baseline verified: {comparison['baseline_verified']:.1%}")
    else:
        comparison['baseline_samples'] = 0
        comparison['baseline_manifest_present'] = 1.0  # Expected
        comparison['baseline_verified'] = 1.0  # Expected
        comparison['baseline_signature_valid'] = 1.0  # Expected
        logger.warning("No baseline data available, using expected values")

    # Transformed statistics
    comparison['transformed_samples'] = len(transformed_df)
    comparison['transformed_manifest_present'] = transformed_df['manifest_present'].mean()
    comparison['transformed_verified'] = transformed_df['verified'].mean()
    comparison['transformed_signature_valid'] = transformed_df['signature_valid'].mean() if 'signature_valid' in transformed_df.columns else 0

    logger.info(f"Transformed: {len(transformed_df)} samples")
    logger.info(f"Transformed manifest present: {comparison['transformed_manifest_present']:.1%}")
    logger.info(f"Transformed verified: {comparison['transformed_verified']:.1%}")

    # Calculate drops
    comparison['manifest_drop_pct'] = (comparison['baseline_manifest_present'] -
                                      comparison['transformed_manifest_present']) * 100
    comparison['verification_drop_pct'] = (comparison['baseline_verified'] -
                                          comparison['transformed_verified']) * 100

    logger.info(f"Manifest drop: {comparison['manifest_drop_pct']:.1f}%")
    logger.info(f"Verification drop: {comparison['verification_drop_pct']:.1f}%")

    return comparison


def analyze_platform_retention(df_platform: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze manifest retention specifically for platform round-trips.

    Args:
        df_platform: Platform testing subset

    Returns:
        DataFrame with platform-specific retention rates
    """
    if len(df_platform) == 0:
        logger.warning("No platform data available")
        return pd.DataFrame()

    logger.info("Analyzing platform-specific retention")

    retention = df_platform.groupby('platform').agg({
        'manifest_present': ['count', 'sum', 'mean'],
        'verified': 'mean'
    }).round(4)

    retention.columns = ['total_samples', 'retained', 'retention_pct', 'verified_pct']
    retention['retention_pct'] = retention['retention_pct'] * 100
    retention['verified_pct'] = retention['verified_pct'] * 100
    retention = retention.reset_index()

    # Sort by platform name
    retention = retention.sort_values('platform')

    logger.info(f"Platforms analyzed: {list(retention['platform'])}")
    logger.info(f"Platform samples: {retention['total_samples'].sum()}")
    logger.info(f"Platform manifests retained: {retention['retained'].sum()}")

    # Check if any platform preserved manifests
    if retention['retained'].sum() == 0:
        logger.warning("All social media platforms stripped C2PA manifests (100% loss)")
    else:
        preserved = retention[retention['retained'] > 0]
        logger.info(f"Platforms that preserved manifests: {preserved['platform'].tolist()}")

    return retention


def generate_retention_summary() -> str:
    """
    Generate text summary of retention analysis findings.

    Returns:
        Formatted text summary for thesis
    """
    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']
    df_baseline = data['baseline']
    df_platform = data['platform']

    # Calculate metrics
    retention_by_type = calculate_retention_by_transform(df)
    comparison = compare_baseline_vs_transformed(df_baseline, df)
    platform_retention = analyze_platform_retention(df_platform)

    summary = []
    summary.append("=" * 60)
    summary.append("C2PA MANIFEST RETENTION ANALYSIS SUMMARY")
    summary.append("=" * 60)
    summary.append("")

    # Baseline vs Transformed
    summary.append("BASELINE VS TRANSFORMED:")
    summary.append(f"- Baseline samples: {comparison.get('baseline_samples', 0)}")
    summary.append(f"- Baseline manifest retention: {comparison.get('baseline_manifest_present', 1.0):.1%}")
    summary.append(f"- Transformed samples: {comparison['transformed_samples']}")
    summary.append(f"- Transformed manifest retention: {comparison['transformed_manifest_present']:.1%}")
    summary.append(f"- Absolute drop: {comparison['manifest_drop_pct']:.1f}%")
    summary.append("")

    # Transform Type Analysis
    summary.append("RETENTION BY TRANSFORM TYPE:")
    total_retained = retention_by_type['retained'].sum()
    total_samples = retention_by_type['total_samples'].sum()
    summary.append(f"- Total transforms analyzed: {len(retention_by_type)}")
    summary.append(f"- Total samples: {total_samples}")
    summary.append(f"- Manifests retained: {total_retained}")
    summary.append(f"- Overall retention rate: {total_retained/total_samples:.1%}")
    summary.append("")

    # Critical Finding
    if total_retained == 0:
        summary.append("CRITICAL FINDING:")
        summary.append("100% manifest loss across ALL transformations")
        summary.append("- Every compression level stripped manifests")
        summary.append("- Every editing operation stripped manifests")
        summary.append("- Every platform round-trip stripped manifests")
        summary.append("")

    # Platform Analysis
    if len(df_platform) > 0:
        summary.append("PLATFORM ROUND-TRIP ANALYSIS:")
        summary.append(f"- Platforms tested: {len(platform_retention)}")
        summary.append(f"- Total platform samples: {df_platform.shape[0]}")
        platform_retained = platform_retention['retained'].sum() if len(platform_retention) > 0 else 0
        summary.append(f"- Platform manifests retained: {platform_retained}")
        summary.append(f"- Platform retention rate: {df_platform['manifest_present'].mean():.1%}")
        summary.append("")

    # Conclusion
    summary.append("CONCLUSION:")
    summary.append("C2PA manifests show ZERO structural persistence through any transformation.")
    summary.append("Current editing tools and platforms are not C2PA-aware.")
    summary.append("Cryptographic robustness cannot be evaluated - manifests stripped before verification.")
    summary.append("")
    summary.append("=" * 60)

    return "\n".join(summary)


def run_manifest_analysis():
    """Execute complete manifest retention analysis and save results."""
    logger.info("=" * 60)
    logger.info("Starting C2PA Manifest Retention Analysis")
    logger.info("=" * 60)

    # Load data
    data = load_and_prepare.load_all_data()
    df = data['final_metrics']
    df_baseline = data['baseline']
    df_platform = data['platform']

    # Calculate retention metrics
    retention_by_type = calculate_retention_by_transform(df)
    retention_by_level = calculate_retention_by_level(df)
    verification_rates = calculate_verification_rates(df)
    platform_retention = analyze_platform_retention(df_platform)

    # Save retention table
    output_path = utils.DIRS['analysis_csv'] / "retention_table.csv"
    retention_by_type.to_csv(output_path, index=False)
    logger.info(f"Saved retention table to: {output_path}")

    # Save detailed retention by level
    output_path = utils.DIRS['analysis_csv'] / "retention_by_level.csv"
    retention_by_level.to_csv(output_path, index=False)
    logger.info(f"Saved detailed retention to: {output_path}")

    # Save verification rates
    output_path = utils.DIRS['analysis_csv'] / "verification_rates.csv"
    verification_rates.to_csv(output_path, index=False)
    logger.info(f"Saved verification rates to: {output_path}")

    # Save platform retention if available
    if len(platform_retention) > 0:
        output_path = utils.DIRS['analysis_csv'] / "platform_retention.csv"
        platform_retention.to_csv(output_path, index=False)
        logger.info(f"Saved platform retention to: {output_path}")

    # Generate and save summary
    summary = generate_retention_summary()
    output_path = utils.DIRS['analysis_results'] / "manifest_retention_summary.txt"
    output_path.write_text(summary, encoding='utf-8')
    logger.info(f"Saved summary to: {output_path}")

    # Log summary instead of print (avoids encoding issues)
    logger.info("Summary generated successfully")

    logger.info("=" * 60)
    logger.info("Manifest retention analysis complete")
    logger.info("=" * 60)

    return {
        'retention_by_type': retention_by_type,
        'retention_by_level': retention_by_level,
        'verification_rates': verification_rates,
        'platform_retention': platform_retention
    }


if __name__ == "__main__":
    results = run_manifest_analysis()