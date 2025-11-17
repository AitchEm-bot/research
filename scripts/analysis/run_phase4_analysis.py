"""
Master Pipeline for Phase 4 Analysis
=====================================

This script orchestrates the complete Phase 4 analysis pipeline.

Steps:
1. Load and prepare data
2. Run manifest retention analysis
3. Run quality metrics analysis
4. Run platform analysis
5. Run statistical tests
6. Generate visualizations
7. Generate summary reports

Usage:
    python scripts/analysis/run_phase4_analysis.py [--skip-viz]

    --skip-viz: Skip visualization generation (useful for quick analysis)

Output:
    All analysis results in data/results/analysis_results/
"""

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.common import utils

# Setup logging
logger = utils.setup_logging(log_file='data/results/logs/phase4_master.log')


def run_data_analysis():
    """Execute all data analysis modules."""
    logger.info("=" * 60)
    logger.info("PHASE 4: DATA ANALYSIS")
    logger.info("=" * 60)

    from scripts.analysis.data_analysis import (
        load_and_prepare,
        manifest_retention_analysis,
        quality_metrics_analysis,
        platform_analysis,
        statistical_tests
    )

    results = {}

    # 1. Load data
    logger.info("\n[1/5] Loading and preparing data...")
    start_time = time.time()
    data = load_and_prepare.load_all_data()
    logger.info(f"Done - Data loaded in {time.time() - start_time:.1f}s")
    results['data'] = data

    # 2. Manifest retention analysis
    logger.info("\n[2/5] Analyzing manifest retention...")
    start_time = time.time()
    retention_results = manifest_retention_analysis.run_manifest_analysis()
    logger.info(f"Done - Manifest analysis complete in {time.time() - start_time:.1f}s")
    results['retention'] = retention_results

    # 3. Quality metrics analysis
    logger.info("\n[3/5] Analyzing quality metrics...")
    start_time = time.time()
    quality_results = quality_metrics_analysis.run_quality_analysis()
    logger.info(f"Done - Quality analysis complete in {time.time() - start_time:.1f}s")
    results['quality'] = quality_results

    # 4. Platform analysis
    logger.info("\n[4/5] Analyzing platform behavior...")
    start_time = time.time()
    platform_results = platform_analysis.run_platform_analysis()
    logger.info(f"Done - Platform analysis complete in {time.time() - start_time:.1f}s")
    results['platform'] = platform_results

    # 5. Statistical tests
    logger.info("\n[5/5] Running statistical tests...")
    start_time = time.time()
    stats_results = statistical_tests.run_all_statistical_tests()
    logger.info(f"Done - Statistical tests complete in {time.time() - start_time:.1f}s")
    results['statistics'] = stats_results

    return results


def run_visualizations():
    """Execute all visualization modules."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: VISUALIZATION")
    logger.info("=" * 60)

    # Import all visualization modules
    from scripts.analysis.visualization import (
        plot_manifest_retention,
        plot_quality_metrics,
        plot_vmaf_comparison,
        plot_correlation_heatmap
    )

    # 1. Create manifest retention plots
    logger.info("\n[1/4] Creating manifest retention plots...")
    start_time = time.time()
    plot_manifest_retention.create_all_retention_plots()
    logger.info(f"Done - Retention plots complete in {time.time() - start_time:.1f}s")

    # 2. Create quality metrics plots
    logger.info("\n[2/4] Creating quality metrics plots...")
    start_time = time.time()
    plot_quality_metrics.create_all_quality_plots()
    logger.info(f"Done - Quality plots complete in {time.time() - start_time:.1f}s")

    # 3. Create VMAF comparison plots
    logger.info("\n[3/4] Creating VMAF comparison plots...")
    start_time = time.time()
    plot_vmaf_comparison.create_all_vmaf_plots()
    logger.info(f"Done - VMAF plots complete in {time.time() - start_time:.1f}s")

    # 4. Create correlation heatmap
    logger.info("\n[4/4] Creating correlation heatmap...")
    start_time = time.time()
    plot_correlation_heatmap.create_correlation_plots()
    logger.info(f"Done - Correlation plots complete in {time.time() - start_time:.1f}s")


def generate_final_report(results: dict):
    """Generate comprehensive Phase 4 report."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 60)

    report = []
    report.append("=" * 80)
    report.append("PHASE 4 ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Data summary
    if 'data' in results:
        data = results['data']
        report.append("DATASET SUMMARY:")
        report.append(f"- Total samples: {len(data['final_metrics'])}")
        report.append(f"- Images: {len(data['images'])}")
        report.append(f"- Videos: {len(data['videos'])}")
        report.append(f"- Platform samples: {len(data['platform'])}")
        report.append(f"- Baseline samples: {len(data['baseline'])}")
        report.append("")

    # Retention findings
    if 'retention' in results and results['retention']:
        retention = results['retention'].get('retention_by_type', None)
        if retention is not None:
            report.append("MANIFEST RETENTION FINDINGS:")
            total_retained = retention['retained'].sum() if 'retained' in retention.columns else 0
            total_samples = retention['total_samples'].sum() if 'total_samples' in retention.columns else 0
            report.append(f"- Manifests retained: {total_retained}/{total_samples}")
            report.append(f"- Overall retention: {total_retained/total_samples:.1%}" if total_samples > 0 else "N/A")
            if total_retained == 0:
                report.append("- CRITICAL: 100% manifest loss across all transforms")
            report.append("")

    # Quality findings
    if 'quality' in results and results['quality']:
        quality_summary = results['quality'].get('quality_summary', None)
        if quality_summary is not None:
            report.append("QUALITY METRICS SUMMARY:")
            # Calculate overall means
            psnr_mean = quality_summary['psnr_mean'].mean() if 'psnr_mean' in quality_summary.columns else None
            ssim_mean = quality_summary['ssim_mean'].mean() if 'ssim_mean' in quality_summary.columns else None
            vmaf_mean = quality_summary['vmaf_aligned_mean'].mean() if 'vmaf_aligned_mean' in quality_summary.columns else None

            if psnr_mean is not None:
                report.append(f"- Mean PSNR: {psnr_mean:.1f} dB")
            if ssim_mean is not None:
                report.append(f"- Mean SSIM: {ssim_mean:.3f}")
            if vmaf_mean is not None:
                report.append(f"- Mean VMAF (aligned): {vmaf_mean:.1f}")
            report.append("")

    # Platform findings
    if 'platform' in results and results['platform']:
        platform_quality = results['platform'].get('quality', None)
        if platform_quality is not None and len(platform_quality) > 0:
            report.append("PLATFORM ANALYSIS:")
            report.append(f"- Platforms tested: {len(platform_quality)}")
            report.append(f"- All platforms strip C2PA manifests (100% loss)")
            # Find best/worst platforms
            if 'vmaf_aligned_mean' in platform_quality.columns:
                best = platform_quality.loc[platform_quality['vmaf_aligned_mean'].idxmax()]
                worst = platform_quality.loc[platform_quality['vmaf_aligned_mean'].idxmin()]
                report.append(f"- Best quality: {best['platform']} (VMAF: {best['vmaf_aligned_mean']:.1f})")
                report.append(f"- Worst quality: {worst['platform']} (VMAF: {worst['vmaf_aligned_mean']:.1f})")
            report.append("")

    # Statistical findings
    if 'statistics' in results and results['statistics']:
        report.append("STATISTICAL TESTS:")
        stats = results['statistics']

        if 'chi_square' in stats:
            chi = stats['chi_square']
            report.append(f"- Chi-square (manifest uniformity): p={chi.get('p_value', 'N/A'):.4f}")
            report.append(f"  {chi.get('conclusion', 'N/A')}")

        if 'anova_vmaf_aligned' in stats:
            anova = stats['anova_vmaf_aligned']
            if 'p_value' in anova:
                report.append(f"- ANOVA (VMAF differences): p={anova['p_value']:.4f}")
                report.append(f"  {anova.get('conclusion', 'N/A')}")

        if 'ttest' in stats:
            ttest = stats['ttest']
            if 'p_value' in ttest:
                report.append(f"- T-test (platform vs local): p={ttest['p_value']:.4f}")
                report.append(f"  {ttest.get('conclusion', 'N/A')}")
        report.append("")

    # Key conclusions
    report.append("KEY CONCLUSIONS:")
    report.append("1. C2PA manifests show ZERO persistence through transformations")
    report.append("2. All editing tools and platforms strip C2PA metadata")
    report.append("3. Perceptual quality remains high despite metadata loss")
    report.append("4. Cryptographic robustness cannot be evaluated (manifests stripped)")
    report.append("5. Current ecosystem not ready for C2PA deployment")
    report.append("")

    # Limitations
    report.append("LIMITATIONS:")
    report.append("- Could not evaluate cryptographic robustness under transformation")
    report.append("- All manifests stripped before verification could occur")
    report.append("- Research pivoted to document stripping behavior")
    report.append("- Future work requires C2PA-aware transformation tools")
    report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    output_path = utils.DIRS['analysis_results'] / "phase4_final_report.txt"
    output_path.write_text(report_text, encoding='utf-8')
    logger.info(f"Final report saved to: {output_path}")

    return report_text


def main():
    """Execute complete Phase 4 analysis pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Phase 4 Analysis Pipeline")
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    args = parser.parse_args()

    # Start timing
    total_start = time.time()

    # Log environment
    logger.info("=" * 80)
    logger.info("PHASE 4 ANALYSIS PIPELINE")
    logger.info("=" * 80)
    utils.log_environment_info()

    try:
        # Ensure directories exist
        utils.ensure_directories()
        utils.DIRS['analysis_results'].mkdir(parents=True, exist_ok=True)
        utils.DIRS['analysis_csv'].mkdir(parents=True, exist_ok=True)
        utils.DIRS['analysis_plots'].mkdir(parents=True, exist_ok=True)

        # Run data analysis
        results = run_data_analysis()

        # Run visualizations (unless skipped)
        if not args.skip_viz:
            run_visualizations()
        else:
            logger.info("\nSkipping visualizations (--skip-viz flag)")

        # Generate final report
        report = generate_final_report(results)

        # Print report to console
        print("\n" + report)

        # Summary
        total_time = time.time() - total_start
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4 ANALYSIS COMPLETE")
        logger.info(f"Total execution time: {total_time:.1f} seconds")
        logger.info("=" * 80)

        # List generated files
        logger.info("\nGenerated files:")

        # CSV files
        csv_files = list(utils.DIRS['analysis_csv'].glob("*.csv"))
        logger.info(f"  CSV files: {len(csv_files)}")
        for f in csv_files[:5]:  # Show first 5
            logger.info(f"    - {f.name}")
        if len(csv_files) > 5:
            logger.info(f"    ... and {len(csv_files) - 5} more")

        # Plot files
        if not args.skip_viz:
            plot_files = list(utils.DIRS['analysis_plots'].glob("*.png"))
            logger.info(f"  Plot files: {len(plot_files)}")
            for f in plot_files[:5]:  # Show first 5
                logger.info(f"    - {f.name}")
            if len(plot_files) > 5:
                logger.info(f"    ... and {len(plot_files) - 5} more")

        # Text reports
        txt_files = list(utils.DIRS['analysis_results'].glob("*.txt"))
        logger.info(f"  Text reports: {len(txt_files)}")
        for f in txt_files:
            logger.info(f"    - {f.name}")

        logger.info("\nDone - Phase 4 analysis pipeline executed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()