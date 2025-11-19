# Is C2PA's Metadata Robust in AI-Generated Content?

## Overview

This project implements an end-to-end, reproducible research pipeline to test the robustness of C2PA (Coalition for Content Provenance and Authenticity) manifests embedded in AI-generated images and videos under various compression and editing transformations.

**Research Question**: How well do C2PA content credentials survive real-world transformations such as JPEG compression, video re-encoding, platform round-trips, and multi-generation copying?

## Project Status

**Current Phase**: Phase 4 - Analysis & Visualization ✅

Pipeline completion status:
- ✅ Phase 1: Generation & C2PA Embedding (100 images, 110 videos)
- ✅ Phase 2: Transformations & Compression (~3,460 transformed assets)
- ✅ Phase 2.5: Social Media Round-Trip Testing (160 platform samples)
- ✅ Phase 3: Verification & Metric Computation (final_metrics.csv generated)
- ✅ Phase 4: Data Analysis & Visualization

## Project Structure

```
research/
├── data/
│   ├── assets/                      # Raw generated/external assets
│   │   ├── raw_images/              # 100 images (SD1.4, 1024×1024)
│   │   ├── raw_videos/              # Internal videos (SVD)
│   │   ├── raw_images_for_videos/   # Conditioning images for SVD
│   │   └── raw_out_videos/          # 60 external videos (Veo3.1)
│   ├── prepared_assets/             # Processed assets ready for testing
│   │   ├── signed_assets/           # C2PA signed assets (210 total)
│   │   │   ├── images/              # 100 signed images
│   │   │   └── videos/
│   │   │       ├── internal/        # 50 signed internal videos (SVD)
│   │   │       └── external/        # 60 signed external videos (Veo3.1)
│   │   ├── c2pa_manifests/          # Extracted C2PA manifest JSONs
│   │   ├── transformed/             # ~3,460 transformed assets
│   │   │   ├── compression/
│   │   │   │   ├── images/          # JPEG q95/q75/q50/q25, PNG c9/c0
│   │   │   │   └── videos/          # H.264/H.265 bitrates, FPS adjustments
│   │   │   └── editing/
│   │   │       ├── images/          # resize, crop, rotate, brightness, etc.
│   │   │       └── videos/          # resize, crop, trim, brightness, etc.
│   │   └── platform_tests/          # Phase 2.5 social media testing
│   │       ├── instagram/
│   │       ├── twitter/
│   │       ├── facebook/
│   │       ├── youtube/
│   │       ├── tiktok/
│   │       └── auto_sample_tracking.csv
│   └── results/                     # All outputs (CSV files and logs)
│       ├── c2pa_validation.csv      # C2PA verification results
│       ├── quality_metrics.csv      # Quality metrics (PSNR/SSIM/VMAF)
│       ├── platform_results.csv     # Phase 2.5 platform testing results
│       ├── final_metrics.csv        # Merged comprehensive results (~3,620 rows)
│       └── logs/                    # All execution logs
├── scripts/
│   ├── common/                      # Shared utilities
│   │   └── utils.py                 # Centralized functions (logging, CSV, paths)
│   ├── c2pa/                        # C2PA operations
│   │   ├── embedding/               # C2PA manifest signing
│   │   │   ├── embed_c2pa_v2.py
│   │   │   └── extract_manifests.py
│   │   └── verification/            # C2PA manifest verification
│   │       ├── verify_c2pa.py
│   │       └── verify_original_manifests.py
│   └── processing/                  # Data processing pipeline
│       ├── generation/              # Asset generation
│       │   ├── generate_images.py
│       │   ├── generate_videos.py
│       │   └── generate_video_images.py
│       ├── transformations/         # Compression and editing
│       │   ├── compress_images.py
│       │   ├── compress_videos.py
│       │   └── edit_assets.py
│       ├── metrics/                 # Quality metrics and result merging
│       │   ├── calculate_quality_metrics.py
│       │   └── merge_results.py
│       └── preprocessing/           # External assets and platform preparation
│           ├── external/            # External video preparation
│           │   └── prepare_external_videos.py
│           └── platform/            # Phase 2.5 platform testing
│               ├── prepare_platform_uploads.py
│               ├── process_platform_returns.py
│               ├── rename_platform_returns.py
│               └── rename_platform_uploads.py
├── CLAUDE.md                        # Project memory & agent constraints
├── FLOW_DIAGRAM.md                  # Pipeline visualization
└── README.md                        # This file
```

## Research Pipeline Phases

### PHASE 1 — Generation & C2PA Embedding

**Goal:** Generate AI-produced images and videos, then embed C2PA manifests.

**Internal Pipeline:**
- 100 images (Stable Diffusion v1.4, 1024×1024, seeds 42-141)
- 50 videos (Stable Video Diffusion, image-to-video)
- All assets signed with c2patool (built-in test certificate)

**External Videos:**
- 60 videos from Google Veo3.1
- Automatically signed during preparation
- Enables cross-platform AI comparison

**Deliverables:**
- ✅ 100 signed images in `data/prepared_assets/signed_assets/images/`
- ✅ 110 signed videos (50 internal + 60 external)
- ✅ Metadata preserved: seed, model version, generation prompts

---

### PHASE 2 — Transformations & Compression Testing

**Goal:** Apply controlled transformations to assess how content modifications affect C2PA metadata.

**Image Transformations:**
- JPEG compression (q95, q75, q50, q25)
- PNG compression (c0, c9 - lossless)
- Resize (75%, 50%, 25%)
- Crop (center 80%, 60%)
- Rotation (90°, 180°)
- Brightness adjustment (-40 to +40)
- Contrast adjustment
- Saturation adjustment

**Video Transformations:**
- H.264 re-encoding (5000k, 2000k, 500k bitrates)
- H.265 re-encoding (2000k, 500k bitrates)
- FPS adjustment (30fps, 10fps, 5fps, 3fps)
- Resize (75%, 50%)
- Crop (center 80%)
- Trim (first 50%, middle 50%)
- Brightness adjustment (-40 to +40)

**Deliverables:**
- ✅ ~3,460 transformed assets in `data/prepared_assets/transformed/`
- ✅ Comprehensive transformation coverage across both images and videos

---

### PHASE 2.5 — Social Media Round-Trip Testing

**Goal:** Test whether C2PA manifests survive after uploading and downloading from major social platforms.

**Platforms Tested:**
- Instagram (video, image, post) - 25 images + 10 videos
- Twitter/X (video, image, upload) - 25 images + 10 videos
- Facebook (video, image, post) - 25 images + 10 videos
- YouTube (video, upload) - 10 videos
- TikTok (video, upload) - 10 videos

**Workflow:**
1. Auto-sampled 160 assets (100 images + 60 videos)
2. Manual upload to platforms (via mobile/web apps)
3. Manual download using third-party tools (FastDL, Snaplytics, SnapTik)
4. Automated processing with C2PA verification + quality metrics

**Download Tools Used:**
- Instagram: FastDL (https://fastdl.app/en2)
- Twitter: Snaplytics (https://snaplytics.io/twitter-img-downloader/)
- TikTok: SnapTik (https://snaptik.cx/)
- Facebook/YouTube: Direct platform download

**Expected Outcomes:**
- ✅ Most platforms STRIPPED C2PA manifests (manifest_present = 0)
- ✅ Quality degradation documented via PSNR/SSIM/VMAF metrics
- ✅ Platform-specific compression characteristics analyzed

**Deliverables:**
- ✅ `data/results/platform_results.csv` (160 platform round-trip results)
- ✅ Integrated into `final_metrics.csv` with platform metadata

---

### PHASE 3 — Verification & Metric Computation

**Goal:** Validate C2PA manifests post-transformations and measure perceptual quality degradation.

**C2PA Verification Metrics:**
- manifest_present (0/1)
- verified (0/1) - INTEGRITY validation (claimSignature.validated)
- signature_valid, hash_match, assertion_uris_match (0/1)
- trust_verified (informational, not failure metric)
- validation_state, failure_reason (descriptive)

**Quality Metrics:**
- **Images**: PSNR, SSIM (stretched + aligned variants)
- **Videos**: VMAF (stretched + aligned variants, aspect ratio aware)
- **Alignment methods**: same_aspect_ratio, crop_reference_center_square, scale_both_to_minimum
- **Lossless detection**: lossless_match flag (PSNR >= 100 dB)

**Deliverables:**
- ✅ `data/results/c2pa_validation.csv` (~3,460 transformed + 160 platform)
- ✅ `data/results/quality_metrics.csv` (~3,460 transformed + 160 platform)
- ✅ `data/results/final_metrics.csv` (~3,620 total rows, 29 columns)

---

### PHASE 4 — Data Analysis & Visualization

**Goal:** Analyze correlations between visual quality degradation and metadata loss.

**Analysis Tasks:**
- VSR/SVR/HSR (Verification Success Rate, Signature Validity Rate, Hash Success Rate)
- Correlation analysis for PSNR/SSIM/VMAF vs Manifest Retention
- Distribution plots by transform type and platform
- Heatmaps for integrity loss patterns
- Transform impact visualization
- Platform-specific comparison charts

**Deliverables:**
- ✅ `data/results/analysis_results/csv/` - Statistical summaries (5 CSV files)
- ✅ `data/results/analysis_results/plots/` - 11+ publication-ready plots (PNG, 300 DPI)
- ✅ `data/results/analysis_results/report.html` - Interactive HTML dashboard

---

## Dependencies

### System Requirements
- **Python**: >= 3.12 (tested with 3.12.6)
- **CUDA GPU**: NVIDIA GPU with CUDA 12.1+ support (tested on RTX 4060 Laptop with 8GB VRAM)
- **ffmpeg**: For video operations (install via system package manager or winget)
- **c2patool**: C2PA command-line tool from contentauth/c2pa-rs
- **OS**: Windows 10/11, Linux (Ubuntu/WSL2)

### Installation

```bash
# Install all dependencies with CUDA 12.1 support
pip install -r requirements.txt

# Install c2patool (download from releases or use local build in tools/c2patool/)
```

**Notes:**
- CUDA 12.1 build is compatible with CUDA 12.x drivers (12.1-12.9)
- Scripts automatically enable memory optimizations for GPUs with ≤8GB VRAM
- c2patool 0.24.0+ required for proper C2PA manifest handling

### Windows-Specific Setup

```bash
# Install ffmpeg using winget (Windows 11)
winget install ffmpeg
```

## Quick Start

### Option 1: Quick Install (Recommended - Using Docker)

**For peer reviewers and researchers who want to reproduce results immediately:**

**Linux/macOS:**
```bash
curl -sSL https://raw.githubusercontent.com/AitchEm-bot/research/master/quick-install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/AitchEm-bot/research/master/quick-install.ps1 | iex
```

**Then run (after restarting terminal on Windows):**
```bash
# Quick test with preset assets (10-20 minutes)
c2pa test

# Full pipeline with preset assets (4-8 hours)
c2pa run

# Phase-by-phase execution
c2pa phase 0             # Asset generation/loading
c2pa phase 1             # C2PA embedding
c2pa phase 2             # Transformations
c2pa phase 3             # Verification & metrics
c2pa phase 4             # Analysis & visualization

# Check status
c2pa status

# View results
ls ./c2pa-results/
```

**What this does:**
- ✅ Installs Docker image (aitchem037/c2pa-research:latest)
- ✅ Sets up `c2pa` command-line wrapper
- ✅ Includes preset assets (10 images + 2 videos)
- ✅ Automatic GPU support and volume mounting
- ✅ Results appear in `./c2pa-results/`

**See [README_DOCKER.md](README_DOCKER.md) for complete Docker documentation.**

---

### Option 2: Manual Setup (For Development)

**For researchers who want to modify the pipeline or run without Docker:**

#### Step 1: Clone Repository

```bash
git clone https://github.com/AitchEm-bot/research.git
cd research
```

#### Step 2: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# OR
.venv\Scripts\activate  # Windows

# Install all dependencies with CUDA 12.1 support
pip install -r requirements.txt

# Install FFmpeg (system-wide)
# Linux: sudo apt install ffmpeg
# macOS: brew install ffmpeg
# Windows: winget install ffmpeg
```

#### Step 3: Install c2patool

Download from [contentauth/c2pa-rs releases](https://github.com/contentauth/c2pa-rs/releases) and place in `tools/c2patool/` or add to PATH.

#### Step 4: Run Pipeline

```bash
# Phase 1: Generate images (or use preset assets)
python scripts/processing/generation/generate_images.py \
    --seed 42 --count 100 --output-dir data/assets/raw_images

# Phase 1.5: Sign assets with C2PA
python scripts/c2pa/embedding/embed_c2pa_v2.py

# Phase 2: Run transformations
python scripts/processing/transformations/compress_images.py
python scripts/processing/transformations/compress_videos.py
python scripts/processing/transformations/edit_assets.py

# Phase 3: Verify C2PA & calculate metrics
python scripts/c2pa/verification/verify_c2pa.py
python scripts/processing/metrics/calculate_quality_metrics.py
python scripts/processing/metrics/merge_results.py

# Phase 4: Analysis & visualization
python scripts/analysis/run_phase4_analysis.py

# Phase 2.5 (Optional): Platform testing
python scripts/processing/preprocessing/platform/prepare_platform_uploads.py --auto-sample
# [Manual upload/download to social media platforms]
python scripts/processing/preprocessing/platform/process_platform_returns.py
```

#### Step 5: View Results

```bash
# Results are saved in:
# data/results/csv/final_metrics.csv - Complete dataset (~3,620 rows)
# data/results/analysis_results/plots/ - Visualization outputs
# data/results/analysis_results/report.html - Interactive dashboard
```

## Testing & Debugging

All scripts support `--test` flag for smoke testing:

```bash
# Test C2PA verification (4 sample assets)
python scripts/c2pa/verification/verify_c2pa.py --test

# Test quality metrics (4 sample assets)
python scripts/processing/metrics/calculate_quality_metrics.py --test

# Test platform processing (2 sample files from first platform)
python scripts/processing/preprocessing/platform/process_platform_returns.py --test
```

## Technical Details

### AI Models Used

**Image Generation:**
- Model: Stable Diffusion v1.4 (CompVis/stable-diffusion-v1-4)
- Paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022)
- Resolution: 1024×1024 pixels
- Dataset: 100 images with diverse prompts (seeds 42-141)

**Video Generation (Internal):**
- Model: Stable Video Diffusion (stabilityai/stable-video-diffusion-img2vid-xt)
- Paper: "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (Blattmann et al., arXiv 2311.15127)
- Status: PREPRINT (not yet peer-reviewed)
- Resolution: 512×512 pixels, 25 frames

**Video Generation (External):**
- Source: Google Veo3.1 (60 videos)
- Processing: Automatic C2PA signing via prepare_external_videos.py

### C2PA Implementation

- Uses c2patool (v0.24.0+) from contentauth/c2pa-rs
- Built-in ES256 test certificates for authentic cryptographic signatures
- Python scripts invoke c2patool via subprocess
- Verification uses INTEGRITY validation (claimSignature.validated + hash match)
- Trust validation is informational only (not failure metric)

### VMAF Alignment Methods

The pipeline uses aspect-ratio-aware VMAF calculation:
- **vmaf_stretched**: Traditional method (scales distorted to reference, may distort aspect)
- **vmaf_aligned**: Crops/scales reference to match distorted aspect ratio
- **alignment_method**: same_aspect_ratio, crop_reference_center_square, scale_both_to_minimum
- Platform transforms (Instagram 16:9→1:1 crop) benefit from aligned metrics

## Social Media Accounts (Phase 2.5)

Research accounts created for platform testing:

- **Instagram**: [@independant_researcher](https://www.instagram.com/independant_researcher/)
- **Twitter**: [@Independant_R](https://x.com/Independant_R)
- **Facebook**: [Hani Moustafa](https://www.facebook.com/profile.php?id=61583369476134)
- **YouTube**: [@IndependantResearcher](http://www.youtube.com/@IndependantResearcher)
- **TikTok**: [@independant_researcher](https://www.tiktok.com/@independant_researcher)

All accounts contain AI-generated content only (no personal data or real individuals).

## Ethics & Safety

This research pipeline is designed for legitimate provenance testing:

- ✅ **Do**: Use for testing C2PA robustness with synthetic content
- ✅ **Do**: Generate abstract, non-person content for testing
- ❌ **Don't**: Generate synthetic media of real persons without consent
- ❌ **Don't**: Use for malicious deepfakes or misinformation
- ❌ **Don't**: Bypass authentication or violate platform ToS

## References

- **Stable Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- **C2PA Specification**: Coalition for Content Provenance and Authenticity. [c2pa.org/specifications](https://c2pa.org/specifications/)
- **c2patool**: C2PA command-line tool. [github.com/contentauth/c2pa-rs](https://github.com/contentauth/c2pa-rs)

## License

This is a research project. See individual library licenses for dependencies.

## Contact

Project Lead: Hani Moustafa

For questions or issues with this pipeline, please refer to the project documentation (CLAUDE.md, FLOW_DIAGRAM.md) or contact the project lead.
