# Is C2PA's Metadata Robust in AI-Generated Content?

## Overview

This project implements an end-to-end, reproducible research pipeline to test the robustness of C2PA (Coalition for Content Provenance and Authenticity) manifests embedded in AI-generated images and videos under various compression and editing transformations.

**Research Question**: How well do C2PA content credentials survive real-world transformations such as JPEG compression, video re-encoding, and multi-generation copying?

## Project Status

Current Phase: **Initial Scaffold - Generation & Embedding**

- ✅ Image generation (Stable Diffusion via diffusers)
- ✅ Video generation (procedural frame-based, placeholder for diffusion models)
- ✅ C2PA manifest embedding (shim implementation, ready for c2pa-python)
- ⏳ Transformations (compression, re-encoding) - Next phase
- ⏳ Verification & metrics collection - Next phase
- ⏳ Analysis & reporting - Next phase

## Project Structure

```
research/
├── data/
│   ├── raw_images/          # Generated AI images (10 samples)
│   ├── raw_videos/          # Generated AI videos (2 samples)
│   ├── manifests/           # C2PA manifest JSON files
│   ├── transformed/         # Assets after compression/editing
│   └── metrics/             # Collected metrics data
├── scripts/
│   ├── generation/
│   │   ├── generate_images.py   # Stable Diffusion image generation
│   │   └── generate_videos.py   # Procedural video generation
│   ├── embedding/
│   │   └── embed_c2pa.py        # C2PA manifest creation
│   ├── transformations/         # (Next phase)
│   ├── verification/            # (Next phase)
│   └── analysis/                # (Next phase)
├── results/
│   ├── csv/                 # Metrics CSV files
│   ├── plots/               # Visualization plots
│   └── logs/                # Execution logs
├── CLAUDE.md                # Project constraints & memory
├── system-prompt.txt        # Technical specifications
└── README.md                # This file
```

## Dependencies

### System Requirements
- **Python**: >= 3.12 (tested with 3.12.6)
- **CUDA GPU**: NVIDIA GPU with CUDA 12.1+ support (tested on RTX 4060 Laptop with 8GB VRAM)
- **ffmpeg**: For video operations (install via system package manager or winget)
- **OS**: Windows 10/11, Linux (Ubuntu/WSL2)

### Installation

#### Option 1: Using requirements.txt (Recommended)

```bash
# Install all dependencies with CUDA 12.1 support
pip install -r requirements.txt
```

#### Option 2: Manual Installation

```bash
# Core deep learning (CUDA 12.1 build for RTX 4060 and newer GPUs)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Generative AI models
pip install diffusers==0.31.0 transformers==4.47.1 accelerate==1.2.1

# Memory optimization for 8GB VRAM (optional but recommended)
pip install xformers==0.0.28.post2

# Image/video processing
pip install opencv-python==4.10.0.84 Pillow==11.0.0 ffmpeg-python==0.2.0

# Scientific computing & analysis
pip install numpy==2.2.1 pandas==2.2.3 scikit-image==0.24.0
pip install matplotlib==3.9.3 seaborn==0.13.2

# CLI utilities
pip install typer==0.15.1 python-dotenv==1.0.1 tqdm==4.67.1

# C2PA (optional - using fallback shim if unavailable)
# pip install c2pa-python==0.4.2
```

**Notes**:
- The `c2pa-python` library is optional. Scripts use a fallback shim if unavailable.
- `xformers` provides memory-efficient attention for 8GB VRAM GPUs
- CUDA 12.1 build is compatible with CUDA 12.x drivers (12.1-12.9)

### Windows-Specific Setup

#### Installing ffmpeg on Windows

```bash
# Using winget (Windows 11)
winget install ffmpeg

# Or download from: https://www.gyan.dev/ffmpeg/builds/
# Add to PATH after extraction
```

#### GPU Memory Optimization for RTX 4060 (8GB VRAM)

The scripts automatically enable memory optimizations for GPUs with ≤8GB VRAM:
- Half-precision (FP16) inference
- Attention slicing
- Model offloading to CPU when needed

To force CPU-only mode (if GPU unavailable):
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/WSL
set CUDA_VISIBLE_DEVICES=       # Windows CMD
```

## Quick Start - Smoke Test

Follow these steps to run the smoke test and generate the initial test dataset:

### Step 1: Generate Images

Generate 10 deterministic 512×512 images using Stable Diffusion:

```bash
python scripts/generation/generate_images.py \
    --output-dir data/raw_images \
    --count 10 \
    --seed 42 \
    --resolution 512
```

**Expected output**:
- 10 PNG images in `data/raw_images/`
- Naming pattern: `img_000_seed42_TIMESTAMP.png` through `img_009_seed51_TIMESTAMP.png`
- 10 JSON metadata files (one per image)
- Total time: ~5-15 minutes (depending on GPU)

**Options**:
- `--seed`: Random seed for reproducibility (default: 42)
- `--count`: Number of images to generate (default: 10)
- `--resolution`: Image size in pixels (default: 512)

### Step 2: Generate Videos

Generate 2 short low-resolution test videos (3 seconds each):

```bash
python scripts/generation/generate_videos.py \
    --output-dir data/raw_videos \
    --count 2 \
    --seed 42 \
    --duration 3.0 \
    --fps 10 \
    --width 256 \
    --height 256
```

**Expected output**:
- 2 MP4 videos in `data/raw_videos/`
- Naming pattern: `video_000_seed42_TIMESTAMP.mp4`, `video_001_seed43_TIMESTAMP.mp4`
- 2 JSON metadata files (one per video)
- Total time: ~10-30 seconds

**Options**:
- `--seed`: Random seed for reproducibility (default: 42)
- `--count`: Number of videos to generate (default: 2)
- `--duration`: Video length in seconds (default: 3.0)
- `--fps`: Frames per second (default: 10)
- `--width` / `--height`: Video resolution (default: 256×256)

### Step 3: Create C2PA Manifests

Create C2PA content credentials for all generated images and videos:

```bash
python scripts/embedding/embed_c2pa.py \
    --images-dir data/raw_images \
    --videos-dir data/raw_videos \
    --output-dir data/manifests
```

**Expected output**:
- Manifest JSON files in `data/manifests/`
- One manifest per asset (12 total: 10 images + 2 videos)
- Naming pattern: `img_000_seed42_TIMESTAMP_manifest.json`
- Test keypair placeholders in `data/manifests/test_keys/`

**Notes**:
- By default, uses fallback shim (creates JSON manifests without real signing)
- To use real C2PA signing (when c2pa-python is available), add `--use-real-c2pa`
- Manifests include: creation timestamp, seed, model info, assertions, and signature placeholders

### Verify Smoke Test Success

Check that all expected files were created:

**Linux/WSL:**
```bash
# Count generated files
ls data/raw_images/*.png | wc -l    # Should show: 10
ls data/raw_videos/*.mp4 | wc -l    # Should show: 2
ls data/manifests/*_manifest.json | wc -l  # Should show: 12

# Inspect a sample image
file data/raw_images/img_000_seed42_*.png

# Inspect a sample video
ffmpeg -i data/raw_videos/video_000_seed42_*.mp4

# View a sample manifest
cat data/manifests/img_000_seed42_*_manifest.json | head -30
```

**Windows (PowerShell):**
```powershell
# Count generated files
(Get-ChildItem data\raw_images\*.png).Count    # Should show: 10
(Get-ChildItem data\raw_videos\*.mp4).Count    # Should show: 2
(Get-ChildItem data\manifests\*_manifest.json).Count  # Should show: 12

# Inspect a sample video
ffmpeg -i data\raw_videos\video_000_seed42_*.mp4

# View a sample manifest (first 30 lines)
Get-Content data\manifests\img_000_seed42_*_manifest.json | Select-Object -First 30
```

## Technical Details

### Image Generation

- **Model**: Stable Diffusion v1.4 (CompVis)
- **Citation**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022
- **Resolution**: 512×512 pixels
- **Format**: PNG (lossless)
- **Determinism**: Seeded with `--seed` + image index

### Video Generation

- **Current method**: Procedural frame-based generation (OpenCV)
- **Placeholder for**: Stable Video Diffusion or similar diffusion models
- **Resolution**: 256×256 pixels (low-res for fast testing)
- **Format**: MP4 (H.264 codec)
- **Duration**: 2-4 seconds at 10 fps (~30 frames)
- **Determinism**: Seeded with `--seed` + video index

### C2PA Manifests

- **Specification**: C2PA Technical Specification v1.0+ (industry standard)
- **Library**: c2pa-python (with fallback shim if unavailable)
- **Manifest contents**:
  - Claim generator info
  - Creation timestamp
  - Actions assertion (c2pa.created)
  - Hash assertion (placeholder)
  - Signature info (test keys only)
  - Asset metadata (seed, model version, etc.)

## External Video Support (Optional)

The pipeline supports testing external AI-generated videos from platforms like Sora 2, Runway, Pika, etc.

### Adding External Videos

1. **Place videos** in `data/raw_out_videos/`
   - Supported formats: .mp4, .mov, .avi
   - Minimum resolution: 256×256 pixels
   - Minimum duration: 1 second
   - Maximum file size: 500 MB (recommended)

2. **Prepare videos** for testing:
   ```bash
   python scripts/external/prepare_external_videos.py
   ```

   This script will:
   - Check if videos already have C2PA manifests
   - If signed: preserve original manifest
   - If unsigned: sign with test certificate
   - Move to `data/manifests/videos/external/`

3. **Automatic integration**
   - External videos are automatically included in transformation pipeline
   - Results are merged into `final_metrics.csv` with video_source tracking
   - Enables comparative analysis between different AI platforms

### Supported External Sources

- OpenAI Sora 2
- Runway Gen-3
- Pika Labs
- Synthesia
- Any other AI video generation platform

See `data/raw_out_videos/README.md` for detailed instructions.

## Phase 2.5: Social Media Platform Testing

Test C2PA manifest persistence through real-world social media platforms.

### Supported Platforms

- **Instagram**: video, image, story, reel
- **Twitter**: video, image
- **Facebook**: video, image
- **YouTube Shorts**: video
- **TikTok**: video
- **WhatsApp**: video, image, status

### Workflow

1. **Prepare uploads**:
   ```bash
   python scripts/platform/prepare_platform_uploads.py
   ```
   - Interactive menu to select assets
   - Choose platform and upload mode
   - Verifies C2PA signature before upload
   - Generates upload instructions

2. **Manual upload**:
   - Upload assets to platform via mobile/web app
   - No filters or editing applied

3. **Manual download**:
   - Download assets from platform
   - Save to `data/platform_tests/{platform}/returned/`
   - Follow naming convention: `{original}__{platform}__{mode}__{timestamp}.{ext}`

4. **Log metadata**:
   - Record upload/download timestamps in `data/platform_tests/platform_manifest.csv`
   - Use provided CSV template

5. **Process returns**:
   ```bash
   python scripts/platform/process_platform_returns.py
   ```
   - Verifies C2PA signatures
   - Calculates quality metrics (PSNR, SSIM, VMAF)
   - Generates `data/results/platform_results.csv`

6. **Merge with final results**:
   ```bash
   python scripts/metrics/merge_results.py
   ```
   - Appends platform results to `final_metrics.csv`

### Expected Outcomes

- Most platforms STRIP C2PA manifests during transcoding
- Quality degradation varies by platform compression policies
- Results inform real-world C2PA persistence analysis

See `data/platform_tests/README_PHASE_2.5.md` for detailed instructions.

## Next Steps

After completing the smoke test, the following phases will be implemented:

1. **Transformations** (Phase 2)
   - JPEG compression at various quality levels
   - Video re-encoding (H.264, H.265, VP9)
   - Image editing operations (resize, crop, rotate)
   - Multi-generation re-encoding

2. **Verification** (Phase 3)
   - C2PA manifest verification
   - Cryptographic signature validation
   - Hash integrity checks
   - Metrics: VSR, MRR, SVR, HIM, MCR

3. **Analysis** (Phase 4)
   - Perceptual quality metrics (PSNR, SSIM, VMAF)
   - Aggregation of results to CSV
   - Statistical analysis and plotting
   - HTML report generation

4. **Docker & Reproducibility** (Phase 5)
   - Complete Dockerfile with CUDA support
   - Automated pipeline execution
   - CI/CD integration

## Ethics & Safety

This research pipeline is designed for legitimate provenance testing. Please observe the following:

- ✅ **Do**: Use for testing C2PA robustness with synthetic content
- ✅ **Do**: Generate abstract, non-person content for testing
- ❌ **Don't**: Generate synthetic media of real persons without consent
- ❌ **Don't**: Use for malicious deepfakes or misinformation
- ❌ **Don't**: Bypass authentication or violate platform ToS

## References

- **Stable Diffusion**: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models," CVPR 2022. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- **C2PA Specification**: Coalition for Content Provenance and Authenticity. [c2pa.org/specifications](https://c2pa.org/specifications/)
- **c2pa-python**: Python bindings for C2PA. [github.com/contentauth/c2pa-python](https://github.com/contentauth/c2pa-python)

## License

This is a research project. See individual library licenses for dependencies.

## Contact

Project Lead: AitchEm

For questions or issues with this pipeline, please refer to the project documentation or contact the project lead.
