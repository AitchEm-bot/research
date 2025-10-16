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
- Python >= 3.10
- CUDA-capable GPU (recommended for image generation)
- ffmpeg (for video operations)

### Python Packages

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install diffusers transformers accelerate
pip install opencv-python Pillow numpy
pip install pandas matplotlib seaborn scikit-image
pip install c2pa-python  # Optional - scripts work with fallback shim
```

**Note**: The `c2pa-python` library is optional for initial testing. If not available, the embedding script will create manifest JSON files using a fallback shim.

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
