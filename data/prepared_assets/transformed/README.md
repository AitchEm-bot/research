# Transformed Assets

This directory contains C2PA-signed assets after undergoing compression and editing transformations for robustness testing.

## Purpose

Test **C2PA manifest persistence** under realistic transformation scenarios:

1. Load signed assets from `data/prepared_assets/signed_assets/`
2. Apply compression or editing transformations
3. Save transformed assets to this directory
4. Verify C2PA manifest retention
5. Calculate quality degradation metrics

## Directory Structure

```
transformed/
├── compression/
│   ├── images/
│   │   ├── jpeg/
│   │   │   ├── q95/          # JPEG quality 95
│   │   │   ├── q75/          # JPEG quality 75
│   │   │   ├── q50/          # JPEG quality 50
│   │   │   └── q25/          # JPEG quality 25
│   │   └── png/
│   │       ├── c9/           # PNG compression level 9 (maximum)
│   │       └── c0/           # PNG compression level 0 (none/store)
│   └── videos/
│       ├── h264/
│       │   ├── bitrate5000k/ # H.264 5000 kbps
│       │   ├── bitrate2000k/ # H.264 2000 kbps
│       │   └── bitrate500k/  # H.264 500 kbps
│       ├── h265/
│       │   ├── bitrate2000k/ # H.265 2000 kbps
│       │   └── bitrate500k/  # H.265 500 kbps
│       └── fps/
│           ├── fps5/         # Frame rate 5 fps
│           └── fps3/         # Frame rate 3 fps
└── editing/
    ├── images/
    │   ├── resize/
    │   │   ├── res512/       # 512×512
    │   │   ├── res256/       # 256×256
    │   │   └── res128/       # 128×128
    │   ├── crop/
    │   │   ├── crop80/       # 80% center crop
    │   │   ├── crop60/       # 60% center crop
    │   │   └── crop40/       # 40% center crop
    │   ├── rotate/
    │   │   ├── rot90/        # 90° clockwise
    │   │   ├── rot180/       # 180°
    │   │   └── rot270/       # 270° clockwise
    │   ├── brightness/
    │   │   ├── bright1.3/    # +30% brightness
    │   │   ├── bright1.1/    # +10% brightness
    │   │   ├── bright0.9/    # -10% brightness
    │   │   └── bright0.7/    # -30% brightness
    │   ├── contrast/
    │   │   ├── contrast1.3/  # +30% contrast
    │   │   ├── contrast1.1/  # +10% contrast
    │   │   ├── contrast0.9/  # -10% contrast
    │   │   └── contrast0.7/  # -30% contrast
    │   └── saturation/
    │       ├── sat1.3/       # +30% saturation
    │       ├── sat1.1/       # +10% saturation
    │       ├── sat0.9/       # -10% saturation
    │       └── sat0.7/       # -30% saturation
    └── videos/
        ├── resize/
        │   ├── res256/       # 256×256
        │   └── res128/       # 128×128
        ├── crop/
        │   ├── crop80/       # 80% center crop
        │   └── crop60/       # 60% center crop
        ├── trim/
        │   ├── trim_start5/  # Remove first 5 frames
        │   └── trim_end5/    # Remove last 5 frames
        ├── brightness/
        │   ├── bright1.2/    # +20% brightness
        │   └── bright0.8/    # -20% brightness
        ├── contrast/
        │   ├── contrast1.2/  # +20% contrast
        │   └── contrast0.8/  # -20% contrast
        └── saturation/
            ├── sat1.2/       # +20% saturation
            └── sat0.8/       # -20% saturation
```

## Transformation Categories

### 1. Compression Transforms

**JPEG (Images)**: Lossy compression at 4 quality levels
- **q95**: Minimal loss, nearly indistinguishable from original
- **q75**: Standard web quality
- **q50**: Moderate compression
- **q25**: High compression, visible artifacts

**PNG (Images)**: Lossless compression at 2 levels
- **c9**: Maximum compression (slowest)
- **c0**: No compression (fastest, largest files)
- **Note**: Both produce pixel-identical output (PSNR = ∞, SSIM = 1.0)

**H.264 (Videos)**: Most widely used video codec
- **5000 kbps**: High quality
- **2000 kbps**: Standard quality
- **500 kbps**: Low quality, compression artifacts visible

**H.265/HEVC (Videos)**: Modern efficient codec
- **2000 kbps**: High quality (better than H.264 at same bitrate)
- **500 kbps**: Standard quality

**FPS Adjustment (Videos)**: Frame rate reduction
- **5 fps**: Low frame rate, motion stuttering
- **3 fps**: Very low frame rate, slideshow effect

### 2. Editing Transforms

**Resize**: Downsampling to lower resolutions
- Creates aspect ratio changes
- Tests C2PA under resolution modification

**Crop**: Center crop at various percentages
- Removes portions of image/video
- Tests manifest survival when content removed

**Rotate**: Rotation by multiples of 90°
- Tests lossless geometric transformations
- May trigger re-encoding

**Brightness/Contrast/Saturation**: Color adjustments
- Multiplicative factors (1.0 = no change)
- Tests manifest under color space transformations

**Trim** (Videos only): Remove frames from start or end
- Tests manifest when temporal content modified

## Transform Execution

### Image Compression
```bash
python scripts/processing/transformations/compress_images.py
```

### Video Compression
```bash
python scripts/processing/transformations/compress_videos.py
```

### Editing Transforms
```bash
python scripts/processing/transformations/edit_assets.py
```

## File Naming Convention

Transformed files preserve original names with transform suffix:

**Images**:
```
img_042_seed42_20251113_233209_signed_jpeg_q75.jpg
img_042_seed42_20251113_233209_signed_resize_res256.png
```

**Videos**:
```
video_15_signed_h264_bitrate2000k.mp4
video_15_signed_crop_crop80.mp4
```

## Expected Dataset Size

**Images** (100 original × transforms):
- JPEG: 100 × 4 quality levels = 400 files
- PNG: 100 × 2 compression levels = 200 files
- Resize: 100 × 3 resolutions = 300 files
- Crop: 100 × 3 percentages = 300 files
- Rotate: 100 × 3 angles = 300 files
- Color adjustments: 100 × 3 types × 4 levels = 1,200 files
- **Total: ~2,700 transformed images**

**Videos** (110 original × transforms):
- H.264: 110 × 3 bitrates = 330 files
- H.265: 110 × 2 bitrates = 220 files
- FPS: 110 × 2 rates = 220 files
- Resize: 110 × 2 resolutions = 220 files
- Crop: 110 × 2 percentages = 220 files
- Trim: 110 × 2 types = 220 files
- Color adjustments: 110 × 3 types × 2 levels = 660 files
- **Total: ~2,090 transformed videos**

**Combined: ~4,790 transformed assets**

## Quality Metrics

After transformation, quality is measured using:

**Images**:
- **PSNR** (Peak Signal-to-Noise Ratio): dB, higher is better, ∞ for lossless
- **SSIM** (Structural Similarity Index): 0-1, 1.0 is perfect
- **Lossless match**: Boolean, exact pixel match (PSNR ≥ 100 dB)

**Videos**:
- **VMAF** (Video Multimethod Assessment Fusion): 0-100, perceptual quality
- **VMAF Aligned**: Aspect-ratio-aware VMAF (more accurate for crops/resizes)
- **Alignment method**: How aspect ratios were matched (same/crop/scale)

See `data/results/csv/quality_metrics.csv` for full results.

## C2PA Verification

All transformed assets are verified for C2PA manifest retention:

**Expected Result**: **100% manifest loss**

Current findings show all transformation tools (PIL, OpenCV, ffmpeg) **strip C2PA metadata** during processing. This is the primary research outcome.

See `data/results/csv/c2pa_validation.csv` for verification results.

## Important Notes

### PNG Lossless Compression
Both PNG c0 and c9 produce **pixel-identical output** to the original:
- PSNR = "inf" (infinite)
- SSIM = 1.0 (perfect)
- lossless_match = 1 (exact)

This is mathematically correct - PNG DEFLATE compression is lossless regardless of level.

### VMAF for Videos
Two VMAF metrics are calculated:
- **vmaf_stretched**: Traditional method, scales distorted to reference (may show aspect ratio artifacts)
- **vmaf_aligned**: Crops/scales reference to match distorted aspect ratio (more accurate for platform transforms)

For transforms that preserve aspect ratio (compression only), both metrics are identical.

### Transform Testing Strategy
Transforms are designed to cover:
1. **Common workflows**: JPEG export, video encoding, resizing
2. **Platform behaviors**: Simulating Instagram/Twitter processing
3. **Edge cases**: Extreme compression, aggressive cropping
4. **Lossless operations**: Testing if C2PA survives bit-exact transforms

## Related Scripts

- **Compression**: `scripts/processing/transformations/compress_images.py`, `compress_videos.py`
- **Editing**: `scripts/processing/transformations/edit_assets.py`
- **Quality metrics**: `scripts/processing/metrics/calculate_quality_metrics.py`
- **C2PA verification**: `scripts/c2pa/verification/verify_c2pa.py`

## Disk Space Requirements

- **Transformed images**: ~15 GB (compressed formats reduce size)
- **Transformed videos**: ~40 GB (varies by bitrate)
- **Total**: ~55 GB

Ensure adequate disk space before running transformation pipeline.

## Troubleshooting

**Q: Transformation script failing?**
- Check signed assets exist in `data/prepared_assets/signed_assets/`
- Verify ffmpeg is installed for video transforms
- Review logs in `data/results/logs/`

**Q: Quality metrics showing "NA"?**
- Ensure original reference file exists
- Check file formats are compatible
- For videos, verify ffmpeg can read both files

**Q: Why do lossless transforms still lose C2PA?**
- C2PA metadata is stored outside the pixel data
- Even lossless operations often re-encode the container
- This is a key research finding: tools don't preserve C2PA
