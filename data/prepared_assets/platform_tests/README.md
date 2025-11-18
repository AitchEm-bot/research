# Phase 2.5: Social Media Platform Round-Trip Testing

This directory contains infrastructure for testing C2PA manifest persistence through social media platform upload/download cycles.

## Purpose

Measure C2PA robustness in **real-world social media workflows**:

1. Upload C2PA-signed media to social platforms
2. Download the returned/processed media
3. Verify C2PA manifest retention
4. Measure quality degradation (PSNR/SSIM/VMAF)

## Directory Structure

```
platform_tests/
├── instagram/
│   ├── uploads/          # Assets to upload
│   └── returned/         # Downloaded from Instagram
├── twitter/
│   ├── uploads/
│   └── returned/
├── facebook/
│   ├── uploads/
│   └── returned/
├── youtube_shorts/
│   ├── uploads/
│   └── returned/
├── tiktok/
│   ├── uploads/
│   └── returned/
├── whatsapp/
│   ├── uploads/
│   └── returned/
└── platform_manifest.csv  # Manual upload/download tracking
```

## Tested Platforms

| Platform | Upload Modes | Asset Types |
|----------|-------------|-------------|
| Instagram | video, image, story, reel | Images, Videos |
| Twitter | video, image | Images, Videos |
| Facebook | video, image | Images, Videos |
| YouTube Shorts | short | Videos only |
| TikTok | video | Videos only |
| WhatsApp | video, image, status | Images, Videos |

## Testing Workflow

### 1. Prepare Assets for Upload

```bash
python scripts/processing/preprocessing/platform/prepare_platform_uploads.py
```

**Options**:
- **Interactive mode**: Manually select assets from transformed folder
- **Auto-sampling mode**: `--auto-sample` flag automatically samples:
  - **100 images** (25 per image-supporting platform)
  - **60 videos** (10 per platform)

**Output**: Assets copied to `{platform}/uploads/` with verification

### 2. Manual Upload

**IMPORTANT**: Uploads must be done manually via mobile/web apps (platforms don't provide upload APIs for this use case).

For each platform:
1. Navigate to `platform_tests/{platform}/uploads/`
2. Upload assets using the platform's app/website
3. Note upload timestamps

### 3. Manual Download

After platform processing:
1. Download assets from the platform
2. Save to `platform_tests/{platform}/returned/`
3. Use naming convention: `{original}__{platform}__{mode}__{timestamp}.{ext}`

**Example**: `seed42_h264_bitrate2000k__instagram__reel__20251112-143022.mp4`

### 4. Manual Logging

Record metadata in `platform_manifest.csv`:

| Column | Description |
|--------|-------------|
| `original_filename` | Source file from uploads/ |
| `platform` | Platform name (instagram, twitter, etc.) |
| `mode` | Upload mode (video, image, reel, etc.) |
| `upload_timestamp` | ISO 8601 upload time |
| `download_timestamp` | ISO 8601 download time |
| `notes` | Optional observations |

### 5. Automated Processing

```bash
python scripts/processing/preprocessing/platform/process_platform_returns.py
```

This script:
- Scans all `returned/` folders
- Parses filenames for metadata
- Runs C2PA verification
- Calculates quality metrics (PSNR, SSIM, VMAF)
- Joins with manual CSV log
- Generates `data/results/csv/platform_results.csv`

### 6. Merge into Final Dataset

```bash
python scripts/processing/metrics/merge_results.py
```

Appends platform results to `final_metrics.csv` with:
- `platform` column
- `platform_mode` column
- `media_source` column (internal/external/unknown)
- `transform_type` = "platform_roundtrip"

## Auto-Sampling Strategy

When using `--auto-sample` flag:

**Images** (100 total):
- 25 per platform (Instagram, Twitter, Facebook, WhatsApp)
- Diverse transform types (JPEG compression, PNG, editing)
- Original signed assets (not pre-transformed)

**Videos** (60 total):
- 10 per platform (all 6 platforms)
- Mix of internal (SVD) and external (Veo3.1) sources
- Diverse quality levels

## Expected Outcomes

Based on current analysis:

**Manifest Retention**: All platforms **strip C2PA manifests** (0% retention)

**Quality Degradation** (ranked best → worst):
1. **WhatsApp**: 99.9 VMAF (minimal compression)
2. **Facebook**: 94.9 VMAF
3. **TikTok**: 92.6 VMAF
4. **Instagram**: 89.5 VMAF
5. **YouTube**: 86.3 VMAF
6. **Twitter**: 84.9 VMAF (highest compression)

## File Naming Convention

### Upload Files
Original signed assets, preserving source naming:
```
img_042_seed42_20251113_233209_signed.png
video_15_signed.mp4
```

### Returned Files
Must follow this pattern for automated parsing:
```
{original_name}__{platform}__{mode}__{timestamp}.{ext}
```

**Components**:
- `{original_name}`: Original filename (without _signed suffix)
- `{platform}`: Platform name (lowercase)
- `{mode}`: Upload mode (video/image/reel/story/short/status)
- `{timestamp}`: YYYYmmdd-HHMMSS format
- `{ext}`: File extension

## VMAF Alignment Note

Platform testing uses **vmaf_aligned** metric to account for aspect ratio changes:

- **Stretched VMAF**: Scales distorted video to reference (may show artifacts from aspect ratio mismatch)
- **Aligned VMAF**: Crops reference to match distorted aspect ratio (more accurate for platform transforms)

Platforms often crop videos (e.g., Instagram 16:9 → 1:1), making aligned VMAF more representative of actual visual quality.

## Manual Tracking CSV Format

```csv
original_filename,platform,mode,upload_timestamp,download_timestamp,notes
img_042_signed.png,instagram,image,2025-01-12T14:30:22Z,2025-01-12T14:35:10Z,Minor color shift observed
video_15_signed.mp4,twitter,video,2025-01-12T15:00:00Z,2025-01-12T15:05:30Z,Significant compression
```

## Troubleshooting

**Q: Files not being detected by process_platform_returns.py?**
- Verify filename follows naming convention exactly
- Check file is in correct `returned/` subfolder
- Ensure CSV has matching entry

**Q: Quality metrics showing "NA"?**
- For videos: Ensure original reference exists in `signed_assets/`
- For images: Check PSNR/SSIM calculation isn't failing
- Review `data/results/logs/process_platform_returns.log`

**Q: Platform won't accept upload?**
- Check file size limits (varies by platform)
- Verify codec compatibility (H.264 most widely supported)
- Try different upload mode (e.g., story vs. post)

## Related Scripts

- **Preparation**: `scripts/processing/preprocessing/platform/prepare_platform_uploads.py`
- **Processing**: `scripts/processing/preprocessing/platform/process_platform_returns.py`
- **Merging**: `scripts/processing/metrics/merge_results.py`
- **Analysis**: `scripts/analysis/data_analysis/platform_analysis.py`

## Important Notes

1. **Manual intervention required**: No automated API upload/download
2. **Time-consuming**: Plan for several hours per platform
3. **Account requirements**: May need active accounts on all platforms
4. **Rate limits**: Space out uploads to avoid platform throttling
5. **Terms of Service**: Ensure compliance with platform policies

## Citation

If using platform testing results in research, acknowledge the manual testing methodology and cite platform specifications where available.
