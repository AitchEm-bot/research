# Phase 2.5 — Social Media Round-Trip Testing

## ✅ Objective

Test C2PA manifest persistence after uploading to social media platforms and re-downloading the returned files.

**Research Question**: Do social media platforms preserve, strip, or modify C2PA manifests during their upload/processing pipelines?

---

## 📋 Overview

This phase tests real-world manifest robustness by:
1. **Uploading** C2PA-signed images/videos to social platforms
2. **Downloading** the processed/returned versions
3. **Verifying** if C2PA manifests survived
4. **Measuring** quality degradation (PSNR/SSIM/VMAF)

---

## 🏗️ Folder Structure

```
platform_tests/
├── README_PHASE_2.5.md              (this file)
├── platform_manifest.csv            (manual tracking log - you fill this)
├── platform_manifest_TEMPLATE.csv   (copy this to start)
│
├── instagram/
│   ├── uploads/          # Place files here before uploading
│   └── returned/         # Save downloaded files here
│
├── twitter/
│   ├── uploads/
│   └── returned/
│
├── facebook/
│   ├── uploads/
│   └── returned/
│
├── youtube_shorts/
│   ├── uploads/
│   └── returned/
│
├── tiktok/
│   ├── uploads/
│   └── returned/
│
└── whatsapp/
    ├── compressed/
    │   ├── uploads/
    │   └── returned/
    └── file_mode/
        ├── uploads/
        └── returned/
```

---

## 🚀 Workflow

### Step 1: Pre-Upload Preparation

Run the preparation script to verify C2PA signatures exist:

```bash
python scripts/platform/prepare_platform_uploads.py
```

**The script will**:
- Let you select assets (images/videos) to test
- Let you select platforms
- Copy files to `platform_tests/<platform>/uploads/`
- **Verify C2PA signatures are present**
- Warn if any signature is missing/invalid
- Generate upload instructions

**IMPORTANT**: Only upload assets that show "✓ C2PA manifest verified"

---

### Step 2: Manual Upload (YOU DO THIS)

For each platform, upload your test asset:

#### Instagram
- **Post**: Upload to feed (no filters, no edits)
- **Story**: Upload as story (24h expiry - download quickly!)
- **Reel**: Upload as reel (vertical video)

#### Twitter (X)
- Upload as tweet attachment (no cropping)

#### Facebook
- Upload to your timeline as post

#### YouTube Shorts
- Upload as Short (vertical video < 60s)

#### TikTok
- Upload as video post (no effects, no filters)

#### WhatsApp
- **Compressed mode**: Send to a contact (default compression)
- **File mode**: Send as document/file (no compression)

**⚠️ CRITICAL RULES**:
- ❌ NO filters
- ❌ NO edits
- ❌ NO auto-enhance
- ❌ NO cropping (use original dimensions)
- ✅ Upload in highest quality available

---

### Step 3: Download Returned Files

After upload completes (wait a few seconds for processing):

1. **Download the file** back from the platform
2. **Rename using the naming convention** (see below)
3. **Place in the correct `/returned/` folder**

---

## 📝 File Naming Convention (MANDATORY)

```
{original_name}__{platform}__{mode}__{timestamp}.{ext}
```

### Examples:

```
img_004_seed21__instagram__post__20250211.jpg
img_005_seed22__instagram__story__20250211.jpg
img_006_seed23__instagram__reel__20250211.mp4

video_002_seed100__twitter__upload__20250211.mp4
video_003_seed101__facebook__post__20250211.mp4
video_004_seed102__youtube_shorts__upload__20250211.mp4
video_005_seed103__tiktok__upload__20250211.mp4

img_007_seed24__whatsapp__compressed__20250211.jpg
img_008_seed25__whatsapp__file__20250211.jpg
```

### Valid Platform Modes:

| Platform | Valid Modes |
|----------|-------------|
| instagram | `post`, `story`, `reel` |
| twitter | `upload` |
| facebook | `post` |
| youtube_shorts | `upload` |
| tiktok | `upload` |
| whatsapp | `compressed`, `file` |

**Timestamp format**: `YYYYMMDD` (e.g., `20250211` for Feb 11, 2025)

---

### Step 4: Log Metadata

Open `platform_manifest.csv` and add a row:

| Column | Description | Example |
|--------|-------------|---------|
| `original_filename` | Name before upload | `img_004_seed21_signed.png` |
| `returned_filename` | Name after download | `img_004_seed21__instagram__post__20250211.jpg` |
| `platform` | Platform name | `instagram` |
| `mode` | Upload mode | `post` |
| `upload_method` | web or mobile | `mobile` |
| `user_account` | (optional) Your account | `@testuser123` |
| `timestamp_upload` | Upload time | `2025-02-11 14:30:00` |
| `timestamp_download` | Download time | `2025-02-11 14:35:00` |
| `returned_resolution` | New resolution | `1080x1080` |
| `returned_filesize_bytes` | File size | `245678` |
| `notes` | Observations | `Instagram removed audio track` |

---

### Step 5: Process Returned Files

After downloading and logging all returned files:

```bash
python scripts/platform/process_platform_returns.py
```

**The script will**:
- Scan all `/returned/` folders
- Parse filenames to extract metadata
- Match with originals in `data/manifests/`
- Run C2PA verification
- Calculate quality metrics (PSNR/SSIM/VMAF)
- Join with your `platform_manifest.csv` data
- Produce `data/metrics/platform_results.csv`

---

### Step 6: Merge with Main Dataset

```bash
python scripts/metrics/merge_results.py
```

This updates `final_metrics.csv` to include platform round-trip results with:
- `transform_type = "platform_roundtrip"`
- New columns: `platform`, `platform_mode`

---

## 📊 Expected Outcomes

Based on preliminary research, we expect:

| Platform | Manifest Survival | Quality Degradation |
|----------|-------------------|---------------------|
| Instagram | ❓ Unknown | High (aggressive compression) |
| Twitter | ❓ Unknown | Medium |
| Facebook | ❓ Unknown | High |
| YouTube Shorts | ❓ Unknown | Medium (H.264 re-encode) |
| TikTok | ❓ Unknown | High |
| WhatsApp (compressed) | ❓ Unknown | Very High |
| WhatsApp (file) | ❓ Unknown | Low/None |

**Your testing will provide definitive answers!**

---

## 🔍 What We're Measuring

### C2PA Manifest Persistence
- **manifest_present**: Is C2PA metadata still there?
- **verified**: Does cryptographic validation pass?
- **signature_valid**: Is the signature intact?
- **hash_match**: Does content hash still match?

### Quality Degradation
- **PSNR**: Peak Signal-to-Noise Ratio (higher = better)
- **SSIM**: Structural Similarity (0-1, higher = better)
- **VMAF**: Video Multimethod Assessment Fusion (0-100, higher = better)

---

## ⚠️ Important Notes

### Account Privacy
- Use a **test account**, not your personal account
- Downloaded files contain your account data - they're gitignored for privacy

### Platform Terms of Service
- Respect each platform's ToS
- Don't mass-upload automated content
- This is research, not automation

### Time Sensitivity
- Instagram Stories expire in 24 hours
- Download immediately after upload
- Some platforms may process videos asynchronously (wait a few minutes)

### File Formats
- Platforms may convert formats (PNG → JPG, MP4 → MOV)
- Extension in filename should match the downloaded file's actual format

---

## 🐛 Troubleshooting

**"Original file not found"**
- Check that the filename prefix matches an asset in `data/manifests/images/` or `data/manifests/videos/`

**"Invalid filename format"**
- Must follow: `{original}__{platform}__{mode}__{timestamp}.{ext}`
- Use double underscores (`__`) as separators

**"Platform mode not recognized"**
- Check the Valid Platform Modes table above
- Mode must be lowercase

**"VMAF calculation failed"**
- May occur if platform changed video dimensions
- Script should auto-handle this with scaling

---

## 📈 Next Steps

After Phase 2.5 completes:
- **Phase 4**: Statistical analysis and visualization
- Generate VSR (Verification Success Rate) by platform
- Create comparison heatmaps
- Document findings in thesis

---

## 📚 Additional Resources

- C2PA Specification: https://c2pa.org/specifications/specifications/2.1/index.html
- c2patool Documentation: https://github.com/contentauth/c2pa-rs
- Project README: `../README.md`
- Phase Overview: `../../PHASES.md`

---

**Questions? Check `PHASES.md` or contact the project maintainer.**
