CLAUDE.md — Project memory & agent constraints
==============================================

Purpose
-------
This file contains compact, authoritative project context the Claude agent must consult or obey whenever producing code or prose. It is not a conversational artifact — it is a set of immutable constraints and critical references.

Project title
-------------
Is C2PA's Metadata Robust in AI-Generated Content?

Primary objectives
------------------
1. Implement an end-to-end, reproducible pipeline to test the robustness of C2PA manifests on AI-generated images and videos under compression and editing transforms.
2. Compare C2PA-only approach with watermark/fingerprint baseline (optional later phase).
3. Produce reproducible plots, CSV metrics, and a short HTML report.

Required dependencies (install via pip in Docker)
------------------------------------------------
- Python >= 3.10
- torch (compatible CUDA build)
- diffusers >= 0.25.0
- transformers
- accelerate
- ffmpeg-python
- opencv-python
- Pillow
- numpy
- pandas
- matplotlib
- seaborn
- scikit-image
- imageio-ffmpeg
- pyvmaf (or Python wrapper that calls the VMAF CLI)
- typer or argparse (for CLI)

System binaries (Dockerfile must install)
----------------------------------------
- c2patool (official C2PA command-line tool from contentauth/c2pa-rs)
- ffmpeg
- libvmaf (VMAF binary or ffmpeg + libvmaf)

Folder structure (canonical)
----------------------------
Follow the `project_root/` layout exactly. Scripts must create folders if they do not exist.

```
project_root/
├── data/
│   ├── assets/                      # Raw generated/external assets
│   │   ├── raw_images/              # Generated images (internal pipeline, SD1.4)
│   │   ├── raw_videos/              # Generated videos (internal pipeline, SVD)
│   │   ├── raw_images_for_videos/   # Conditioning images for video generation
│   │   └── raw_out_videos/          # External videos (Veo3.1, Sora 2, Runway, etc.)
│   ├── prepared_assets/             # Processed assets ready for testing
│   │   ├── manifests/               # C2PA signed assets
│   │   │   ├── images/              # Signed images (all sources)
│   │   │   └── videos/
│   │   │       ├── internal/        # Signed internal videos (SVD)
│   │   │       └── external/        # Signed external videos (Veo3.1, etc.)
│   │   ├── c2pa_manifests/          # Extracted C2PA manifest JSONs
│   │   │   ├── images/
│   │   │   └── videos/
│   │   │       ├── internal/
│   │   │       └── external/
│   │   ├── transformed/             # Assets after transformations
│   │   │   ├── compression/
│   │   │   │   ├── images/
│   │   │   │   │   ├── jpeg/        # q95/, q75/, q50/, q25/
│   │   │   │   │   └── png/         # c9/, c0/
│   │   │   │   └── videos/
│   │   │   │       ├── h264/        # bitrate5000k/, bitrate2000k/, bitrate500k/
│   │   │   │       ├── h265/        # bitrate2000k/, bitrate500k/
│   │   │   │       └── fps/         # fps5/, fps3/
│   │   │   └── editing/
│   │   │       ├── images/          # resize/, crop/, rotate/
│   │   │       └── videos/          # resize/, crop/, trim/
│   │   └── platform_tests/          # Phase 2.5 social media testing
│   │       ├── instagram/
│   │       │   ├── uploads/         # Assets to upload
│   │       │   └── returned/        # Downloaded assets
│   │       ├── twitter/
│   │       ├── facebook/
│   │       ├── youtube_shorts/
│   │       ├── tiktok/
│   │       ├── whatsapp/
│   │       └── platform_manifest.csv # Manual upload/download tracking
│   └── results/                     # All outputs (CSV files and logs)
│       ├── c2pa_validation.csv      # C2PA verification results
│       ├── quality_metrics.csv      # Quality metrics (PSNR/SSIM/VMAF)
│       ├── platform_results.csv     # Phase 2.5 platform testing results
│       ├── final_metrics.csv        # Merged comprehensive results
│       └── logs/                    # All execution logs
├── scripts/
│   ├── common/                      # Shared utilities
│   │   └── utils.py                 # Centralized functions (logging, CSV, paths, etc.)
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
│               └── process_platform_returns.py
```

Metric definitions (canonical strings to be used in CSV)
--------------------------------------------------------
**Core Metrics:**
- filename
- asset_type (image/video)
- transform_type (jpeg_compression, png_compression, h264_compression, h265_compression, fps_adjustment, resize, crop, rotate, trim, platform_roundtrip)
- transform_level (quality value, bitrate, fps, percentage, platform name)
- seed
- model_version
- timestamp

**C2PA Verification Metrics:**
- manifest_present (0/1)
- verified (0/1) - Based on INTEGRITY validation only: claimSignature.validated = 1, otherwise 0
- signature_valid (0/1) - Cryptographic validity: claimSignature.validated status
- hash_match (0/1) - Hash consistency: assertion.dataHash.match or assertion.bmffHash.match status
- assertion_uris_match (0/1) - All assertion.hashedURI.match checks passed
- trust_verified (0/1) - Certificate trust chain (informational only, NOT a failure metric)
- validation_state (string) - c2patool validation_state field for reference
- failure_reason (string) - Human-readable failure description
- c2pa_processing_time_ms (float)

**Quality Metrics:**
- psnr (float, "inf" for lossless, or "NA")
- ssim (float or "NA")
- vmaf (float or "NA") - Backward compatibility (same as vmaf_stretched)
- vmaf_stretched (float or "NA") - VMAF with distorted video scaled to reference dimensions (may include aspect ratio distortion)
- vmaf_aligned (float or "NA") - VMAF with intelligent aspect ratio alignment (crops reference if aspect changed)
- vmaf_method (string) - Alignment method used: "same_aspect_ratio", "crop_reference_center_square", "scale_both_to_minimum", or "NA"
- lossless_match (0/1) - Whether pixels are identical to original
- lossless_transform (0/1) - Whether transform is known to be lossless (png_c0, png_c9)
- quality_processing_time_ms (float)

**Note on VMAF Metrics:**
- vmaf_stretched: Traditional method that scales distorted video to match reference dimensions. May show artificially low scores when aspect ratios differ (e.g., Instagram crops 16:9 to 1:1)
- vmaf_aligned: Crops or scales reference to match distorted aspect ratio before comparison. More accurately reflects perceptual quality when platforms apply cropping
- For same aspect ratio transforms (compression only), both metrics are identical
- For aspect ratio changes (editing/platform cropping), vmaf_aligned typically shows higher scores reflecting actual visual similarity

**Phase 2.5 Platform Testing (optional columns):**
- platform (string) - Platform name (instagram, twitter, facebook, youtube, tiktok, whatsapp)
- platform_mode (string) - Upload mode (post, upload, etc.)
- video_source (string) - Video origin: "internal" (generated by SVD), "external" (Veo3.1 or other platforms), "N/A" (for images), or "unknown"
- upload_timestamp (ISO 8601)
- download_timestamp (ISO 8601)

C2PA Implementation Method
-------------------------
- Uses c2patool (v0.24.0+) from contentauth/c2pa-rs for C2PA manifest embedding and verification
- c2patool includes built-in ES256 test certificates that produce spec-compliant C2PA manifests
- Built-in test certificates provide authentic cryptographic signatures without requiring CA infrastructure
- Python scripts invoke c2patool via subprocess for signing and verification operations
- This approach ensures full cryptographic validity while maintaining reproducibility

C2PA Hash Verification for Videos (BMFF Format-Specific Binding)
----------------------------------------------------------------
- For video files (MP4/BMFF format), c2patool uses content-aware hashing via "c2pa.hash.bmff.v3"
- This hashes only the media data streams (video/audio tracks), excluding the C2PA manifest box itself
- The validation code reports "assertion.bmffHash.match" (not "assertion.dataHash.match")
- This ensures that adding C2PA metadata does not break hash validation
- Any modification to actual visual/auditory content will break the hash
- Images use standard "assertion.dataHash.match" for the entire file
- Verification scripts must check for BOTH hash types to support images and videos

Research Scope Clarification
----------------------------
- This project tests C2PA manifest ROBUSTNESS under transformations, NOT PKI trust validation
- Verification metrics (VSR, SVR) measure cryptographic INTEGRITY, not CA trust chains
- Built-in test certificates produce genuine C2PA signatures with valid cryptographic properties
- Trust-related validation status (signingCredential.untrusted) is EXPECTED and RECORDED but does NOT indicate failure
- Success criteria: claimSignature.validated, assertion.hashedURI.match, assertion.dataHash.match
- The study measures manifest structural integrity and signature mathematics, not global PKI trust
- Full thesis scope restored: "Assessing the Cryptographic and Structural Integrity of C2PA Metadata in AI-Generated Media"

External Video Workflow (Phase 2.5 Extension)
---------------------------------------------
**Purpose**: Test C2PA robustness on videos from external generative AI platforms (Sora 2, Runway, Pika, etc.)

**Workflow**:
1. Place external videos in `data/raw_out_videos/`
2. Run `scripts/external/prepare_external_videos.py` to:
   - Check if videos already have C2PA manifests
   - If signed: preserve original manifest and move to `manifests/videos/external/`
   - If unsigned: sign with test certificate and move to `manifests/videos/external/`
3. External videos are automatically included in transformation pipeline alongside internal videos
4. Results are merged into `final_metrics.csv` with video_source tracking

**Supported Formats**: .mp4, .mov, .avi

**Quality Requirements**:
- Minimum resolution: 256x256 pixels
- Minimum duration: 1 second
- Maximum file size: 500 MB (for practical processing)

**Note**: PNG lossless compression handling
- PNG uses DEFLATE compression algorithm (lossless)
- Both png_c0 and png_c9 produce pixel-identical output to signed original
- Quality metrics: PSNR = "inf", SSIM = 1.0, lossless_match = 1
- This is mathematically correct and expected behavior

Phase 2.5: Social Media Platform Round-Trip Testing
---------------------------------------------------
**Purpose**: Measure C2PA manifest persistence through social media upload/download cycles

**Test Platforms**:
- Instagram (video, image, story, reel)
- Twitter (video, image)
- Facebook (video, image)
- YouTube Shorts (video)
- TikTok (video)
- WhatsApp (video, image, status)

**Workflow** (Manual with script assistance):
1. **Preparation**: Run `scripts/platform/prepare_platform_uploads.py`
   - **Interactive mode**: Manually select individual assets from transformed folder
   - **Auto-sampling mode** (NEW): `--auto-sample` flag automatically samples:
     - 100 images (25 per image-supporting platform: Instagram, Twitter, Facebook, WhatsApp)
     - 60 videos (10 per platform: all 6 platforms)
   - Sources: Original signed assets from data/manifests/ (not transformed)
   - Verify C2PA signature before upload
   - Copy to `platform_tests/{platform}/uploads/`
   - Generate upload instructions and tracking CSV

2. **Manual Upload**: Upload assets to platform via mobile/web app

3. **Manual Download**: Download assets from platform, place in `platform_tests/{platform}/returned/`

4. **File Naming Convention**: `{original}__{platform}__{mode}__{timestamp}.{ext}`
   - Example: `seed42_h264_bitrate2000k__instagram__reel__20250112-143022.mp4`

5. **Manual Logging**: Record metadata in `platform_tests/platform_manifest.csv`:
   - original_filename, platform, mode, upload_timestamp, download_timestamp, notes

6. **Processing**: Run `scripts/platform/process_platform_returns.py`
   - Scan returned folders
   - Parse filenames
   - Run C2PA verification
   - Calculate quality metrics (PSNR, SSIM, VMAF)
   - Join with manual CSV log
   - Generate `platform_results.csv`

7. **Merging**: Run `scripts/metrics/merge_results.py`
   - Append platform_results.csv to final_metrics.csv
   - Add columns: platform, platform_mode, video_source
   - Set transform_type = "platform_roundtrip"

**Expected Outcomes**:
- Most platforms will STRIP C2PA manifests (manifest_present = 0)
- Quality degradation varies by platform (compression, transcoding)
- Results inform real-world C2PA persistence analysis

Ethics & safety constraints
---------------------------
- Do not generate or encourage generation of synthetic media depicting real, private persons without signed consent.
- Avoid enabling any approaches that are designed to stealthily bypass security measures.

AI Model Citations
------------------
**Image Generation**: Stable Diffusion v1.4 (CompVis/stable-diffusion-v1-4)
- Paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., CVPR 2022, peer-reviewed)
- HuggingFace: https://huggingface.co/CompVis/stable-diffusion-v1-4
- Resolution: 1024×1024 (native maximum for SD v1.4)
- Dataset: 100 images with diverse prompts (stored in data/raw_images/prompts.txt)
- Seeds: 42-141 (100 sequential seeds for reproducibility)

**Video Generation**: Stable Video Diffusion (stabilityai/stable-video-diffusion-img2vid-xt)
- Paper: "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (Blattmann et al., arXiv 2311.15127, November 2023)
- **Status**: PREPRINT (not yet peer-reviewed) - annotate as such in thesis and code comments
- HuggingFace: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- Architecture: Image-to-video latent diffusion model (25 frames, 1024x576 native resolution, downscaled to 512x512 for this project)
- **Note**: Legacy video support (seed 4/42/43 from sd-v1.4-legacy-video) has been REMOVED. Pipeline now only recognizes SVD videos (seed 100+)
- VMAF Note: SVD paper does not use VMAF for evaluation. VMAF may not accurately assess diffusion-generated videos (designed for compression artifacts, not generative synthesis)

**External Videos**: Google Veo3.1 (and other external AI platforms)
- Source: 60 videos in data/raw_out_videos/
- Processing: Automatic C2PA signing via prepare_external_videos.py
- Platforms tested: Veo3.1 (may include Sora 2, Runway, Pika, etc. in future)

Citation policy
---------------
- Use peer-reviewed citations when justifying model/tool choices.
- If a tool or paper is a preprint or non-peer-reviewed, annotate it as such in comments or README.

Behavior constraints for Claude
-------------------------------
- Provide runnable code only; avoid pseudo-only solutions unless explicitly requested.
- Ask only one clarifying question at a time when missing critical info.
- Produce well-commented code and a README snippet for every produced file.
- After each phase, print a short structured checkpoint summary (files generated, how to run smoke test, expected outputs).

Versioning & reproducibility
----------------------------
- Every script must print environment info (python version, torch version, CUDA driver) to logs when run.
- Save random seeds and model checkpoints in `results/logs/`.

User contact
------------
User: AitchEm (project lead). The user will confirm before moving to the next phase.

