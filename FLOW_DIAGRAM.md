## EXTERNAL VIDEO FLOW
  ---
  Visual Flow Diagram

  📁 data/raw_out_videos/ (60 Veo3.1 videos)
           ↓
      [prepare_external_videos.py]
      ✓ Check C2PA signature (native Google manifests)
      ✓ Preserve existing manifests
           ↓
  📁 data/manifests/videos/external/ (60 signed videos)
           ↓
      [compress_videos.py]
      ✓ H.264: 3 variants (5000k, 2000k, 500k)
      ✓ H.265: 2 variants (2000k, 500k)
      ✓ FPS: 2 variants (5fps, 3fps)
           ↓
  📁 data/transformed/compression/videos/ (420 files: 60 × 7)
           ↓
      [edit_assets.py]
      ✓ Crop: 3 variants (90%, 75%, 50%)
      ✓ Resize: 2 variants (256×256, 768×768)
      ✓ Rotate: 3 variants (90°, 180°, 270°)
      ✓ Color: 6 variants (brightness/contrast/saturation ±20%)
           ↓
  📁 data/transformed/editing/videos/ (840 files: 60 × 14)
           ↓
      [verify_c2pa.py]
      ✓ Check C2PA survival
      ✓ Extract BMFF hash validation
      ✓ Identify failure reasons
           ↓
  📄 data/metrics/c2pa_validation.csv (1260 rows: 60 × 21)
           ↓
      [calculate_quality_metrics.py]
      ✓ Calculate VMAF scores
      ✓ Compare to original
           ↓
  📄 data/metrics/quality_metrics.csv (1260 rows)
           ↓
      [merge_results.py]
      ✓ Join C2PA + Quality data
      ✓ Add metadata columns
           ↓
  📄 data/metrics/final_metrics.csv
     (1260 rows with complete analysis)

  ---


## IMAGE GENERATION FLOW
 ---
  Visual Flow Diagram

  🎨 START: AI Generation
           ↓
      [generate_images.py]
      ✓ Stable Diffusion v1.4
      ✓ 100 unique prompts (from prompts.txt)
      ✓ 1024×1024 PNG (native max resolution)
      ✓ Seeds: 42-141 (100 images)
           ↓
  📁 data/raw_images/ (100 images)
           ↓
      [embed_c2pa_v2.py]
      ✓ Create C2PA manifests
      ✓ Sign with ES256 test cert
      ✓ Verify integrity
           ↓
  📁 data/manifests/images/ (100 signed images)
           ↓
      ┌─────────────┴─────────────┐
      ↓                           ↓
  [compress_images.py]    [edit_assets.py]
  ✓ JPEG: q95,75,50,25    ✓ Crop: 90%,75%,50%
  ✓ PNG: c9,c0 (lossless) ✓ Resize: 256×256, 1024×1024
      ↓                   ✓ Rotate: 90°,180°,270°
     (600 images)         ✓ Color: brightness/contrast/sat
                               ↓
                          (1800 images)
      └─────────────┬─────────────┘
                    ↓
  📁 data/transformed/ (2400 total: 100 × 24)
           ↓
      [verify_c2pa.py]
      ✓ Check manifest survival
      ✓ Extract validation flags
      ✓ Classify failure reasons
           ↓
  📄 c2pa_validation.csv (2400 rows)
           ↓
      [calculate_quality_metrics.py]
      ✓ Calculate PSNR/SSIM
      ✓ Detect lossless matches
      ✓ Compare to original
           ↓
  📄 quality_metrics.csv (2400 rows)
           ↓
      [merge_results.py]
      ✓ Join C2PA + Quality
      ✓ Validate dataset
      ✓ Report statistics
           ↓
  📄 final_metrics.csv
     (2400 rows - complete analysis)

  ---

  

## VIDEO GENERATION FLOW (REMOVED - Only External Videos Now)
  ---
  **NOTE**: Internal SVD video generation has been deprecated.

  The pipeline now focuses exclusively on:
  - **100 internal images** (1024×1024, Stable Diffusion v1.4)
  - **60 external videos** (Google Veo3.1 from data/raw_out_videos/)

  Legacy video support (seed 4/42/43) has been removed from verification scripts.

  For video testing, use EXTERNAL VIDEO FLOW above.

  ---

## PLATFORM ROUND-TRIP FLOW (Phase 2.5)
---
  Visual Flow Diagram

  🎯 STEP 1: PREPARATION (Automated)
           ↓
      [prepare_platform_uploads.py --auto-sample]
      ✓ Randomly sample 100 images + 60 videos
      ✓ Sources: data/manifests/ (original signed only)
      ✓ Distribution per platform:
        - Instagram: 25 images + 10 videos
        - Twitter: 25 images + 10 videos
        - Facebook: 25 images + 10 videos
        - WhatsApp: 25 images + 10 videos
        - YouTube Shorts: 0 images + 10 videos
        - TikTok: 0 images + 10 videos
      ✓ Verify C2PA pre-upload
      ✓ Copy to platform-specific uploads/ folders
      ✓ Generate tracking CSV
           ↓
  📁 platform_tests/{platform}/uploads/ (160 total assets)
           ↓
  📱 STEP 2: MANUAL UPLOAD (You do this)
      ✓ Transfer to mobile (if needed)
      ✓ Upload to platform app
      ✓ NO filters, NO edits
      ✓ Highest quality only
           ↓
  ☁️ Platform Processing
      (Transcoding, compression, metadata handling)
           ↓
  💾 STEP 3: MANUAL DOWNLOAD (You do this)
      ✓ Download returned file
      ✓ Rename: {original}__{platform}__{mode}__{timestamp}
      ✓ Save to returned/ folder
      ✓ Log in platform_manifest.csv
           ↓
  📁 platform_tests/{platform}/returned/
           ↓
  🔍 STEP 4: AUTOMATED PROCESSING
           ↓
      [process_platform_returns.py]
      ✓ Scan returned files
      ✓ Parse filenames
      ✓ Find original assets
      ✓ Run C2PA verification
      ✓ Calculate quality metrics (PSNR/SSIM/VMAF)
      ✓ Merge with manual log
           ↓
  📄 platform_results.csv (160 rows)
           ↓
      [merge_results.py]
      ✓ Append to final_metrics.csv
      ✓ Set transform_type="platform_roundtrip"
           ↓
  📄 final_metrics.csv
     (complete dataset: 2400 images + 1260 videos + 160 platform = 3820 rows)

  ---