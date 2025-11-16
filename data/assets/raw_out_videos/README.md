# External Video Sources

This directory contains externally generated videos from AI platforms:
- **Google Veo3.1** (60 videos currently included)
- Sora 2, Runway Gen-3, Pika Labs (future sources)

## Quick Start

### 1. Place Videos Here
Drop your externally generated video files in this directory (`data/assets/raw_out_videos/`).

**Supported formats:** `.mp4`, `.mov`, `.avi`

### 2. Run Preparation Script
```bash
python scripts/processing/preprocessing/external/prepare_external_videos.py
```

### 3. What Happens
The script automatically:
1. Scans for video files in this directory
2. Checks existing C2PA signature status
3. Signs unsigned videos with built-in test certificates
4. Moves signed videos to `data/prepared_assets/manifests/videos/external/`
5. Logs all operations to `data/results/logs/`

### 4. After Processing
Signed videos are moved to:
```
data/prepared_assets/manifests/videos/external/
├── video_1_signed.mp4
├── video_2_signed.mp4
└── ...
```

They then automatically enter the transformation pipeline (Phase 2) alongside internally generated videos.

---

## Important Notes

- **No manual signing needed** - script handles C2PA embedding automatically
- **Already-signed videos** - preserved without re-signing
- **Metadata tracking** - source info stored in JSON sidecar files
- **Integration** - external videos are marked with `media_source=external` in final_metrics.csv

---

## Quality Requirements

- **Minimum resolution:** 256×256 pixels
- **Minimum duration:** 1 second
- **Maximum file size:** 500 MB (for practical processing)
- **Preferred codecs:** H.264 or H.265

---

## Troubleshooting

**"Video not found"**
- Ensure files are directly in this folder (not subfolders)

**"Signing failed"**
- Verify c2patool is installed: Run `tools/c2patool/c2patool/c2patool.exe --version`
- Check video file integrity

**"Already signed"**
- Informational message - video will be moved to manifests/ without re-signing

For more details, see the main `README.md` in the project root.
