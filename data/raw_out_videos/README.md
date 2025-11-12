# External Video Sources

This folder is for externally generated videos from sources like:
- Sora 2 (OpenAI)
- Runway Gen-3
- Pika Labs
- Other text-to-video or image-to-video generation services

## Workflow

### 1. Place Videos Here
Drop your externally generated video files in this directory:
```
data/raw_out_videos/
├── my_sora_video_001.mp4
├── runway_output_002.mp4
└── ...
```

Supported formats: `.mp4`, `.mov`, `.avi`

### 2. Run Preparation Script
```bash
python scripts/external/prepare_external_videos.py
```

### 3. What the Script Does
The script will automatically:
1. **Scan** this folder for video files
2. **Check** C2PA signature status using `c2patool`
3. **Sign** any unsigned videos with built-in test certificates
4. **Move** signed videos to `data/manifests/videos/external/`
5. **Generate** metadata JSON sidecar files
6. **Log** all operations to `data/external_videos.log`

### 4. After Processing
Your videos will be moved to:
```
data/manifests/videos/external/
├── video_ext_001_signed.mp4
├── video_ext_002_signed.mp4
└── ...
```

They will then automatically enter the Phase 2 transformation pipeline along with internally generated videos.

---

## Important Notes

- **No manual signing needed** - the script handles C2PA signing automatically
- **Original filenames preserved** - videos are renamed with `_signed` suffix
- **Already-signed videos** - if a video already has a C2PA manifest, it will be moved without re-signing
- **Metadata tracking** - generation parameters and source info stored in JSON sidecar files

---

## File Naming Convention

After processing, files will be named:
```
video_ext_{number}_{source}_{timestamp}_signed.mp4
```

Example: `video_ext_001_sora_20250211_signed.mp4`

---

## Quality Requirements

For best results with the testing pipeline:
- **Resolution**: 512×512 or higher (will be resized if needed)
- **Frame rate**: Any (will be normalized to 7fps for comparison)
- **Duration**: 25 frames minimum (about 3.5 seconds at 7fps)
- **Codec**: H.264 or H.265 preferred

---

## Troubleshooting

**"Video not found"**
- Check that files are directly in this folder (not in subfolders)

**"Signing failed"**
- Verify c2patool is installed: `c2patool --version`
- Check video file is not corrupted

**"Already signed"**
- This is informational - video will still be moved to manifests/

For more help, see `README.md` in the project root.
