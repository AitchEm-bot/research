# Raw Images for Video Generation (SVD Conditioning)

This directory contains conditioning images used to generate videos via Stable Video Diffusion (SVD).

## Contents

- **Conditioning images** for SVD img2vid pipeline
- **Source**: Selected from `data/assets/raw_images/` (SD1.4 outputs)
- **Format**: PNG (1024×1024)
- **Purpose**: First frames for video generation

## Video Generation Workflow

```
raw_images_for_videos/  →  SVD img2vid  →  raw_videos/
     (conditioning)          (25 frames)     (output)
```

1. **Select images**: Choose diverse scenes from `raw_images/`
2. **SVD processing**: stabilityai/stable-video-diffusion-img2vid-xt
3. **Output**: 25-frame videos at 512×512 resolution

## Relationship to Other Directories

- **Source**: `data/assets/raw_images/` (SD1.4 generated images)
- **Output**: `data/assets/raw_videos/` (SVD generated videos)
- **Signed versions**: `data/prepared_assets/signed_assets/videos/internal/`

## SVD Model Details

- **Model**: stabilityai/stable-video-diffusion-img2vid-xt
- **Architecture**: Image-to-video latent diffusion
- **Output**: 25 frames per video
- **Native resolution**: 1024×576 (downscaled to 512×512 for this project)
- **Status**: PREPRINT (arXiv 2311.15127, not yet peer-reviewed)

## Generation Command

```bash
python scripts/processing/generation/generate_videos.py
```

The script will:
1. Read conditioning images from this directory
2. Generate 25-frame videos using SVD
3. Save to `data/assets/raw_videos/`

## Important Notes

- **Legacy support removed**: Old SD1.4 video generation (seeds 4/42/43) no longer supported
- **Current pipeline**: Only SVD-based video generation (seeds 100+)
- **Conditioning quality**: High-quality conditioning images produce better videos

## Citation

If using SVD-generated videos in research:

```
Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M.,
Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V.,
& Rombach, R. (2023). Stable Video Diffusion: Scaling Latent Video
Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127.
```

**Note**: Annotate as PREPRINT in thesis - not yet peer-reviewed.

**Model Card**: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
