# Raw Generated Videos (Stable Video Diffusion)

This directory contains raw videos generated using Stable Video Diffusion (SVD) for C2PA robustness testing.

## Contents

- **Internal videos** generated via SVD img2vid pipeline
- **Conditioning**: Images from `data/assets/raw_images_for_videos/`
- **Format**: MP4 (H.264)
- **Resolution**: 512×512 pixels
- **Duration**: 25 frames (~1 second at 25 fps)

## Generation Pipeline

```
Conditioning Image  →  SVD img2vid-xt  →  25-frame Video
   (1024×1024)         (diffusion model)     (512×512)
```

## Video Specifications

- **Model**: stabilityai/stable-video-diffusion-img2vid-xt
- **Frames**: 25 per video
- **Resolution**: 512×512 (downscaled from 1024×576 native)
- **Frame rate**: 25 fps
- **Codec**: H.264
- **Format**: MP4

## Purpose

These videos serve as **internal test media** for C2PA robustness testing:

1. **C2PA Signing**: Videos signed with test certificates → `data/prepared_assets/signed_assets/videos/internal/`
2. **Transformation Testing**: Signed videos undergo compression and editing
3. **Verification**: Test C2PA manifest persistence on video transformations

## Naming Convention

```
video_{index}_seed{seed}_{timestamp}.mp4
```

Example: `video_001_seed100_20251113_235959.mp4`

## Generation Command

```bash
python scripts/processing/generation/generate_videos.py
```

**Requirements**:
- CUDA-compatible GPU (8GB+ VRAM recommended)
- ~30 minutes for typical batch on RTX 3080

## Quality Considerations

**VMAF Limitations**: The SVD paper does not use VMAF for evaluation. VMAF is designed for compression artifacts, not generative synthesis quality. Results should be interpreted accordingly.

## Related Directories

- **Conditioning source**: `data/assets/raw_images_for_videos/`
- **Signed versions**: `data/prepared_assets/signed_assets/videos/internal/`
- **External videos**: `data/assets/raw_out_videos/` (Veo3.1, Sora, etc.)

## Legacy Support

**REMOVED**: Old SD1.4-based video generation (seeds 4/42/43) has been deprecated. The pipeline now exclusively recognizes SVD videos (seed 100+).

## Important Notes

1. **Model Status**: SVD is a PREPRINT (arXiv 2311.15127, November 2023) - not yet peer-reviewed
2. **Annotation Required**: Mark as preprint in thesis acknowledgments
3. **Resolution Tradeoff**: 512×512 used instead of 1024×576 for computational efficiency

## Citation

```
Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M.,
Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., Jampani, V.,
& Rombach, R. (2023). Stable Video Diffusion: Scaling Latent Video
Diffusion Models to Large Datasets. arXiv preprint arXiv:2311.15127.
```

**Model Card**: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
**Paper**: https://arxiv.org/abs/2311.15127

## Troubleshooting

**Out of Memory (OOM)?**
- Reduce batch size in generation script
- Use mixed precision (FP16)
- Close other GPU applications

**Generation too slow?**
- Ensure CUDA is properly installed
- Check GPU utilization: `nvidia-smi`
- Consider reducing video count
