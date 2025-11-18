# Raw Generated Images (Stable Diffusion 1.4)

This directory contains raw images generated using Stable Diffusion v1.4 for C2PA robustness testing.

## Contents

- **100 images** generated at native 1024×1024 resolution
- **Seeds**: 42-141 (sequential, reproducible)
- **Model**: CompVis/stable-diffusion-v1-4
- **Format**: PNG (lossless)
- **Naming**: `img_{index:03d}_seed{seed}_{timestamp}.png`

## Generation Details

Images were generated using diverse prompts designed to cover various visual categories:
- Natural scenes (landscapes, animals, plants)
- Urban environments (architecture, streets, interiors)
- Abstract concepts
- Objects and still life
- People and portraits

**Prompts**: See `prompts.txt` in this directory for the full list of generation prompts.

## Purpose

These images serve as the **source material** for the C2PA robustness pipeline:

1. **C2PA Signing**: Images are signed with test certificates → `data/prepared_assets/signed_assets/images/`
2. **Transformation Testing**: Signed images undergo compression and editing → `data/prepared_assets/transformed/`
3. **Verification**: Transformed images are verified to test C2PA manifest persistence

## Quality Specifications

- **Resolution**: 1024×1024 pixels (SD1.4 native maximum)
- **Bit depth**: 8-bit RGB
- **Color space**: sRGB
- **No post-processing**: Raw model outputs without upscaling or enhancement

## Reproducibility

To regenerate these images:

```bash
python scripts/processing/generation/generate_images.py --seed 42 --count 100 --output-dir data/assets/raw_images
```

**Note**: Requires approximately 6GB VRAM and ~10 minutes on RTX 3080.

## Related Directories

- **Input to**: `scripts/c2pa/embedding/embed_c2pa_v2.py` (C2PA signing)
- **Signed versions**: `data/prepared_assets/signed_assets/images/`
- **Video conditioning**: Some images used in `data/assets/raw_images_for_videos/`

## Citation

If using these images in research, cite the Stable Diffusion v1.4 model:

```
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
High-Resolution Image Synthesis with Latent Diffusion Models.
In CVPR 2022.
```

**Model Card**: https://huggingface.co/CompVis/stable-diffusion-v1-4
