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
- diffusers
- transformers
- accelerate
- c2pa-python
- ffmpeg-python
- opencv-python
- Pillow
- numpy
- pandas
- matplotlib
- seaborn
- scikit-image
- pyvmaf (or Python wrapper that calls the VMAF CLI)
- typer or argparse (for CLI)

System binaries (Dockerfile must install)
----------------------------------------
- ffmpeg
- libvmaf (VMAF binary or ffmpeg + libvmaf)

Folder structure (canonical)
----------------------------
Follow the `project_root/` layout exactly. Scripts must create folders if they do not exist.

Metric definitions (canonical strings to be used in CSV)
--------------------------------------------------------
- filename
- asset_type (image/video)
- transform
- level
- manifest_present (0/1)
- verified (0/1)
- signature_valid (0/1)
- hash_match (0/1)
- psnr (float or NA)
- ssim (float or NA)
- vmaf (float or NA)
- seed
- model_version
- timestamp

Ethics & safety constraints
---------------------------
- Do not generate or encourage generation of synthetic media depicting real, private persons without signed consent.
- Avoid enabling any approaches that are designed to stealthily bypass security measures.

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

