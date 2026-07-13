# StableMind: Source-Free Cross-Subject fMRI Decoding with Regularized Adaptation

Curated StableMind training and evaluation code built on top of MindEyeV2.

## Overview

This repository contains the training, reconstruction, enhancement, and evaluation pipeline.

The main entry points are:

- `Train_Tuner_StableMind_sr_blur_combo.py`: main training and fine-tuning script
- `run_stablemind_sr_blur_combo_job.sh`: end-to-end training + evaluation launcher
- `evaluate_tuner.sh`: reconstruction, enhanced reconstruction, and final evaluation runner
- `recon_inference_tuner.py`: base reconstruction inference
- `enhanced_recon_inference_tuner.py`: enhanced reconstruction stage
- `final_evaluations_tuner.py`: final metric computation and result export

## Repository Structure

```text
MindEyeV2-main-stablemind-complete/
├── README.md
├── Train_Tuner_StableMind_sr_blur_combo.py
├── run_stablemind_sr_blur_combo_job.sh
├── evaluate_tuner.sh
├── recon_inference_tuner.py
├── enhanced_recon_inference_tuner.py
├── final_evaluations_tuner.py
├── setup.sh
└── ...
```

## Requirements

create/install dependencies:

```bash
bash setup.sh
```

Key dependencies used by the current code include:

- `torch==2.1.0`
- `torchvision==0.16.0`
- `accelerate==0.24.1`
- `diffusers==0.23.0`
- `transformers==4.37.2`
- `webdataset==0.2.73`
- `open_clip_torch==2.24.0`

## Quick Start

Inspect training arguments:

```bash
python Train_Tuner_StableMind_sr_blur_combo.py --help
```

Run one training job followed by evaluation:

```bash
bash run_stablemind_sr_blur_combo_job.sh 0 StableMind_subj01_1se_10bs_SrcBlurCombo 1
```

Evaluate an existing model only:

```bash
MODEL_NAME=<your_model_dir_name> GPU=0 bash evaluate_tuner.sh
```