#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU="${GPU:-3}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/tmp/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/huggingface/transformers}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
#MODEL_NAME="final_subj01_pretrained_1sess_24bs_AugBlur_P0.5_W0.6"
#/data/lsm/model_mindeyev2/train_logs/subj01_lora_skip_10/last.pth

DEFAULT_MODELS=(
# "StableMind_subj01_1se_10bs_AugM01_F23_Blur_center_only_lora+skip8_al8_sk1.5_A_F_OABlur0.7_W0.3_Ada0.0_center_only_LoDrop0.33_Plan_0mix_1mix_2fourier_3fourier_4none_P0.5"
# "StableMind_subj01_1se_10bs_AugM01_F23_Blur_radius_aware_lora+skip8_al8_sk1.5_A_F_OABlur0.7_W0.3_Ada0.2_radius_aware_LoDrop0.33_Plan_0mix_1mix_2fourier_3fourier_4none_P0.5"
# "StableMind_subj01_1se_10bs_AugM01_F23_Blur_fixed_semantic_lora+skip8_al8_sk1.5_A_F_OABlur0.7_W0.3_Ada0.2_fixed_semantic_LoDrop0.33_Plan_0mix_1mix_2fourier_3fourier_4none_P0.5"
# "StableMind_subj01_1se_10bs_AugM01_F23_Blur_radius_aware_lora+skip8_al8_sk1.5_A_F_OABlur0.7_W0.3_Ada0.25_radius_aware_LoDrop0.33_Plan_0mix_1mix_2fourier_3fourier_4none_P0.5"
# "StableMind_subj01_1se_10bs_AugM01_F23_Blur_fixed_semantic_lora+skip8_al8_sk1.5_A_F_OABlur0.7_W0.3_Ada0.3_fixed_semantic_LoDrop0.33_Plan_0mix_1mix_2fourier_3fourier_4none_P0.5"
)

if [ -n "${MODEL_NAME:-}" ]; then
  MODELS=("${MODEL_NAME}")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

ENABLE_SPECTRAL_GATE=0
SPEC_GATE_INIT_BIAS=3.0
ENABLE_LEARN_NORM=0

#SUBJ=1
#CKPT_ROOT="/data1/gjt/NSD_CAM/"
#CKPT_FILE_ROOT="/data1/gjt/NSD_CAM/"
#DATA_PATH="/data1/zyl/NSD/"
#CACHE_DIR="/data0/gjt/CLIP-ViT-bigG-14/"

# CKPT_ROOT="/data/data/gjt/MindEyeV2-results/train_logs/"
# CKPT_FILE_ROOT="/data/data/gjt/MindEyeV2-results/train_logs/"
# CKPT_ROOT="${CKPT_ROOT:-/data/data/gjt/MindEyeV2-main/src/train_logs/}"
CKPT_ROOT="${CKPT_ROOT:-/data/data/gjt/MindEyeV2-results/train_logs/}"
CKPT_FILE_ROOT="${CKPT_FILE_ROOT:-${CKPT_ROOT}}"
#CKPT_FILE_ROOT="/data/data/yulin/NSD/train_logs/"
DATA_PATH="/data/data/yulin/NSD/"
CACHE_DIR="/data/data/shumeng/hub/"
#CKPT_FILE_ROOT="/data/lsm/model_mindeyev2/train_logs/"

for MODEL in "${MODELS[@]}"; do
  if [[ "${MODEL}" =~ subj0*([0-9]+) ]]; then
    SUBJ="${BASH_REMATCH[1]}"
  elif [[ "${MODEL}" =~ subj([0-9]+) ]]; then
    SUBJ="${BASH_REMATCH[1]}"
  else
    SUBJ="${SUBJ:-1}"
  fi
  SUBJ=$((10#${SUBJ}))
  echo "Evaluating subject ${SUBJ}"

  echo "Evaluating ${MODEL} SUBJ ${SUBJ}"
  CUDA_VISIBLE_DEVICES=${GPU} "${ACCELERATE_BIN}" launch --mixed_precision=fp16 "${SCRIPT_DIR}/recon_inference_tuner.py" \
    --ckpt_root="${CKPT_ROOT}" \
    --ckpt_file_root="${CKPT_FILE_ROOT}" \
    --model_name="${MODEL}" \
    --subj="${SUBJ}" \
    --hidden_dim=4096 \
    --enable_spectral_gate="${ENABLE_SPECTRAL_GATE}" \
    --spec_gate_init_bias="${SPEC_GATE_INIT_BIAS}" \
    --enable_learnnorm="${ENABLE_LEARN_NORM}" \
    --data_path="${DATA_PATH}" \
    --cache_dir="${CACHE_DIR}"

  CUDA_VISIBLE_DEVICES=${GPU} "${ACCELERATE_BIN}" launch --mixed_precision=fp16 "${SCRIPT_DIR}/enhanced_recon_inference_tuner.py" \
    --ckpt_root="${CKPT_ROOT}" \
    --model_name="${MODEL}" \
    --subj="${SUBJ}"

  CUDA_VISIBLE_DEVICES=${GPU} "${ACCELERATE_BIN}" launch --mixed_precision=fp16 "${SCRIPT_DIR}/final_evaluations_tuner.py" \
    --ckpt_root="${CKPT_ROOT}" \
    --model_name="${MODEL}" \
    --enhanced_flag=1 \
    --subj="${SUBJ}" \
    --enable_learnnorm="${ENABLE_LEARN_NORM}" \
    --data_path="${DATA_PATH}"  \
    --cache_dir="${CACHE_DIR}"
done
