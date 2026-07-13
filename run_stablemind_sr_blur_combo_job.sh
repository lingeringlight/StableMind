#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

GPU="${1:-0}"
MODEL_STEM="${2:-StableMind_subj01_1se_10bs_SrcBlurCombo}"
SUBJ="${3:-1}"
FUSION_MODE="${4:-interp_distill}"
SRC_ALPHA="${5:-0.10}"
SRC_DISTILL_WEIGHT="${6:-0.05}"
BLUR_MODE="${7:-difficulty_semantic}"
DIFFICULTY_SCORE_MODE="${8:-batch_rel_sim}"
FOVEA_P="${9:-0.70}"
FOVEA_WEIGHT="${10:-0.16}"
DIFFICULTY_CENTER_MIX_MIN="${11:-0.05}"
DIFFICULTY_CENTER_MIX_MAX="${12:-0.18}"
DIFFICULTY_RADIUS_BONUS="${13:-0.18}"
DIFFICULTY_RADIUS_SHRINK="${14:-0.00}"
DIFFICULTY_LOSS_WEIGHT_MIN="${15:-0.50}"
RADIUS_MIN_RATIO="${16:-0.18}"
RADIUS_MAX_RATIO="${17:-0.28}"
DIFFICULTY_BATCH_TEMP="${18:-0.028}"
DIFFICULTY_MARGIN_MIX="${19:-0.0}"

SESS="${SESS:-1}"
BASE_ROOT="${BASE_ROOT:-/data/data/yulin/NSD/train_logs/}"
RESULT_ROOT="${RESULT_ROOT:-/data/data/gjt/MindEyeV2-results/train_logs}"
DATA_PATH="${DATA_PATH:-/data/data/yulin/NSD/}"
CACHE_DIR="${CACHE_DIR:-/data/data/yulin/NSD/}"
SAL="${SAL:-/data/data/gjt/saliency_metadata.csv}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/launcher_logs}"
DIFFICULTY_WARMUP_EPOCH="${DIFFICULTY_WARMUP_EPOCH:-10}"
INIT_CKPT_DIR="${INIT_CKPT_DIR:-${BASE_ROOT}/final_multisubject_subj0${SUBJ}}"
SOURCE_RIDGE_CKPT_DIR="${SOURCE_RIDGE_CKPT_DIR:-${BASE_ROOT}/final_multisubject_subj0${SUBJ}}"
if [[ "${STYLE_PLAN_JSON:-}" == "NONE" ]]; then
  STYLE_PLAN_JSON=""
elif [[ -z "${STYLE_PLAN_JSON:-}" ]]; then
  STYLE_PLAN_JSON='{"0":"mix","1":"mix","2":"none","3":"none","4":"none","5":"fourier","6":"none"}'
fi
FOURIER_MODEL_MODE_OVERRIDE="${FOURIER_MODEL_MODE_OVERRIDE:-0}"
FOURIER_MODEL_ALPHA_OVERRIDE="${FOURIER_MODEL_ALPHA_OVERRIDE:-1.0}"
FOURIER_AMP_MODE_OVERRIDE="${FOURIER_AMP_MODE_OVERRIDE:-global}"
AUG_CONSISTENCY_FLAG_OVERRIDE="${AUG_CONSISTENCY_FLAG_OVERRIDE:-0}"
AUG_CONSISTENCY_WEIGHT_OVERRIDE="${AUG_CONSISTENCY_WEIGHT_OVERRIDE:-0}"
STYLE_LAYER_PROB_OVERRIDE="${STYLE_LAYER_PROB_OVERRIDE:-0.5}"
ALIGN_LOSS_MODE_OVERRIDE="${ALIGN_LOSS_MODE_OVERRIDE:-legacy}"
ALIGN_REL_WEIGHT_OVERRIDE="${ALIGN_REL_WEIGHT_OVERRIDE:-0.05}"
ALIGN_REL_TEMP_OVERRIDE="${ALIGN_REL_TEMP_OVERRIDE:-0.07}"
ALIGN_TEACHER_TEMP_OVERRIDE="${ALIGN_TEACHER_TEMP_OVERRIDE:-0.04}"
LORA_DROPOUT_P_OVERRIDE="${LORA_DROPOUT_P_OVERRIDE:-0.33}"
LORA_TIE_DROPOUT_SCALE_OVERRIDE="${LORA_TIE_DROPOUT_SCALE_OVERRIDE:-1}"
BLUR_SCALE_OVERRIDE="${BLUR_SCALE_OVERRIDE:-.5}"
SOURCE_RIDGE_SUBJECT_MODE_OVERRIDE="${SOURCE_RIDGE_SUBJECT_MODE_OVERRIDE:-mean}"
FOVEA_MODE_OVERRIDE="${FOVEA_MODE_OVERRIDE:-2}"
STYLE_AUG_FLAG_OVERRIDE="${STYLE_AUG_FLAG_OVERRIDE:-0}"
STYLE_AUG_LAYERS_OVERRIDE="${STYLE_AUG_LAYERS_OVERRIDE:-0 1 2 3}"

ACCELERATE_BIN="/home/gjt/.conda/envs/gjt_fmri/bin/accelerate"
PYTHON_BIN="/home/gjt/.conda/envs/gjt_fmri/bin/python"
TRAIN_PY="${SCRIPT_DIR}/Train_Tuner_StableMind_sr_blur_combo.py"
EVAL_SH="${SCRIPT_DIR}/evaluate_tuner.sh"
OUT_LOG="${LOG_DIR}/${MODEL_STEM}_train_eval.out"

mkdir -p "${LOG_DIR}" "${RESULT_ROOT}"

export MINDEYE_TRAIN_LOGS_ROOT="${RESULT_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/tmp/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/huggingface/transformers}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

RESUME_ARGS=()
EXISTING_MODEL_DIR="$(find "${RESULT_ROOT}" -maxdepth 1 -type d -name "${MODEL_STEM}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | awk '{print $2}')"
if [[ -n "${EXISTING_MODEL_DIR}" && -f "${EXISTING_MODEL_DIR}/last.pth" ]]; then
  RESUME_ARGS+=( "--resume_ckpt=${EXISTING_MODEL_DIR}/last.pth" )
fi

{
  echo ">>> GPU=${GPU}"
  echo ">>> MODEL_STEM=${MODEL_STEM}"
  echo ">>> SUBJ=${SUBJ}"
  echo ">>> FUSION_MODE=${FUSION_MODE}"
  echo ">>> SRC_ALPHA=${SRC_ALPHA}"
  echo ">>> SRC_DISTILL_WEIGHT=${SRC_DISTILL_WEIGHT}"
  echo ">>> BLUR_MODE=${BLUR_MODE}"
  echo ">>> DIFFICULTY_SCORE_MODE=${DIFFICULTY_SCORE_MODE}"
  if [[ ${#RESUME_ARGS[@]} -gt 0 ]]; then
    echo ">>> RESUME=${RESUME_ARGS[0]}"
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${TRAIN_PY}" \
    --model_name="${MODEL_STEM}" \
    --subj="${SUBJ}" \
    --num_sessions="${SESS}" \
    --data_path="${DATA_PATH}" \
    --cache_dir="${CACHE_DIR}" \
    --saliency_csv="${SAL}" \
    "${RESUME_ARGS[@]}" \
    --no-multi_subject \
    --batch_size=10 \
    --multisubject_ckpt="${INIT_CKPT_DIR}" \
    --source_ridge_fusion_mode="${FUSION_MODE}" \
    --source_ridge_ckpt="${SOURCE_RIDGE_CKPT_DIR}" \
    --source_ridge_alpha="${SRC_ALPHA}" \
    --source_ridge_distill_weight="${SRC_DISTILL_WEIGHT}" \
    --source_ridge_subject_mode="${SOURCE_RIDGE_SUBJECT_MODE_OVERRIDE}" \
    --max_lr=3e-4 \
    --mixup_pct=0.33 \
    --num_epochs=150 \
    --use_prior \
    --prior_scale=30 \
    --clip_scale=1 \
    --blurry_recon \
    --blur_scale="${BLUR_SCALE_OVERRIDE}" \
    --n_blocks=4 \
    --hidden_dim=4096 \
    --use_image_aug=1 \
    --center_mix=0.18 \
    --color_jitter_flag=0 \
    --fovea_mode="${FOVEA_MODE_OVERRIDE}" \
    --fovea_p="${FOVEA_P}" \
    --fovea_weight="${FOVEA_WEIGHT}" \
    --blur_kernel_size=51 \
    --blur_experiment_mode="${BLUR_MODE}" \
    --radius_scale=0.92 \
    --radius_min_ratio="${RADIUS_MIN_RATIO}" \
    --radius_max_ratio="${RADIUS_MAX_RATIO}" \
    --difficulty_warmup_epoch="${DIFFICULTY_WARMUP_EPOCH}" \
    --difficulty_bank_momentum=0.85 \
    --difficulty_tau=0.28 \
    --difficulty_score_mode="${DIFFICULTY_SCORE_MODE}" \
    --difficulty_batch_temp="${DIFFICULTY_BATCH_TEMP}" \
    --difficulty_margin_mix="${DIFFICULTY_MARGIN_MIX}" \
    --difficulty_power=1.0 \
    --difficulty_default_score=0.5 \
    --difficulty_center_mix_min="${DIFFICULTY_CENTER_MIX_MIN}" \
    --difficulty_center_mix_max="${DIFFICULTY_CENTER_MIX_MAX}" \
    --difficulty_radius_bonus="${DIFFICULTY_RADIUS_BONUS}" \
    --difficulty_radius_shrink="${DIFFICULTY_RADIUS_SHRINK}" \
    --difficulty_loss_weight_min="${DIFFICULTY_LOSS_WEIGHT_MIN}" \
    --finetune_mode=lora+skip \
    --lora_rank=8 \
    --lora_alpha=8 \
    --lora_dropout_p="${LORA_DROPOUT_P_OVERRIDE}" \
    --lora_tie_dropout_scale="${LORA_TIE_DROPOUT_SCALE_OVERRIDE}" \
    --skip_include_align \
    --skip_include_final \
    --style_aug_flag="${STYLE_AUG_FLAG_OVERRIDE}" \
    --style_aug_layers ${STYLE_AUG_LAYERS_OVERRIDE} \
    --style_Style_or_Fourier=0.3 \
    ${STYLE_PLAN_JSON:+--style_aug_plan_json="${STYLE_PLAN_JSON}"} \
    --style_layer_select=0 \
    --style_aug_plan_select_one=0 \
    --style_layer_prob="${STYLE_LAYER_PROB_OVERRIDE}" \
    --Fourier_amp_mode="${FOURIER_AMP_MODE_OVERRIDE}" \
    --Fourier_model_mode="${FOURIER_MODEL_MODE_OVERRIDE}" \
    --Fourier_model_alpha="${FOURIER_MODEL_ALPHA_OVERRIDE}" \
    --aug_consistency_flag="${AUG_CONSISTENCY_FLAG_OVERRIDE}" \
    --aug_consistency_weight="${AUG_CONSISTENCY_WEIGHT_OVERRIDE}" \
    --align_loss_mode="${ALIGN_LOSS_MODE_OVERRIDE}" \
    --align_rel_weight="${ALIGN_REL_WEIGHT_OVERRIDE}" \
    --align_rel_temp="${ALIGN_REL_TEMP_OVERRIDE}" \
    --align_teacher_temp="${ALIGN_TEACHER_TEMP_OVERRIDE}"

  MODEL_DIR="$(find "${RESULT_ROOT}" -maxdepth 1 -type d -name "${MODEL_STEM}_*" -printf '%T@ %p\n' | sort -nr | head -n 1 | awk '{print $2}')"
  if [[ -z "${MODEL_DIR}" ]]; then
    echo "Cannot locate model dir for ${MODEL_STEM}" >&2
    exit 1
  fi
  MODEL_NAME="$(basename "${MODEL_DIR}")"
  echo ">>> MODEL_NAME=${MODEL_NAME}"

  GPU="${GPU}" \
  ACCELERATE_BIN="${ACCELERATE_BIN}" \
  MODEL_NAME="${MODEL_NAME}" \
  CKPT_ROOT="${RESULT_ROOT}/" \
  CKPT_FILE_ROOT="${RESULT_ROOT}/" \
  bash "${EVAL_SH}"
} > "${OUT_LOG}" 2>&1
