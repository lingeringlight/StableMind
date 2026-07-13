# # Import packages & functions

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
import pandas as pd


import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator

from inpainting_images import FoveaBlur, FoveaBlurTorch, FoveaBlurTorch_adaptive, FoveaBlurTorch_Fast_adaptive
# from MindTuner_modules import MindTunerSkipConnector, clip_like_contrastive, AdaptiveProjector
# from MindTuner_modules import MindTunerSkipConnector, clip_like_contrastive, replace_linear_with_lora, enable_lora_params

from MindTuner_modules import LoRALinear, SkipLoRALayer, compute_skip_loss
from fmri_transform import LearnableNorm, SpectralGate, BrainPost, ConditionalLearnableNorm, FourierSBMM


# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# sys.path.append('src/generative_models/')
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
try:
    import wandb
except ImportError:
    wandb = None
# wandb.login(key='4bc22ef1a03ba5c7e846579c683fb22da88fa7ba')


# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils

# methods related
# from models import BrainNetwork
# from models import *
from models_tuner_stablemind import BrainNetwork, PriorNetwork, BrainDiffusionPrior
try:
    from stablemind_duala_sdp import (
        DualaFeatureStatsAugmenter,
        build_imageid_to_label_from_json,
        get_duala_category_names,
        labels_for_image_ids,
    )
except ImportError:
    DualaFeatureStatsAugmenter = None

    def build_imageid_to_label_from_json(*args, **kwargs):
        raise RuntimeError("Duala SDP support is not included in this curated StableMind package.")

    def get_duala_category_names():
        return []

    def labels_for_image_ids(*args, **kwargs):
        raise RuntimeError("Duala SDP support is not included in this curated StableMind package.")


def _map_lora_names(state_dict: dict, prefer_ab: bool = True) -> dict:
    if state_dict is None:
        return state_dict
    mapped = {}
    if prefer_ab:
        repl = ((".lora_A", ".A"), (".lora_B", ".B"))
    else:
        repl = ((".A", ".lora_A"), (".B", ".lora_B"))
    for k, v in state_dict.items():
        nk = k
        for a, b in repl:
            nk = nk.replace(a, b)
        mapped[nk] = v
    return mapped

### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None:
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)

data_type = torch.float16 # change depending on your mixed_precision
num_devices = torch.cuda.device_count()
if num_devices==0: num_devices = 1

# First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
global_batch_size = 32

print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer to enable regularization
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out


def _shorten_model_name_if_needed(model_name, max_len=180, keep_prefix=140):
    if len(model_name) <= max_len:
        return model_name
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:12]
    return f"{model_name[:keep_prefix]}_h{digest}"


def _align_voxel_dim(voxel, target_dim):
    current_dim = voxel.shape[-1]
    if current_dim == target_dim:
        return voxel
    if current_dim > target_dim:
        return voxel[..., :target_dim]
    return F.pad(voxel, (0, target_dim - current_dim))


class SourceRidgeBank(torch.nn.Module):
    def __init__(self, source_linears):
        super().__init__()
        self.linears = torch.nn.ModuleList(source_linears)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, voxel):
        source_feats = []
        for linear in self.linears:
            voxel_aligned = _align_voxel_dim(voxel, linear.in_features)
            source_feats.append(linear(voxel_aligned[:, 0]).unsqueeze(1))
        source_stack = torch.stack(source_feats, dim=1)  # [B, S, 1, H]
        source_mean = source_stack.mean(dim=1)
        return source_mean, source_stack


def build_source_ridge_bank(source_ckpt, hidden_dim, subj, device):
    if source_ckpt is None:
        return None, []
    ckpt_path = os.path.join(source_ckpt, "last.pth") if os.path.isdir(source_ckpt) else source_ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    source_subjects = [s for s in range(1, 9) if s != subj]
    source_linears = []
    for idx, _ in enumerate(source_subjects):
        w_key = f"ridge.linears.{idx}.weight"
        b_key = f"ridge.linears.{idx}.bias"
        if w_key not in state_dict or b_key not in state_dict:
            raise KeyError(f"Missing source ridge params: {w_key} / {b_key} in {ckpt_path}")
        in_features = state_dict[w_key].shape[1]
        linear = torch.nn.Linear(in_features, hidden_dim)
        linear.load_state_dict({"weight": state_dict[w_key], "bias": state_dict[b_key]})
        source_linears.append(linear)
    bank = SourceRidgeBank(source_linears).to(device)
    bank.eval()
    del checkpoint
    return bank, source_subjects


def fuse_source_ridge_feature(voxel_ridge, source_mean, alpha):
    if alpha <= 0:
        return voxel_ridge
    return (1.0 - alpha) * voxel_ridge + alpha * source_mean


def select_source_ridge_feature(source_mean, source_stack, target_ridge, mode):
    if mode == "mean":
        return source_mean

    if mode == "random_one":
        bsz, num_src = source_stack.shape[0], source_stack.shape[1]
        idx = torch.randint(0, num_src, (bsz,), device=source_stack.device)
        return source_stack[torch.arange(bsz, device=source_stack.device), idx]

    if mode == "nearest_one":
        src = F.normalize(source_stack[:, :, 0].float(), dim=-1)
        tgt = F.normalize(target_ridge[:, 0].float(), dim=-1)
        scores = (src * tgt[:, None, :]).sum(dim=-1)
        idx = scores.argmax(dim=1)
        return source_stack[torch.arange(source_stack.shape[0], device=source_stack.device), idx]

    raise ValueError(f"Unknown source_ridge_subject_mode: {mode}")

def mp_infonce_weighted(zb, z_pos_list, pos_weights=None, temp=0.06):
    if pos_weights is None:
        pos_weights = [1.0] * len(z_pos_list)

    z_all_pos = torch.cat(z_pos_list, dim=0)           # [kB, D], k=len(z_pos_list)
    logits = (zb @ z_all_pos.t()) / temp               # [B, kB]

    B = zb.size(0)
    num_pos = len(z_pos_list)

    pos_cols = [torch.arange(B, device=zb.device) + j*B for j in range(num_pos)]

    weighted_pos_scores = []
    for j in range(num_pos):
        s_ij = logits[torch.arange(B), pos_cols[j]]                 # [B]
        weighted_pos_scores.append(s_ij + math.log(max(pos_weights[j], 1e-8)))
    num = torch.logsumexp(torch.stack(weighted_pos_scores, dim=1), dim=1)   # [B]
    den = torch.logsumexp(logits, dim=1)                           # [B]
    return (-(num - den)).mean()


def _soft_ce_from_logits(logits_a, logits_b, temp=0.006):
    target = F.softmax(logits_b / temp, dim=-1)
    loss = -(target * F.log_softmax(logits_a / temp, dim=-1)).sum(dim=-1)
    return loss.mean()


def dual_view_shared_positive_clip_loss(z_view1, z_view2, z_target, temp=0.006, mode="soft", perm=None, betas=None, select=None):
    """Align two stochastic brain views to the same clean image targets."""
    if mode == "mixco":
        if perm is None or betas is None or select is None:
            raise ValueError("mixco mode requires perm / betas / select")
        loss_v1 = utils.mixco_nce(z_view1, z_target, temp=temp, perm=perm, betas=betas, select=select)
        loss_v2 = utils.mixco_nce(z_view2, z_target, temp=temp, perm=perm, betas=betas, select=select)
    else:
        loss_v1 = utils.soft_clip_loss(z_view1, z_target, temp=temp)
        loss_v2 = utils.soft_clip_loss(z_view2, z_target, temp=temp)
    return 0.5 * (loss_v1 + loss_v2)


def dual_view_distribution_consistency_loss(z_view1, z_view2, z_target, temp=0.07, teacher_temp=0.04):
    """Match the two brain-view similarity distributions over one clean image batch."""
    logits_v1 = z_view1 @ z_target.t()
    logits_v2 = z_view2 @ z_target.t()
    logits_teacher = 0.5 * (logits_v1.detach() + logits_v2.detach())
    loss_v1 = _soft_ce_from_logits(logits_v1, logits_teacher, temp=teacher_temp)
    loss_v2 = _soft_ce_from_logits(logits_v2, logits_teacher, temp=teacher_temp)
    return 0.5 * (loss_v1 + loss_v2)


def _normalize_radius_ratios(radii, min_ratio, max_ratio, eps=1e-6):
    span = max(max_ratio - min_ratio, eps)
    return ((radii - min_ratio) / span).clamp(0.0, 1.0)


def _apply_radius_power(radii, min_ratio, max_ratio, power):
    if abs(power - 1.0) < 1e-6:
        return radii
    radii_norm = _normalize_radius_ratios(radii, min_ratio, max_ratio)
    radii_pow = radii_norm.pow(power)
    return min_ratio + radii_pow * (max_ratio - min_ratio)


def _lookup_difficulty_scores(image_ids, difficulty_bank, default_score=0.5, device=None):
    scores = [float(difficulty_bank.get(int(image_id), default_score)) for image_id in image_ids.tolist()]
    return torch.tensor(scores, dtype=torch.float32, device=device)


def _update_difficulty_bank(difficulty_bank, image_ids, easy_scores, momentum=0.85):
    for image_id, easy_score in zip(image_ids.tolist(), easy_scores.tolist()):
        image_id = int(image_id)
        easy_score = float(easy_score)
        prev = difficulty_bank.get(image_id)
        if prev is None:
            difficulty_bank[image_id] = easy_score
        else:
            difficulty_bank[image_id] = momentum * prev + (1.0 - momentum) * easy_score


def _batch_relative_easy_scores(raw_scores, temp=0.03, eps=1e-6):
    z = (raw_scores - raw_scores.mean()) / (raw_scores.std(unbiased=False) + eps)
    return torch.sigmoid(z / max(temp, eps))


def _retrieval_margin_scores(clip_voxels_norm, clip_target_norm):
    sim_mat = clip_voxels_norm @ clip_target_norm.t()
    diag = sim_mat.diag()
    if sim_mat.shape[0] <= 1:
        return diag
    off_diag = sim_mat.masked_fill(
        torch.eye(sim_mat.shape[0], device=sim_mat.device, dtype=torch.bool),
        -1e4,
    )
    return diag - off_diag.max(dim=1).values


def _compute_clean_easy_scores(
    clip_voxels_norm,
    clip_target_norm,
    score_mode="batch_relative",
    batch_temp=0.03,
    tau=0.28,
    margin_mix=0.0,
):
    clean_sim = (clip_voxels_norm.detach() * clip_target_norm.detach()).sum(dim=-1)
    mode = "batch_rel_sim" if score_mode == "batch_relative" else score_mode

    if mode == "batch_rel_sim":
        return _batch_relative_easy_scores(clean_sim, temp=batch_temp)
    if mode == "batch_rel_margin":
        margins = _retrieval_margin_scores(clip_voxels_norm.detach(), clip_target_norm.detach())
        return _batch_relative_easy_scores(margins, temp=batch_temp)
    if mode == "batch_rel_sim_margin":
        easy_sim = _batch_relative_easy_scores(clean_sim, temp=batch_temp)
        margins = _retrieval_margin_scores(clip_voxels_norm.detach(), clip_target_norm.detach())
        easy_margin = _batch_relative_easy_scores(margins, temp=batch_temp)
        return ((1.0 - margin_mix) * easy_sim + margin_mix * easy_margin).clamp(0.0, 1.0)
    if mode == "absolute":
        return ((clean_sim - tau) / max(1e-6, 1.0 - tau)).clamp(0.0, 1.0)

    raise ValueError(f"Unknown difficulty_score_mode: {score_mode}")



def save_ckpt(outdir, tag, epoch):
    ckpt_path = os.path.join(outdir, f'{tag}.pth')
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
        }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag, load_lr=True, load_optimizer=True, load_epoch=True, strict=True, outdir=None,
                multisubj_loading=False):
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir + '/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading:  # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight', None)
    model.load_state_dict(state_dict, strict=strict)


    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch", epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

def resume_ckpt(tag, load_lr=True, load_optimizer=True, load_epoch=True, strict=True, outdir=None,
              multisubj_loading=False, explicit_path=None):
    if explicit_path is not None:
        ckpt_path = explicit_path
    else:
        assert outdir is not None, "Either explicit_path or outdir must be provided."
        ckpt_path = os.path.join(outdir, f"{tag}.pth")

    print(f"\n---loading ckpt from {ckpt_path}---\n")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']

    # state_dict = _map_lora_names(state_dict, prefer_ab=True)

    if multisubj_loading:  # keep your special-case removal
        state_dict.pop('ridge.linears.0.weight', None)
    model.load_state_dict(state_dict, strict=strict)

    if load_epoch and ('epoch' in checkpoint):
        globals()["epoch"] = int(checkpoint['epoch'])
        print("Restored epoch:", epoch)

    if 'train_losses' in checkpoint:
        globals()['losses'] = checkpoint['train_losses']
    if 'test_losses' in checkpoint:
        globals()['test_losses'] = checkpoint['test_losses']
    if 'lrs' in checkpoint:
        globals()['lrs'] = checkpoint['lrs']

    if load_optimizer and ('optimizer_state_dict' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr and ('lr_scheduler' in checkpoint):
        _sched_state = checkpoint.get('lr_scheduler', checkpoint.get('lr_scheduler_state_dict', None))
        if _sched_state is not None:
            lr_scheduler.load_state_dict(_sched_state)

    del checkpoint


def save_logs(outdir, logs, epoch):
    logs_path = os.path.join(outdir, f'logs_epoch_{epoch}.json')
    with open(logs_path, 'a') as f:
        json.dump(logs, f)
    print(f"--- Saved logs to {logs_path} ---")

# # Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="baseline_finetuned_sub01",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--data_path", type=str, default="/data/data/yulin/NSD/",
        help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="/data/data/yulin/NSD/",
        help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
    )
    parser.add_argument(
        "--resume_ckpt", type=str, default=None,
        help="Path to a checkpoint file (*.pth) or a directory containing the ckpt (default tag is set by --resume_tag)."
    )
    parser.add_argument("--saliency_csv",  type=str,
                        default="/ssd/gjt/NSD_CAM/saliency_metadata.csv",
                        help="image_index,center_x,center_y,radius_ratio")
    parser.add_argument(
        "--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--multisubject_ckpt", type=str, default=None,
        help="Path to pre-trained multisubject model to finetune a single subject from. multisubject must be False.",
    )
    parser.add_argument(
        "--source_ridge_fusion_mode", type=str, default="off",
        choices=["off", "interp", "distill", "interp_distill"],
        help="Use frozen source-subject ridge features from multisubject ckpt and fuse/distill them into current-subject ridge features.",
    )
    parser.add_argument(
        "--source_ridge_subject_mode", type=str, default="mean",
        choices=["mean", "random_one", "nearest_one"],
        help="Select source prior by mean, random one-source, or nearest one-source.",
    )
    parser.add_argument(
        "--source_ridge_ckpt", type=str, default=None,
        help="Checkpoint dir/file for frozen source-subject ridge bank. Defaults to --multisubject_ckpt when omitted.",
    )
    parser.add_argument(
        "--source_ridge_alpha", type=float, default=0.25,
        help="Interpolation weight for source ridge mean feature before backbone.",
    )
    parser.add_argument(
        "--source_ridge_distill_weight", type=float, default=0.05,
        help="Cosine distillation weight between current-subject ridge feature and frozen source ridge mean feature.",
    )
    parser.add_argument(
        "--num_sessions", type=int, default=1,
        help="Number of training sessions to include",
    )
    parser.add_argument(
        "--use_prior", action=argparse.BooleanOptionalAction, default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
    )
    parser.add_argument(
        "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--mixup_pct", type=float, default=.33,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=True,
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale", type=float, default=.5,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--clip_scale", type=float, default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--prior_scale", type=float, default=30,
        help="multiply diffusion prior loss by this",
    )
    parser.add_argument(
        "--use_image_aug", default=0, type=int,
        help="whether to use image augmentation",
    )
    parser.add_argument("--center_mix", type=float, default=0.3,
                        help="(1-center_mix)*image center + center_mix*CAM center")

    parser.add_argument(
        "--fovea_weight", type=float, default=0.3,
        help="weight of the loss for CLIP image augmentation",
    )

    parser.add_argument(
        "--fovea_p", type=float, default=0.5,
        help="prob for CLIP image augmentation",
    )

    parser.add_argument(
        "--fovea_mode", type=float, default=0,
        help="1: origanla; 2: adaptive",
    )

    parser.add_argument(
        "--blur_kernel_size", type=int, default=51,
    )
    parser.add_argument("--blur_experiment_mode", type=str, default="fixed_semantic",
                        choices=["center_only", "fixed_semantic", "radius_aware", "saliency_guided", "difficulty_aware", "difficulty_geo", "difficulty_semantic"],
                        help="difficulty_geo: geometry center + difficulty radius/weight; difficulty_semantic: easy samples move toward saliency center.")
    parser.add_argument("--radius_scale", type=float, default=1.0,
                        help="scale factor applied to saliency_csv radius_ratio when converted to pixels")
    parser.add_argument("--radius_min_ratio", type=float, default=0.18,
                        help="minimum radius ratio clamp when using radius-aware blur")
    parser.add_argument("--radius_max_ratio", type=float, default=0.35,
                        help="maximum radius ratio clamp when using radius-aware blur")
    parser.add_argument("--radius_aware_lambda", type=float, default=0.20,
                        help="maximum semantic shift weight for radius-aware center-prior blur")
    parser.add_argument("--radius_power", type=float, default=1.0,
                        help="nonlinear power on normalized radius inside [radius_min_ratio, radius_max_ratio]")
    parser.add_argument("--center_mix_min", type=float, default=0.05,
                        help="minimum semantic center mixing ratio for saliency-guided blur")
    parser.add_argument("--center_mix_max", type=float, default=0.45,
                        help="maximum semantic center mixing ratio for saliency-guided blur")
    parser.add_argument("--radius_center_slope", type=float, default=0.0,
                        help="extra semantic center weight for smaller radii in saliency-guided blur")
    parser.add_argument("--distance_center_slope", type=float, default=0.0,
                        help="extra semantic center weight for CAMs farther from image center")
    parser.add_argument("--difficulty_warmup_epoch", type=int, default=10,
                        help="Epoch to start using online difficulty-aware blur geometry.")
    parser.add_argument("--difficulty_bank_momentum", type=float, default=0.85,
                        help="EMA momentum for per-image difficulty bank.")
    parser.add_argument("--difficulty_tau", type=float, default=0.28,
                        help="Cosine threshold used to convert clean brain-image alignment into an easy-score.")
    parser.add_argument("--difficulty_score_mode", type=str, default="absolute",
                        choices=["absolute", "batch_relative", "batch_rel_sim", "batch_rel_margin", "batch_rel_sim_margin"],
                        help="batch_rel_sim: batch-relative clean cosine; batch_rel_margin: retrieval margin; batch_rel_sim_margin: hybrid.")
    parser.add_argument("--difficulty_batch_temp", type=float, default=0.03,
                        help="Temperature for batch-relative difficulty scoring.")
    parser.add_argument("--difficulty_margin_mix", type=float, default=0.0,
                        help="Hybrid weight for retrieval-margin easy score when --difficulty_score_mode=batch_rel_sim_margin.")
    parser.add_argument("--difficulty_power", type=float, default=1.0,
                        help="Nonlinear power applied to the easy-score before mapping to blur geometry.")
    parser.add_argument("--difficulty_default_score", type=float, default=0.5,
                        help="Default easy-score used before an image has online difficulty statistics.")
    parser.add_argument("--difficulty_center_mix_min", type=float, default=0.06,
                        help="Minimum semantic center mixing ratio for hard samples in difficulty-aware blur.")
    parser.add_argument("--difficulty_center_mix_max", type=float, default=0.24,
                        help="Maximum semantic center mixing ratio for easy samples in difficulty-aware blur.")
    parser.add_argument("--difficulty_radius_bonus", type=float, default=0.20,
                        help="Extra clear-radius multiplier for hard samples in difficulty-aware blur.")
    parser.add_argument("--difficulty_radius_shrink", type=float, default=0.0,
                        help="Extra clear-radius shrink for easy samples in geometry-center difficulty blur.")
    parser.add_argument("--difficulty_loss_weight_min", type=float, default=0.45,
                        help="Minimum batch blur-loss weight ratio when the batch is hard.")

    parser.add_argument(
        "--color_jitter_flag", type=float, default=0,
        help="1: use aug",
    )

    # finetune mode and adapters
    parser.add_argument("--finetune_mode", type=str, default="lora+skip",
                        choices=["full", "lora", "skip-lora", "lora+skip"],
                        help="Finetune strategy: full params, LoRA, Skip-LoRA, or both")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--skip_activation", type=str, default="gelu", choices=["gelu", "relu", "tanh"],
                        help="Activation used in Skip-LoRA")
    parser.add_argument("--skip_loss_weight", type=float, default=1.5,
                        help="Weight for Skip-LoRA Pearson correlation loss")
    parser.add_argument("--skip_rank", type=int, default=None,
                        help="Rank for Skip-LoRA adapters (defaults to --lora_rank if not set)")
    parser.add_argument("--skip_alpha", type=int, default=None,
                        help="Alpha for Skip-LoRA adapters (defaults to --lora_alpha if not set)")
    parser.add_argument("--skip_include_align", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to include an alignment Skip-LoRA (V->h) injected before the first MLP block")
    parser.add_argument("--skip_include_final", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to include a Skip-LoRA adapter on the final mapping to image tokens; Lskip will still exclude the final mapping pair")
    parser.add_argument("--finetune_prior", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to finetune diffusion prior; defaults to False for LoRA-only backbone")
    parser.add_argument("--lora_on_prior", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply LoRA to diffusion prior Transformer as well (experimental)")
    parser.add_argument("--prior_lora_scope", type=str, default="transformer",
                        choices=["transformer", "all"],
                        help="Where to inject LoRA when --lora_on_prior is enabled.")
    parser.add_argument("--prior_cosine_weight", type=float, default=0.0,
                        help="Optional cosine loss on prior output tokens to stabilize subject-specific diffusion decoding.")
    # 检查 lora 注入哪些层

    parser.add_argument("--lora_dropout_p", type=float, default=0.)
    parser.add_argument("--lora_tie_dropout_scale", type=int, default=0)

    # fMRI denoise / alignment toggles.
    parser.add_argument("--enable_spectral_gate", type=int, default=0, help="spectral network")
    parser.add_argument("--spec_gate_init_bias", type=float, default=3.0, help="Sigmoid(bias)≈0.95")
    parser.add_argument("--enable_learnnorm", type=int, default=0, help="1: learnable normalization"
                                                                        "2: input-dependant learnable normalization")
    # for fMRI encoding to fight with overfitting
    parser.add_argument("--channel_drop", type=float, default=0.0, help="prob for using channel dropout")
    parser.add_argument("--channel_layers", type=int, nargs="+", default=[1,2,3,4], help="layers for using channel dropout")
    parser.add_argument("--style_aug_flag", type=int, default=0, help="1: MixStyle; 2: DSU; 3: FourierMix")
    parser.add_argument("--style_Style_or_Fourier", type=float, default=0.5, help="for style_aug_flag 5")
    parser.add_argument("--style_layer_prob", type=float, default=0.5, help="prob for using style aug")
    parser.add_argument("--style_aug_layers", type=int, nargs="+", default=[0,1,2,3,4], help="layers for using channel dropout")
    parser.add_argument("--style_aug_plan_json", type=str, default=None,
                        help="JSON dict for layer-wise style aug plan. Layer ids: 0=input, 1-4=backbone blocks, 5=behind backbone_linear, 6=behind clip_proj. E.g. {\"0\":\"mix\",\"1\":\"fourier\",\"5\":\"mix\"}. Overrides --style_aug_flag when provided.")
    parser.add_argument("--style_aug_plan_select_one", type=int, default=0,
                        help="When using style_aug_plan_json, randomly select only one non-none layer per iteration.")
    parser.add_argument("--style_layer_select", type=int, default=0, help="0: all random; 1: select one")
    parser.add_argument("--Fourier_amp_mode", type=str, default="global", choices=["global", "band", "swap", "beta_mix"])
    parser.add_argument("--Fourier_model_mode", type=int, default=0, help="0: statistic; 1: element")
    parser.add_argument("--Fourier_model_alpha", type=float, default=1.0, help="weight for noise")
    parser.add_argument("--aug_consistency_flag", type=int, default=0, help="1: behind backbone_linear; 2: behind clip proj")
    parser.add_argument("--aug_consistency_weight", type=float, default=0.5, help="weight for consistency")
    parser.add_argument("--align_loss_mode", type=str, default="legacy",
                        choices=["legacy", "dual_view_shared", "dual_view_shared_dist"],
                        help="legacy: original duplicated-batch CLIP loss; dual_view_shared: two brain views share one image target batch; dual_view_shared_dist: also match similarity distributions")
    parser.add_argument("--align_rel_weight", type=float, default=0.05,
                        help="weight for dual-view distribution consistency")
    parser.add_argument("--align_rel_temp", type=float, default=0.07,
                        help="temperature for dual-view distribution consistency")
    parser.add_argument("--align_teacher_temp", type=float, default=0.04,
                        help="teacher temperature for dual-view distribution consistency")
    parser.add_argument("--duala_sdp_stats_dir", type=str, default="",
                        help="Directory with Duala-style tokenwise source-subject statistics.")
    parser.add_argument("--duala_sdp_category_json", type=str, default="",
                        help="Target-subject category_image_idx JSON used to map image ids to semantic labels.")
    parser.add_argument("--duala_sdp_sigma_subjs", type=str, default="1,2,7",
                        help="Comma-separated source subject ids for subjXX_sigma_wrt_global_tokenwise.npz.")
    parser.add_argument("--duala_sdp_p", type=float, default=0.0,
                        help="Probability of applying Duala SDP feature augmentation after clip projection.")
    parser.add_argument("--duala_sdp_alpha", type=float, default=0.25,
                        help="Interpolation strength toward source-subject category variance.")
    parser.add_argument("--duala_sdp_scale_min", type=float, default=0.5)
    parser.add_argument("--duala_sdp_scale_max", type=float, default=2.0)



    parser.add_argument(
        "--num_epochs", type=int, default=150,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--multi_subject", action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--new_test", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--n_blocks", type=int, default=4,
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=4096,
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
    )
    parser.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--ckpt_interval", type=int, default=5,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-4,
    )

    args = parser.parse_args()

    args.model_name += "_" + args.finetune_mode
    if args.finetune_mode != "full":
        args.model_name += str(args.lora_rank) + "_al" + str(args.lora_alpha)
    if "skip" in args.finetune_mode:
        args.model_name += "_sk" + str(args.skip_loss_weight)
        if args.skip_include_align:
            args.model_name += "_A"
        if args.skip_include_final:
            args.model_name += "_F"

    if args.finetune_prior:
        args.model_name += "_FineTunePrior"
    if args.lora_on_prior:
        args.model_name += "_LoraPrior"
        if args.prior_lora_scope == "transformer":
            args.model_name += "Tr"
    if args.prior_cosine_weight > 0:
        args.model_name += "_PrCos" + str(args.prior_cosine_weight)

    if args.use_image_aug != 0:
        if args.color_jitter_flag == 1:
            args.model_name += "_CA"
        if args.fovea_mode != 0:
            args.model_name += "_OABlur" + str(args.fovea_p) + "_W" + str(args.fovea_weight)
            if args.fovea_mode == 2:
                args.model_name += f"_Ada{args.center_mix}_{args.blur_experiment_mode}"
                if args.blur_experiment_mode == "radius_aware":
                    args.model_name += (
                        f"_Rsc{args.radius_scale}_Ral{args.radius_aware_lambda}_Rp{args.radius_power}"
                    )
                if args.blur_experiment_mode == "saliency_guided":
                    args.model_name += (
                        f"_Rsc{args.radius_scale}_Rp{args.radius_power}_Cm{args.center_mix_min}-{args.center_mix_max}"
                        f"_Rcs{args.radius_center_slope}_Dcs{args.distance_center_slope}"
                    )
                if args.blur_experiment_mode in ("difficulty_aware", "difficulty_geo", "difficulty_semantic"):
                    args.model_name += (
                        f"_Rsc{args.radius_scale}_Dt{args.difficulty_tau}_Dp{args.difficulty_power}"
                        f"_Dcm{args.difficulty_center_mix_min}-{args.difficulty_center_mix_max}"
                        f"_Drb{args.difficulty_radius_bonus}_Drs{args.difficulty_radius_shrink}"
                        f"_Dlw{args.difficulty_loss_weight_min}"
                    )
                    if args.difficulty_score_mode != "absolute":
                        args.model_name += f"_Dsm{args.difficulty_score_mode}_Dbt{args.difficulty_batch_temp}"
                        if args.difficulty_score_mode == "batch_rel_sim_margin":
                            args.model_name += f"_Dmm{args.difficulty_margin_mix}"

    if args.lora_dropout_p != 0.:
        args.model_name += "_LoDrop" + str(args.lora_dropout_p)

    if args.enable_spectral_gate != 0:
        args.model_name += "_SpectralGate" + str(args.spec_gate_init_bias)
    if args.enable_learnnorm != 0:
        if args.enable_learnnorm == 1:
            args.model_name += "_NewLearnNorm"
        elif args.enable_learnnorm == 2:
            args.model_name += "_ConditionalLN"
        elif args.enable_learnnorm == 3:
            args.model_name += "_FourierSBMM"

    if args.channel_drop != 0.0:
        args.model_name += "_CDrop" + str(args.channel_drop) + "_L"
        for layer in args.channel_layers:
            args.model_name += str(layer)

    if args.style_aug_flag != 0:
        if args.style_aug_flag == 1:
            args.model_name += "_MSty"
        elif args.style_aug_flag == 2:
            args.model_name += "_DSU"
        elif args.style_aug_flag == 3:
            args.model_name += "_FMix_" + args.Fourier_amp_mode
        elif args.style_aug_flag == 4:
            args.model_name += "_FModel_"
            if args.Fourier_model_mode == 0:
                args.model_name += "_S_"
            else:
                args.model_name += "_E_"
            args.model_name += str(args.Fourier_model_alpha)
        elif args.style_aug_flag == 5:
            args.model_name += "_FMix_or"
            args.model_name += "_FModel_P" + str(args.style_Style_or_Fourier)
            if args.Fourier_model_mode == 0:
                args.model_name += "_S_"
            else:
                args.model_name += "_E_"
            args.model_name += str(args.Fourier_model_alpha)

        elif args.style_aug_flag == 6:
            args.model_name += "_FMix" + "_FModel"
            if args.Fourier_model_mode == 0:
                args.model_name += "_S_"
            else:
                args.model_name += "_E_"
            args.model_name += str(args.Fourier_model_alpha)

        elif args.style_aug_flag == 7:
            args.model_name += "_MixStyle_or"
            args.model_name += "_FModel_P" + str(args.style_Style_or_Fourier)
            if args.Fourier_model_mode == 0:
                args.model_name += "_S_"
            else:
                args.model_name += "_E_"
            args.model_name += str(args.Fourier_model_alpha)

        elif args.style_aug_flag == 8:
            args.model_name += "_MixStyle" + "_FModel"
            if args.Fourier_model_mode == 0:
                args.model_name += "_S_"
            else:
                args.model_name += "_E_"
            args.model_name += str(args.Fourier_model_alpha)

        args.model_name += "_P" + str(args.style_layer_prob)
        for layer in args.style_aug_layers:
            args.model_name += str(layer)
        if args.style_layer_select == 1:
            args.model_name += "_SeOne"

    if args.duala_sdp_p > 0 and args.duala_sdp_stats_dir:
        args.model_name += f"_DualaSDPp{args.duala_sdp_p}a{args.duala_sdp_alpha}"

    style_aug_plan = None
    effective_style_aug_plan_select_one = bool(args.style_aug_plan_select_one)
    if args.style_aug_plan_json is not None:
        try:
            style_aug_plan = json.loads(args.style_aug_plan_json)
            style_aug_plan = {int(k): str(v) for k, v in style_aug_plan.items()}
        except Exception as e:
            raise ValueError(f"Failed to parse --style_aug_plan_json: {e}")

        # Backward-compatibility: older launchers used --style_layer_select=1 to mean
        # "pick only one augmentation layer per iteration". When switching to
        # --style_aug_plan_json, that intent should not silently disappear.
        if not effective_style_aug_plan_select_one and args.style_layer_select == 1:
            effective_style_aug_plan_select_one = True

        args.model_name += "_Plan"
        for k in sorted(style_aug_plan.keys()):
            args.model_name += f"_{k}{style_aug_plan[k]}"
        args.model_name += "_P" + str(args.style_layer_prob)
        if effective_style_aug_plan_select_one:
            args.model_name += "_SelOne"

    if args.aug_consistency_flag != 0:
        args.model_name += "_Cons"
        if args.aug_consistency_flag == 1:
            args.model_name += "_Li"
        elif args.aug_consistency_flag == 2:
            args.model_name += "_Pr"
        args.model_name += str(args.aug_consistency_weight)
        if args.align_loss_mode != "legacy":
            args.model_name += "_Align" + args.align_loss_mode
            if args.align_loss_mode == "dual_view_shared_dist":
                args.model_name += "_Rw" + str(args.align_rel_weight)

    if args.source_ridge_fusion_mode != "off":
        args.model_name += "_SrcRidge" + args.source_ridge_fusion_mode
        if args.source_ridge_subject_mode != "mean":
            args.model_name += "_SrcSel" + args.source_ridge_subject_mode
        args.model_name += "_A" + str(args.source_ridge_alpha)
        args.model_name += "_Dw" + str(args.source_ridge_distill_weight)

    args.model_name = _shorten_model_name_if_needed(args.model_name)

    seed = args.seed
    utils.seed_everything(seed)
    train_logs_root = os.environ.get(
        "MINDEYE_TRAIN_LOGS_ROOT",
        os.path.join(os.getcwd(), "train_logs"),
    )
    outdir = os.path.abspath(os.path.join(train_logs_root, args.model_name))
    if not os.path.exists(outdir) and args.ckpt_saving:
        os.makedirs(outdir, exist_ok=True)

    # preprocess
    if args.use_image_aug != 0 or args.blurry_recon:
        import kornia
        from kornia.augmentation.container import AugmentationSequential
    print("flur aug:", args.use_image_aug)

    if args.use_image_aug != 0:
        if args.use_image_aug == 1:
            if args.color_jitter_flag == 1:
                img_augment = AugmentationSequential(
                    kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
                    same_on_batch=False,
                    data_keys=["input"],
                )
            if args.fovea_mode == 1:
                fovea_blur = AugmentationSequential(
                    FoveaBlurTorch(h=224, w=224, blur_kernel_size=args.blur_kernel_size, curve_type='exp', system_g=3),
                    same_on_batch=False,
                    data_keys=["input"],
                )
            elif args.fovea_mode == 2:
                fovea_blur = FoveaBlurTorch_Fast_adaptive(h=224, w=224, blur_kernel_size=args.blur_kernel_size, curve_type='exp', system_g=3)

        # elif args.use_image_aug == 2:
        #     fovea_blur = FoveaBlurTorch_adaptive(h=224, w=224, blur_kernel_size=51, curve_type='exp', system_g=3)
        #     # img_augment = AugmentationSequential(
        #     #     FoveaBlurTorch_Fast_adaptive(h=224, w=224, blur_kernel_size=51, curve_type='exp', system_g=3),
        #     #     same_on_batch=False,
        #     #     data_keys=["input"],
        #     # )

    # if args.aug_consistency_flag != 0:
    #     global_batch_size = int(args.batch_size / 2)
    global_batch_size = args.batch_size
    per_batch_size = max(1, global_batch_size // accelerator.num_processes)

    if args.multi_subject:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != args.subj]
    else:
        subj_list = [args.subj]
    print("subj_list", subj_list, "num_sessions", args.num_sessions)

    # # Prep data, models, and dataloaders
    # ### Creating wds dataloader, preload betas and all 73k possible images
    def my_split_by_node(urls): return urls

    # def my_split_by_node(urls):
    #     return wds.split_by_node(urls)

    num_voxels_list = []
    if args.multi_subject:
        nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
        num_samples_per_epoch = (750 * 40) // num_devices
    else:
        num_samples_per_epoch = (750 * args.num_sessions) // num_devices
    print("dividing batch size by subj_list, which will then be concatenated across subj during training...")
    batch_size = per_batch_size // len(subj_list)

    num_iterations_per_epoch = num_samples_per_epoch // (batch_size * len(subj_list))
    print("batch_size =", batch_size, "num_iterations_per_epoch =", num_iterations_per_epoch, "num_samples_per_epoch =",
          num_samples_per_epoch)

    train_data = {}
    train_dl = {}
    num_voxels = {}
    voxels = {}
    for s in subj_list:
        print(f"Training with {args.num_sessions} sessions")
        if args.multi_subject:
            train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s - 1] - 1}" + "}.tar"
        else:
            train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{args.num_sessions - 1}" + "}.tar"
        print(train_url)

        train_data[f'subj0{s}'] = wds.WebDataset(train_url, resampled=True,
                                                 nodesplitter=my_split_by_node) \
            .shuffle(750, initial=1500, rng=random.Random(42)) \
            .decode("torch") \
            .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
                    olds_behav="olds_behav.npy") \
            .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
        # train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size,
        #                                                     num_workers=1,
        #                                                     shuffle=False, drop_last=False, pin_memory=True)
        train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size,
                                                            shuffle=False, drop_last=True, pin_memory=True,
                                                            num_workers=4, persistent_workers=True, prefetch_factor=2)

        f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
        betas = f['betas'][:]
        betas = torch.Tensor(betas).to("cpu").to(data_type)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f'subj0{s}'] = betas[0].shape[-1]
        voxels[f'subj0{s}'] = betas
        print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")
    print("Loaded all subj train dls and betas!\n")

    # Validate only on one subject
    if args.multi_subject:
        subj = subj_list[0]  # cant validate on the actual held out person so picking first in subj_list
    if not args.new_test:  # using old test set from before full dataset released (used in original MindEye paper)
        if args.subj == 3:
            num_test = 2113
        elif args.subj == 4:
            num_test = 1985
        elif args.subj == 6:
            num_test = 2113
        elif args.subj == 8:
            num_test = 1985
        else:
            num_test = 2770
        test_url = f"{args.data_path}/wds/subj0{args.subj}/test/" + "0.tar"
    elif args.new_test:  # using larger test set from after full dataset released
        if args.subj == 3:
            num_test = 2371
        elif args.subj == 4:
            num_test = 2188
        elif args.subj == 6:
            num_test = 2371
        elif args.subj == 8:
            num_test = 2188
        else:
            num_test = 3000
        test_url = f"{args.data_path}/wds/subj0{args.subj}/new_test/" + "0.tar"
    print(test_url)
    test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
        .shuffle(750, initial=1500, rng=random.Random(42)) \
        .decode("torch") \
        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
                olds_behav="olds_behav.npy") \
        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test,
                                          num_workers=1,
                                          shuffle=False, drop_last=True,
                                          pin_memory=True)
    print(f"Loaded test dl for subj{args.subj}!\n")

    # Load 73k NSD images
    f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
    # images = f['images']
    images = f['images'][:]  # 加载到numpy数组（CPU）
    images = torch.from_numpy(images).to(data_type)  # 转为Torch Tensor，默认在CPU上
    print("Loaded all 73k possible NSD images to cpu!", images.shape)


    if args.fovea_mode == 2:
        # Load 73k saliency indexes
        if "gaze_tm.csv" in args.saliency_csv:
            pass
        else:
            sal_df = pd.read_csv(args.saliency_csv, usecols=["image_index", "center_x", "center_y", "radius_ratio"])
            N_total = images.shape[0]
            centers_lookup = np.zeros((N_total, 2), dtype=np.float32)
            radii_lookup = np.zeros((N_total,), dtype=np.float32)
            centers_lookup[sal_df["image_index"].values, 0] = sal_df["center_x"].values
            centers_lookup[sal_df["image_index"].values, 1] = sal_df["center_y"].values
            radii_lookup[sal_df["image_index"].values] = sal_df["radius_ratio"].values
    # ## Load models

    # ### CLIP image embeddings  model
    openclip_cache_dir = "/data/data/shumeng/hub/"
    openclip_local_ckpt = os.path.join(
        openclip_cache_dir,
        "models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "blobs",
        "0d5318839ad03607c48055c45897c655a14c0276a79f6b867934ddd073760e39",
    )
    openclip_pretrained = (
        openclip_local_ckpt
        if os.path.isfile(openclip_local_ckpt)
        else "laion2b_s39b_b160k"
    )
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version=openclip_pretrained,
        output_tokens=True,
        only_tokens=True,
        cache_dir=openclip_cache_dir,
    )
    clip_img_embedder.to(device)
    clip_seq_dim = 256
    clip_emb_dim = 1664

    # ### SD VAE
    if args.blurry_recon:
        from diffusers import AutoencoderKL
        autoenc = AutoencoderKL(
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            sample_size=256,
        )
        ckpt = torch.load(f'{args.cache_dir}/sd_image_var_autoenc.pth')
        autoenc.load_state_dict(ckpt)
        autoenc.eval()
        autoenc.requires_grad_(False)
        autoenc.to(device)
        utils.count_params(autoenc)

        from autoencoder.convnext import ConvnextXL
        cnx = ConvnextXL(f'{args.cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)
        blur_augs = AugmentationSequential(
            kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
            kornia.augmentation.RandomGrayscale(p=0.1),
            kornia.augmentation.RandomSolarize(p=0.1),
            kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
            data_keys=["input"],
        )

    # ### MindEye modules, map each subject signals with different dims to the same dim out_features
    # num_voxels_list: dims for each subject
    model = MindEyeModule()

    model.ridge = RidgeRegression(num_voxels_list, out_features=args.hidden_dim)
    utils.count_params(model.ridge)
    utils.count_params(model)

    # test on subject 1 with fake data
    b = torch.randn((2, 1, num_voxels_list[0]))
    print(b.shape, model.ridge(b, 0).shape)
    model.backbone = BrainNetwork(h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=1, n_blocks=args.n_blocks,
                                  clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
                                  blurry_recon=args.blurry_recon, clip_scale=args.clip_scale,
                                  channel_drop=args.channel_drop, channel_layers=args.channel_layers,
                                  style_aug_flag=args.style_aug_flag, style_aug_layers=args.style_aug_layers,
                                  style_Style_or_Fourier=args.style_Style_or_Fourier,
                                  style_layer_select=args.style_layer_select,
                                  style_layer_prob=args.style_layer_prob,
                                  Fourier_amp_mode=args.Fourier_amp_mode,
                                  enable_learnnorm=args.enable_learnnorm,
                                  Fourier_model_mode=args.Fourier_model_mode,
                                  Fourier_model_alpha=args.Fourier_model_alpha,
                                  style_aug_plan=style_aug_plan,
                                  style_aug_plan_prob=args.style_layer_prob,
                                  style_aug_plan_select_one=effective_style_aug_plan_select_one
                                  )

    utils.count_params(model.backbone)
    utils.count_params(model)

    # test that the model works on some fake data
    b = torch.randn((2, 1, args.hidden_dim))
    print("b.shape", b.shape)

    backbone_, clip_, blur_ = model.backbone(b)
    print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)

    # ### Adding diffusion prior + unCLIP if use_prior=True
    if args.use_prior:
        # setup diffusion prior network
        out_dim = clip_emb_dim
        depth = 6
        dim_head = 52
        heads = clip_emb_dim // 52  # heads * dim_head = clip_emb_dim
        timesteps = 100

        prior_network = PriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens=clip_seq_dim,
            learned_query_mode="pos_emb"
        )

        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

        utils.count_params(model.diffusion_prior)
        utils.count_params(model)

    # load models
    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()

    if args.multisubject_ckpt is not None and args.resume_ckpt is None:
        load_ckpt("last", outdir=args.multisubject_ckpt,
                  load_lr=False, load_optimizer=False, load_epoch=False,
                  strict=False, multisubj_loading=True)

    source_ridge_bank = None
    source_subjects = []
    if args.source_ridge_fusion_mode != "off":
        if len(subj_list) != 1:
            raise ValueError("source_ridge_fusion_mode currently supports single-subject finetuning only.")
        source_ridge_ckpt = args.source_ridge_ckpt or args.multisubject_ckpt
        if source_ridge_ckpt is None:
            raise ValueError("source_ridge_fusion_mode requires --source_ridge_ckpt or --multisubject_ckpt.")
        source_ridge_bank, source_subjects = build_source_ridge_bank(
            source_ridge_ckpt,
            hidden_dim=args.hidden_dim,
            subj=args.subj,
            device=device,
        )
        print(
            f"Loaded frozen source ridge bank from {source_ridge_ckpt} "
            f"with source subjects {source_subjects}; "
            f"fusion_mode={args.source_ridge_fusion_mode}, alpha={args.source_ridge_alpha}, "
            f"distill_weight={args.source_ridge_distill_weight}, "
            f"subject_mode={args.source_ridge_subject_mode}"
        )

    duala_sdp_aug = None
    duala_id2lab = None
    duala_category_names = get_duala_category_names() if DualaFeatureStatsAugmenter is not None else []
    if args.duala_sdp_p > 0 and args.duala_sdp_stats_dir:
        if DualaFeatureStatsAugmenter is None:
            raise ImportError(
                "Duala SDP arguments were enabled, but this curated StableMind package excludes Duala support."
            )
        sigma_subjs = [
            int(x.strip()) for x in args.duala_sdp_sigma_subjs.split(",")
            if x.strip()
        ]
        category_json = args.duala_sdp_category_json
        if not category_json:
            category_json = os.path.join(args.data_path, f"category_image_idx_subj{args.subj}.json")
        duala_id2lab = build_imageid_to_label_from_json(category_json, duala_category_names)
        if not duala_id2lab:
            raise ValueError(f"Duala SDP enabled but no category labels loaded from {category_json}")
        duala_sdp_aug = DualaFeatureStatsAugmenter(
            stats_dir=args.duala_sdp_stats_dir,
            clip_seq_dim=clip_seq_dim,
            device=device,
            alpha=args.duala_sdp_alpha,
            p=args.duala_sdp_p,
            scale_min=args.duala_sdp_scale_min,
            scale_max=args.duala_sdp_scale_max,
            category_names=duala_category_names,
            sigma_subjs=sigma_subjs,
        )
        print(
            f"Loaded Duala SDP stats from {args.duala_sdp_stats_dir}; "
            f"labels={len(duala_id2lab)}, sigma_subjs={sigma_subjs}, "
            f"p={args.duala_sdp_p}, alpha={args.duala_sdp_alpha}"
        )


    def freeze_module_params(module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def mark_only_lora_trainable(module: nn.Module):
        for n, p in module.named_parameters():
            p.requires_grad = getattr(p, 'is_lora', False) or ('lora_' in n)

    def count_trainable_params(module: nn.Module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    if args.finetune_mode in ("lora", "lora+skip", "skip-lora"):
        # Replace Linear layers with LoRA and/or attach Skip-LoRA
        if args.finetune_mode in ("lora", "lora+skip"):
            model.backbone.enable_lora(rank=args.lora_rank, alpha=args.lora_alpha,
                                       dropout_p=args.lora_dropout_p, tie_dropout_scale=args.lora_tie_dropout_scale,
                                       )
        if args.finetune_mode in ("skip-lora", "lora+skip"):
            # raw voxel dim for current subject
            v_in_dim = num_voxels[f'subj0{args.subj}']
            # allow independent skip rank/alpha; fallback to lora_rank/alpha if not provided
            rank_to_use = args.skip_rank if (args.skip_rank is not None) else args.lora_rank
            alpha_to_use = args.skip_alpha if (args.skip_alpha is not None) else args.lora_alpha
            model.backbone.enable_skip_lora(
                v_in_dim=v_in_dim,
                activation=args.skip_activation,
                rank=rank_to_use,
                alpha=alpha_to_use,
                include_final=args.skip_include_final,
                include_align=args.skip_include_align,
                dropout_p=args.lora_dropout_p,
                tie_dropout_scale=args.lora_tie_dropout_scale,
            )
        # freeze all base params of backbone; train only LoRA/Skip-LoRA
        freeze_module_params(model.backbone)
        # re-enable LoRA / Skip-LoRA params
        for n, p in model.backbone.named_parameters():
            is_lora_param = getattr(p, 'is_lora', False) or ('lora_' in n)
            is_skip_param = ('_skip_adapter' in n) or ('_skip_adapters' in n)
            if is_lora_param or is_skip_param:
                p.requires_grad = True
        # keep RidgeRegression trainable (subject adapter)
        for p in model.ridge.parameters():
            p.requires_grad = True

        # diffusion prior: freeze by default unless user opts in
        if args.use_prior:
            if not args.finetune_prior:
                freeze_module_params(model.diffusion_prior)
            elif args.lora_on_prior:
                # Train only LoRA adapters inside the diffusion prior. The old
                # path left non-LoRA prior weights trainable, which made the
                # experiment much less controlled than intended.
                try:
                    from MindTuner_modules import _replace_linear_with_lora
                    freeze_module_params(model.diffusion_prior)
                    prior_lora_target = model.diffusion_prior
                    if args.prior_lora_scope == "transformer":
                        prior_lora_target = model.diffusion_prior.net.causal_transformer
                    _replace_linear_with_lora(
                        prior_lora_target,
                        rank=args.lora_rank,
                        alpha=args.lora_alpha,
                        dropout_p=args.lora_dropout_p,
                        tie_dropout_scale=args.lora_tie_dropout_scale,
                    )
                except Exception as e:
                    print("[WARN] Failed to apply LoRA to diffusion prior:", e)

        # report trainable param count (breakdown) AFTER freezing/unfreezing prior
        backbone_trainable = count_trainable_params(model.backbone)
        ridge_trainable = count_trainable_params(model.ridge)
        prior_trainable = count_trainable_params(model.diffusion_prior) if args.use_prior else 0
        total_trainable = backbone_trainable + ridge_trainable + prior_trainable
        # breakdown inside backbone: LoRA vs Skip-LoRA
        lora_only = 0
        skip_only = 0
        for n, p in model.backbone.named_parameters():
            if not p.requires_grad:
                continue
            if 'lora_' in n:
                lora_only += p.numel()
            elif ('_skip_adapter' in n) or ('_skip_adapters' in n):
                skip_only += p.numel()
        print(
            "Trainable params breakdown (M):\n"
            f"  - backbone (total adapters): {backbone_trainable / 1e6:.2f}M\n"
            f"      • LoRA-only: {lora_only / 1e6:.2f}M\n"
            f"      • Skip-LoRA-only: {skip_only / 1e6:.2f}M\n"
            f"  - ridge: {ridge_trainable / 1e6:.2f}M\n"
            f"  - prior: {prior_trainable / 1e6:.2f}M\n"
            f"  = TOTAL: {total_trainable / 1e6:.2f}M"
        )

    #  set spectral gate network
    if args.enable_spectral_gate:
        max_V = max(num_voxels_list) if not args.multi_subject else max(num_voxels.values())
        model.spec_gate_vox = SpectralGate(D=int(max_V),
                                           init_bias=args.spec_gate_init_bias)
    else:
        model.spec_gate_vox = nn.Identity()

    # if args.multisubject_ckpt is not None and args.resume_ckpt is None:
    #     load_ckpt("last", outdir=args.multisubject_ckpt,
    #               load_lr=False, load_optimizer=False, load_epoch=False,
    #               strict=False, multisubj_loading=True)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # able_name = [n for n, p in model.named_parameters() if p.requires_grad]
    # print(trainable_name)

    def trainable_params(module):
        return [p for p in module.parameters() if p.requires_grad]

    opt_grouped_parameters = [
        {'params': trainable_params(model.ridge), 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if
                    p.requires_grad and (not any(nd in n for nd in no_decay))], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if
                    p.requires_grad and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0},
    ]
    if args.use_prior:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if
                        p.requires_grad and (not any(nd in n for nd in no_decay))], 'weight_decay': 1e-2},
            {'params': [p for n, p in model.diffusion_prior.named_parameters() if
                        p.requires_grad and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
        ])
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

    if args.lr_scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs * num_iterations_per_epoch)),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps = int(np.floor(args.num_epochs * num_iterations_per_epoch))
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2 / args.num_epochs
        )

    print("\nDone with model preparations!")
    num_params = utils.count_params(model)

    # # Weights and Biases
    if local_rank == 0 and args.wandb_log:  # only use main process for wandb logging
        import wandb
        wandb_project = 'mindeye'
        print(f"wandb {wandb_project} run {args.model_name}")
        # need to configure wandb beforehand in terminal with "wandb init"!
        wandb_config = {
            "model_name": args.model_name,
            "global_batch_size": global_batch_size,
            "batch_size": batch_size,
            "num_epochs": args.num_epochs,
            "num_sessions": args.num_sessions,
            "num_params": num_params,
            "clip_scale": args.clip_scale,
            "prior_scale": args.prior_scale,
            "blur_scale": args.blur_scale,
            "use_image_aug": args.use_image_aug,
            "max_lr": args.max_lr,
            "mixup_pct": args.mixup_pct,
            "num_samples_per_epoch": num_samples_per_epoch,
            "num_test": num_test,
            "ckpt_interval": args.ckpt_interval,
            "ckpt_saving": args.ckpt_saving,
            "seed": seed,
            "distributed": distributed,
            "num_devices": num_devices,
            "world_size": world_size,
            "train_url": train_url,
            "test_url": test_url,
        }
        print("wandb_config:\n", wandb_config)
        print("wandb_id:", args.model_name)
        wandb.init(
            id=args.model_name,
            project=wandb_project,
            name=args.model_name,
            config=wandb_config,
            resume="allow",
        )
    else:
        args.wandb_log = False

    loaded_from_resume = False
    resume_ckpt_file = None
    if args.resume_ckpt is not None:
        print("resume from: ", args.resume_ckpt)
        if os.path.isdir(args.resume_ckpt):
            resume_ckpt_file = os.path.join(args.resume_ckpt, "last.pth")
        else:
            resume_ckpt_file = args.resume_ckpt
        resume_ckpt(tag="last", outdir=None, explicit_path=resume_ckpt_file,
                  load_lr=True, load_optimizer=True, load_epoch=True, strict=True,
                  multisubj_loading=False)
        loaded_from_resume = True
    print(epoch)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)

    train_dls = [train_dl[f'subj0{s}'] for s in subj_list]
    model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
    # leaving out test_dl since we will only have local_rank 0 device do evals

    start_epoch = int(epoch) if loaded_from_resume else 0
    if loaded_from_resume:
        start_epoch = min(start_epoch + 1, args.num_epochs - 1)

    print(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")
    progress_bar = tqdm(range(start_epoch, args.num_epochs), ncols=1200, disable=(local_rank!=0))
    test_image, test_voxel = None, None
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))

    def abs_pearson_corr(x, y, eps=1e-6):
        """
        Compute absolute Pearson correlation per-sample between two tensors of shape (B, D), return (B,).
        """
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1)
        x_norm = torch.sqrt((x * x).sum(dim=-1) + eps)
        y_norm = torch.sqrt((y * y).sum(dim=-1) + eps)
        corr = xy / (x_norm * y_norm + eps)
        return corr.abs()

    difficulty_bank = {}
    for epoch in progress_bar:
        model.train()

        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.

        recon_cossim = 0.
        test_recon_cossim = 0.
        recon_mse = 0.
        test_recon_mse = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.

        loss_prior_total = 0.
        loss_prior_cos_total = 0.
        test_loss_prior_total = 0.

        loss_cons_total = 0.
        loss_align_rel_total = 0.

        loss_skip_total = 0.
        loss_source_ridge_distill_total = 0.
        difficulty_easy_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        voxel_iters = {} # empty dict because diff subjects have differing # of voxels
        image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
        centers_iters = torch.zeros(num_iterations_per_epoch, batch_size * len(subj_list), 2).float()
        radii_iters = torch.zeros(num_iterations_per_epoch, batch_size * len(subj_list)).float()
        image_id_iters = torch.zeros(num_iterations_per_epoch, batch_size * len(subj_list), dtype=torch.long)
        annot_iters = {}
        perm_iters, betas_iters, select_iters = {}, {}, {}
        for s, train_dl in enumerate(train_dls):
            with torch.cuda.amp.autocast(dtype=data_type):
                iter = -1
                for behav0, past_behav0, future_behav0, old_behav0 in train_dl:
                    # Load images to cpu from hdf5 (requires sorted indexing)

                    image_idx = behav0[:,0,0].cpu().long().numpy()  # used to index coco_images_224_float16.hdf5
                    image0_t, image_sorted_idx = np.unique(image_idx, return_index=True)
                    if len(image0_t) != len(image_idx): # hdf5 cant handle duplicate indexing
                        continue
                    iter += 1
                    image0 = torch.tensor(images[image0_t], dtype=data_type)  # (B, 3, 224, 224)
                    image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0 # (46, B, 3, 224, 224)
                    image_id_iters[iter, s * batch_size:s * batch_size + batch_size] = torch.from_numpy(image0_t).to(torch.long)

                    if args.fovea_mode == 2:
                        centers_np = centers_lookup[image0_t]  # (B,2) np.float32
                        radii_np = radii_lookup[image0_t]  # (B,) np.float32
                        centers_t = torch.from_numpy(centers_np).to(torch.float32)  # (B,2) torch
                        radii_t = torch.from_numpy(radii_np).to(torch.float32)
                        centers_iters[iter, s * batch_size:s * batch_size + batch_size] = centers_t
                        radii_iters[iter, s * batch_size:s * batch_size + batch_size] = radii_t

                    # Load voxels for current batch, matching above indexing, bhav0: B, 1, 17
                    voxel_idx = behav0[:,0,5].cpu().long().numpy()  # used to index betas_all_subj_fp32_renorm.hdf5
                    voxel_sorted_idx = voxel_idx[image_sorted_idx]
                    voxel0 = voxels[f'subj0{subj_list[s]}'][voxel_sorted_idx]
                    voxel0 = torch.Tensor(voxel0).unsqueeze(1)  # B, 1 15764 (fMRI signals)

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        voxel0, perm, betas, select = utils.mixco(voxel0)
                        perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                        betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                        select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                    voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                    if iter >= num_iterations_per_epoch-1:
                        break

        img_center = torch.tensor([112.0, 112.0], dtype=torch.float32, device=device)
        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
        for train_i in range(num_iterations_per_epoch):
            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()
                loss=0.

                voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device, non_blocking=True) for s in subj_list]    # [Bx1x15724, ]
                image = image_iters[train_i].detach()
                image = image.to(device, non_blocking=True)    # 16x3x224x224
                image_ids_base = image_id_iters[train_i].detach().to(device, non_blocking=True)

                voxel_for_ridge = []
                if args.enable_spectral_gate == 1:
                    for si, s in enumerate(subj_list):
                        v = voxel_list[si]  # [B,1,V_si]
                        V_si = v.shape[-1]
                        if V_si != model.spec_gate_vox.D:  # pad到max_V再裁回，复用共享门参数
                            pad = model.spec_gate_vox.D - V_si
                            v_pad = torch.nn.functional.pad(v, (0, pad))
                            v_flt = model.spec_gate_vox(v_pad)[:, :, :V_si]
                            voxel_for_ridge.append(v_flt)
                        else:
                            voxel_for_ridge.append(model.spec_gate_vox(v))
                else:
                    voxel_for_ridge = voxel_list
                voxel_ridge_list = [model.ridge(voxel_for_ridge[si], si) for si, s in enumerate(subj_list)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)
                base_total_batch = voxel_ridge.shape[0]

                if source_ridge_bank is not None:
                    source_bank_input = voxel_for_ridge[0] if args.enable_spectral_gate == 1 else voxel_list[0]
                    with torch.no_grad():
                        source_ridge_mean, source_ridge_stack = source_ridge_bank(source_bank_input)
                        source_ridge_selected = select_source_ridge_feature(
                            source_ridge_mean,
                            source_ridge_stack,
                            voxel_ridge,
                            args.source_ridge_subject_mode,
                        )
                    if args.source_ridge_fusion_mode in ("distill", "interp_distill"):
                        loss_source_ridge_distill = 1 - F.cosine_similarity(
                            voxel_ridge[:, 0].float(),
                            source_ridge_selected[:, 0].float(),
                            dim=-1,
                        ).mean()
                        if torch.isnan(loss_source_ridge_distill):
                            print("nan loss_source_ridge_distill")
                        else:
                            loss_source_ridge_distill_total += loss_source_ridge_distill.item()
                            loss += args.source_ridge_distill_weight * loss_source_ridge_distill
                    if args.source_ridge_fusion_mode in ("interp", "interp_distill"):
                        voxel_ridge = fuse_source_ridge_feature(
                            voxel_ridge,
                            source_ridge_selected.to(voxel_ridge.dtype),
                            args.source_ridge_alpha,
                        )

                if args.aug_consistency_flag != 0:
                    voxel_ridge = torch.cat([voxel_ridge, voxel_ridge], dim=0)  # 2Bx1x4096
                    image = torch.cat([image, image], dim=0)  # 2Bx1x4096

                # voxel_ridge_list = [model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]    # [16x1x1024, ]
                # voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                use_blur_now = args.use_image_aug != 0 and args.fovea_mode != 0 and (torch.rand(1).item() < args.fovea_p)
                blur_loss_weight = args.fovea_weight
                if args.use_image_aug == 1:
                    if args.color_jitter_flag == 1:
                        image_aug = img_augment(image)
                    if use_blur_now:
                        if args.color_jitter_flag == 0:
                            image_aug = image
                        if args.fovea_mode == 1:
                            image_aug = fovea_blur(image_aug)
                        elif args.fovea_mode == 2:
                            centers_b2 = centers_iters[train_i].detach().to(device)  # [B_total,2] (cx,cy)
                            radii_b2 = radii_iters[train_i].detach().to(device)      # [B_total] ratio

                            if args.aug_consistency_flag != 0:
                                centers_b2 = torch.cat([centers_b2, centers_b2], dim=0)  # 2Bx1x4096
                                radii_b2 = torch.cat([radii_b2, radii_b2], dim=0)

                            if args.blur_experiment_mode == "center_only":
                                centers_eff = img_center[None, :].expand_as(centers_b2).clone()
                                r_px = None
                            elif args.blur_experiment_mode == "fixed_semantic":
                                centers_eff = (1.0 - args.center_mix) * img_center[None, :] + args.center_mix * centers_b2
                                r_px = None
                            elif args.blur_experiment_mode == "radius_aware":
                                centers_eff = (1.0 - args.radius_aware_lambda) * img_center[None, :] + args.radius_aware_lambda * centers_b2
                                radii_eff = torch.clamp(radii_b2 * args.radius_scale, min=args.radius_min_ratio, max=args.radius_max_ratio)
                                radii_eff = _apply_radius_power(radii_eff, args.radius_min_ratio, args.radius_max_ratio, args.radius_power)
                                r_px = radii_eff * min(image.shape[-2], image.shape[-1])
                            elif args.blur_experiment_mode == "saliency_guided":
                                radii_eff = torch.clamp(radii_b2 * args.radius_scale, min=args.radius_min_ratio, max=args.radius_max_ratio)
                                radii_eff = _apply_radius_power(radii_eff, args.radius_min_ratio, args.radius_max_ratio, args.radius_power)
                                radii_norm = _normalize_radius_ratios(radii_eff, args.radius_min_ratio, args.radius_max_ratio)
                                center_dist = torch.norm(centers_b2 - img_center[None, :], dim=-1) / (0.5 * min(image.shape[-2], image.shape[-1]))
                                center_mix_eff = (
                                    args.center_mix
                                    + args.radius_center_slope * (0.5 - radii_norm)
                                    + args.distance_center_slope * center_dist
                                )
                                center_mix_eff = center_mix_eff.clamp(args.center_mix_min, args.center_mix_max)
                                centers_eff = (1.0 - center_mix_eff[:, None]) * img_center[None, :] + center_mix_eff[:, None] * centers_b2
                                r_px = radii_eff * min(image.shape[-2], image.shape[-1])
                            elif args.blur_experiment_mode in ("difficulty_aware", "difficulty_geo", "difficulty_semantic"):
                                if epoch >= args.difficulty_warmup_epoch:
                                    easy_scores_base = _lookup_difficulty_scores(
                                        image_ids_base,
                                        difficulty_bank,
                                        default_score=args.difficulty_default_score,
                                        device=device,
                                    )
                                else:
                                    easy_scores_base = torch.full(
                                        (image_ids_base.shape[0],),
                                        float(args.difficulty_default_score),
                                        device=device,
                                        dtype=torch.float32,
                                    )
                                easy_scores_base = easy_scores_base.clamp(0.0, 1.0).pow(args.difficulty_power)
                                hard_scores_base = 1.0 - easy_scores_base
                                if args.aug_consistency_flag != 0:
                                    easy_scores = torch.cat([easy_scores_base, easy_scores_base], dim=0)
                                    hard_scores = torch.cat([hard_scores_base, hard_scores_base], dim=0)
                                else:
                                    easy_scores = easy_scores_base
                                    hard_scores = hard_scores_base

                                if args.blur_experiment_mode == "difficulty_geo":
                                    centers_eff = img_center[None, :].expand_as(centers_b2).clone()
                                else:
                                    center_mix_eff = (
                                        args.difficulty_center_mix_min
                                        + easy_scores * (args.difficulty_center_mix_max - args.difficulty_center_mix_min)
                                    )
                                    centers_eff = (1.0 - center_mix_eff[:, None]) * img_center[None, :] + center_mix_eff[:, None] * centers_b2

                                radius_scale_factor = (
                                    1.0
                                    + hard_scores * args.difficulty_radius_bonus
                                    - easy_scores * args.difficulty_radius_shrink
                                ).clamp(min=0.05)
                                radii_eff = torch.clamp(
                                    radii_b2 * args.radius_scale * radius_scale_factor,
                                    min=args.radius_min_ratio,
                                    max=args.radius_max_ratio,
                                )
                                radii_eff = _apply_radius_power(
                                    radii_eff,
                                    args.radius_min_ratio,
                                    args.radius_max_ratio,
                                    args.radius_power,
                                )
                                r_px = radii_eff * min(image.shape[-2], image.shape[-1])
                                blur_loss_weight = args.fovea_weight * (
                                    args.difficulty_loss_weight_min
                                    + (1.0 - args.difficulty_loss_weight_min) * easy_scores_base.mean().item()
                                )
                            else:
                                raise ValueError(f"Unknown blur_experiment_mode: {args.blur_experiment_mode}")

                            centers_eff[..., 0].clamp_(0, image.shape[-1] - 1)
                            centers_eff[..., 1].clamp_(0, image.shape[-2] - 1)
                            image_aug = fovea_blur(image, center=centers_eff, r_px=r_px)
                else:
                    image_aug = image

                clip_target = clip_img_embedder(image)  # B, 256, 1664
                assert not torch.any(torch.isnan(clip_target))
                if use_blur_now:
                    clip_target_aug = clip_img_embedder(image_aug)

                if epoch < int(args.mixup_pct * args.num_epochs):
                    perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    select = torch.cat(select_list, dim=0)
                    perm_base, betas_base, select_base = perm, betas, select

                # vox_raw_list = [voxel_list[si] for si,s in enumerate(subj_list)]
                # vox_raw = torch.cat(vox_raw_list, dim=0)

                if args.finetune_mode in ("skip-lora", "lora+skip"):
                    skip_in = torch.cat(voxel_list, dim=0) if len(subj_list) == 1 else None
                    if args.aug_consistency_flag != 0:
                        skip_in = torch.cat([skip_in, skip_in], dim=0)
                    backbone, clip_voxels, blurry_image_enc_, skip_stats = model.backbone(voxel_ridge,
                                                                                          skip_input=skip_in,
                                                                                          return_skip_stats=True)
                else:
                    # print(voxel_ridge.shape)
                    backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

                if duala_sdp_aug is not None:
                    duala_labels = labels_for_image_ids(image_ids_base, duala_id2lab, device)
                    if args.aug_consistency_flag != 0:
                        duala_labels = torch.cat([duala_labels, duala_labels], dim=0)
                    clip_voxels = duala_sdp_aug(clip_voxels, duala_labels)

                if args.aug_consistency_flag != 0:
                    use_dual_view_align = args.align_loss_mode != "legacy"
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        if not use_dual_view_align:
                            perm = torch.cat([perm, perm], dim=0)
                            betas = torch.cat([betas, betas], dim=0)
                            select = torch.cat([select, select], dim=0)

                    if args.aug_consistency_flag == 1:
                        # on backbone
                        feat_batch_1 = backbone[:base_total_batch]
                        feat_batch_2 = backbone[base_total_batch:]
                    else:
                        # on backbone
                        feat_batch_1 = clip_voxels[:base_total_batch]
                        feat_batch_2 = clip_voxels[base_total_batch:]
                    loss_cons = (1 - F.cosine_similarity(feat_batch_1, feat_batch_2, dim=-1)).mean()

                    if torch.isnan(loss_cons):
                        print(f"nan loss_cons")
                    else:
                        loss_cons_total += loss_cons.item()
                        loss += loss_cons * args.aug_consistency_weight

                if args.clip_scale > 0:
                    clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                    use_dual_view_align = args.aug_consistency_flag != 0 and args.align_loss_mode != "legacy"
                    if use_dual_view_align:
                        clip_voxels_norm_view1 = clip_voxels_norm[:base_total_batch]
                        clip_voxels_norm_view2 = clip_voxels_norm[base_total_batch:]
                        clip_target_norm_base = clip_target_norm[:base_total_batch]
                    else:
                        clip_target_norm_base = clip_target_norm
                    clean_easy_scores = _compute_clean_easy_scores(
                        clip_voxels_norm=clip_voxels_norm[:base_total_batch] if use_dual_view_align else clip_voxels_norm,
                        clip_target_norm=clip_target_norm_base,
                        score_mode=args.difficulty_score_mode,
                        batch_temp=args.difficulty_batch_temp,
                        tau=args.difficulty_tau,
                        margin_mix=args.difficulty_margin_mix,
                    )
                    difficulty_easy_total += clean_easy_scores.mean().item()
                    if epoch >= args.difficulty_warmup_epoch:
                        _update_difficulty_bank(
                            difficulty_bank,
                            image_ids_base.detach().cpu(),
                            clean_easy_scores.detach().cpu(),
                            momentum=args.difficulty_bank_momentum,
                        )

                if args.use_prior:  # diffusion from BrainEncoder output to clip target
                    loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                    loss_prior_total += loss_prior.item()
                    loss_prior_scaled = loss_prior * args.prior_scale
                    loss += loss_prior_scaled

                    if args.prior_cosine_weight > 0:
                        prior_cos_loss = 1 - nn.functional.cosine_similarity(
                            prior_out.flatten(1),
                            clip_target.flatten(1),
                            dim=-1
                        ).mean()
                        loss_prior_cos_total += prior_cos_loss.item()
                        loss += args.prior_cosine_weight * prior_cos_loss

                    recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse += mse(prior_out, clip_target).item()

                if args.clip_scale>0:
                    use_dual_view_align = args.aug_consistency_flag != 0 and args.align_loss_mode != "legacy"
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        if use_dual_view_align:
                            loss_clip = dual_view_shared_positive_clip_loss(
                                clip_voxels_norm_view1,
                                clip_voxels_norm_view2,
                                clip_target_norm_base,
                                temp=.006,
                                mode="mixco",
                                perm=perm_base,
                                betas=betas_base,
                                select=select_base,
                            )
                        else:
                            loss_clip = utils.mixco_nce(
                                clip_voxels_norm,
                                clip_target_norm,
                                temp=.006,
                                perm=perm, betas=betas, select=select)
                        if use_blur_now:
                            clip_target_aug_norm = nn.functional.normalize(clip_target_aug.flatten(1), dim=-1)
                            if use_dual_view_align:
                                clip_target_aug_norm_base = clip_target_aug_norm[:base_total_batch]
                                loss_clip_aug = dual_view_shared_positive_clip_loss(
                                    clip_voxels_norm_view1,
                                    clip_voxels_norm_view2,
                                    clip_target_aug_norm_base,
                                    temp=.006,
                                    mode="mixco",
                                    perm=perm_base,
                                    betas=betas_base,
                                    select=select_base,
                                )
                            else:
                                loss_clip_aug = utils.mixco_nce(
                                    clip_voxels_norm,
                                    clip_target_aug_norm,
                                    temp=.006,
                                    perm=perm, betas=betas, select=select)
                            loss_clip += blur_loss_weight * loss_clip_aug
                    else:
                        epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                        if use_dual_view_align:
                            loss_clip = dual_view_shared_positive_clip_loss(
                                clip_voxels_norm_view1,
                                clip_voxels_norm_view2,
                                clip_target_norm_base,
                                temp=epoch_temp,
                                mode="soft",
                            )
                            if args.align_loss_mode == "dual_view_shared_dist":
                                align_rel_loss = dual_view_distribution_consistency_loss(
                                    clip_voxels_norm_view1,
                                    clip_voxels_norm_view2,
                                    clip_target_norm_base,
                                    temp=args.align_rel_temp,
                                    teacher_temp=args.align_teacher_temp,
                                )
                                if torch.isnan(align_rel_loss):
                                    print("nan align_rel_loss")
                                else:
                                    loss_align_rel_total += align_rel_loss.item()
                                    loss += args.align_rel_weight * align_rel_loss
                        else:
                            loss_clip = utils.soft_clip_loss(
                                    clip_voxels_norm,
                                    clip_target_norm,
                                    temp=epoch_temp)
                        if use_blur_now:
                            clip_target_aug_norm = nn.functional.normalize(clip_target_aug.flatten(1), dim=-1)
                            if use_dual_view_align:
                                clip_target_aug_norm_base = clip_target_aug_norm[:base_total_batch]
                                loss_clip_aug = dual_view_shared_positive_clip_loss(
                                    clip_voxels_norm_view1,
                                    clip_voxels_norm_view2,
                                    clip_target_aug_norm_base,
                                    temp=epoch_temp,
                                    mode="soft",
                                )
                            else:
                                loss_clip_aug = utils.soft_clip_loss(clip_voxels_norm, clip_target_aug_norm, temp=epoch_temp)
                            loss_clip += blur_loss_weight * loss_clip_aug
                        # if use_blur_now:
                        #     clip_target_aug_norm = nn.functional.normalize(clip_target_aug.flatten(1), dim=-1)
                        #     loss_clip = mp_infonce_weighted(
                        #         clip_voxels_norm,
                        #         [clip_target_norm, clip_target_aug_norm],
                        #         pos_weights=[1.0, args.fovea_weight],
                        #         temp=epoch_temp
                        #     )
                        # else:
                        #     loss_clip = utils.soft_clip_loss(
                        #         clip_voxels_norm,
                        #         clip_target_norm,
                        #         temp=epoch_temp)

                    loss_clip_total += loss_clip.item()
                    loss_clip *= args.clip_scale
                    loss += loss_clip

                if args.blurry_recon:
                    image_enc_pred, transformer_feats = blurry_image_enc_

                    image_enc = autoenc.encode(2*image-1).latent_dist.mode() * 0.18215
                    loss_blurry = l1(image_enc_pred, image_enc)
                    loss_blurry_total += loss_blurry.item()

                    if epoch < int(args.mixup_pct * args.num_epochs):
                        perm_blurry, betas_blurry, select_blurry = perm, betas, select
                        if args.aug_consistency_flag != 0 and args.align_loss_mode != "legacy":
                            perm_blurry = torch.cat([perm_base, perm_base + base_total_batch], dim=0)
                            betas_blurry = torch.cat([betas_base, betas_base], dim=0)
                            select_blurry = torch.cat([select_base, select_base], dim=0)
                        image_enc_shuf = image_enc[perm_blurry]
                        betas_shape = [-1] + [1]*(len(image_enc.shape)-1)
                        image_enc[select_blurry] = image_enc[select_blurry] * betas_blurry[select_blurry].reshape(*betas_shape) + \
                            image_enc_shuf[select_blurry] * (1 - betas_blurry[select_blurry]).reshape(*betas_shape)

                    image_norm = (image - mean)/std
                    image_aug = (blur_augs(image) - mean)/std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)

                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.2)
                    loss_blurry_cont_total += cont_loss.item()

                    loss += (loss_blurry + 0.1*cont_loss) * args.blur_scale #/.18215

                if args.finetune_mode in ("skip-lora", "lora+skip"):
                    if isinstance(skip_stats, list) and len(skip_stats) > 0:
                        corr_vals = []
                        # If final mapping skip adapter is present, exclude it from Lskip; otherwise include all block pairs
                        use_pairs = skip_stats[:-1] if args.skip_include_final and (len(skip_stats) > 0) else skip_stats
                        for lin, sk in use_pairs:
                            # ensure shapes (B, D)
                            lin_flat = lin.view(lin.size(0), -1)
                            sk_flat = sk.view(sk.size(0), -1)
                            corr = abs_pearson_corr(lin_flat, sk_flat)
                            corr_vals.append(corr.mean())
                        if len(corr_vals):
                            l_skip = torch.stack(corr_vals).mean()
                            if torch.isnan(l_skip):
                                print(f"nan l_skip")
                            else:
                                loss_skip_total += l_skip.item()
                                loss = loss + args.skip_loss_weight * l_skip

                if args.clip_scale > 0:
                    # forward and backward top 1 accuracy
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                if args.blurry_recon:
                    with torch.no_grad():
                        # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                        random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample/ 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        blurry_pixcorr += pixcorr.item()

                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()

        model.eval()
        if local_rank==0:
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type):
                for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
                    # all test samples should be loaded per batch such that test_i should never exceed 0
                    assert len(behav) == num_test

                    ## Average same-image repeats ##
                    if test_image is None:
                        voxel = voxels[f'subj0{args.subj}'][behav[:,0,5].cpu().long()].unsqueeze(1)

                        image = behav[:,0,0].cpu().long()

                        unique_image, sort_indices = torch.unique(image, return_inverse=True)
                        for im in unique_image:
                            locs = torch.where(im == image)[0]
                            if len(locs)==1:
                                locs = locs.repeat(3)
                            elif len(locs)==2:
                                locs = locs.repeat(2)[:3]
                            assert len(locs)==3
                            if test_image is None:
                                test_image = torch.Tensor(images[im][None])
                                test_voxel = voxel[locs][None]
                            else:
                                test_image = torch.vstack((test_image, torch.Tensor(images[im][None])))
                                test_voxel = torch.vstack((test_voxel, voxel[locs][None]))

                    # loss=0.
                    loss = torch.tensor(0.0, device=device)

                    test_indices = torch.arange(len(test_voxel))[:300]
                    voxel = test_voxel[test_indices].to(device)
                    image = test_image[test_indices].to(device)
                    assert len(image) == 300

                    clip_target = clip_img_embedder(image.float())
                    for rep in range(3):
                        if args.enable_spectral_gate == 1:
                            voxel_rep = model.spec_gate_vox(voxel[:, rep])
                            voxel_ridge = model.ridge(voxel_rep, 0)
                        else:
                            voxel_rep = voxel[:, rep]
                            voxel_ridge = model.ridge(voxel_rep, 0)  # 0th index of subj_list

                        if source_ridge_bank is not None and args.source_ridge_fusion_mode in ("interp", "interp_distill"):
                            with torch.no_grad():
                                source_ridge_mean, _ = source_ridge_bank(voxel_rep)
                            voxel_ridge = fuse_source_ridge_feature(
                                voxel_ridge,
                                source_ridge_mean.to(voxel_ridge.dtype),
                                args.source_ridge_alpha,
                            )

                        if args.finetune_mode in ("skip-lora", "lora+skip"):
                            backbone0, clip_voxels0, blurry_image_enc_, _ = model.backbone(voxel_ridge,
                                                                                           skip_input=voxel[:, rep],
                                                                                           return_skip_stats=True)
                        else:
                            backbone0, clip_voxels0, blurry_image_enc_ = model.backbone(voxel_ridge)

                        if rep==0:
                            clip_voxels = clip_voxels0
                            backbone = backbone0
                        else:
                            clip_voxels += clip_voxels0
                            backbone += backbone0
                    clip_voxels /= 3
                    backbone /= 3

                    if args.clip_scale > 0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)

                    if args.use_prior:
                        loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        loss_prior *= args.prior_scale
                        loss += loss_prior

                    if args.clip_scale>0:
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006)

                        test_loss_clip_total += loss_clip.item()
                        loss_clip = loss_clip * args.clip_scale
                        loss += loss_clip

                    if args.blurry_recon:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps]/0.18215).sample / 2 + 0.5).clamp(0,1)
                        pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                        test_blurry_pixcorr += pixcorr.item()

                    if args.clip_scale>0:
                        # forward and backward top 1 accuracy
                        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                    utils.check_loss(loss)
                    test_losses.append(loss.item())

                assert (test_i+1) == 1
                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "test/loss": np.mean(test_losses[-(test_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "test/num_steps": len(test_losses),
                    "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                    "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                    "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                    "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                    "train/loss_clip_total": loss_clip_total / (train_i + 1),
                    "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                    "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                    "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                    "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                    "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                    "train/recon_cossim": recon_cossim / (train_i + 1),
                    "test/recon_cossim": test_recon_cossim / (test_i + 1),
                    "train/recon_mse": recon_mse / (train_i + 1),
                    "test/recon_mse": test_recon_mse / (test_i + 1),
                    "train/loss_prior": loss_prior_total / (train_i + 1),
                    "test/loss_prior": test_loss_prior_total / (test_i + 1),
                    "train/difficulty_easy_mean": difficulty_easy_total / (train_i + 1),
                    }

                if args.prior_cosine_weight > 0:
                    logs["train/loss_prior_cos"] = loss_prior_cos_total / (train_i + 1)

                if args.finetune_mode in ("skip-lora", "lora+skip"):
                    logs["train/loss_skip"] = loss_skip_total / (train_i + 1)

                if args.aug_consistency_flag != 0:
                    logs["train/loss_cons"] = loss_cons_total / (train_i + 1)
                    if args.align_loss_mode == "dual_view_shared_dist":
                        logs["train/loss_align_rel"] = loss_align_rel_total / (train_i + 1)

                if source_ridge_bank is not None and args.source_ridge_fusion_mode in ("distill", "interp_distill"):
                    logs["train/loss_source_ridge_distill"] = loss_source_ridge_distill_total / (train_i + 1)

                # if finished training, save jpg recons if they exist
                if (epoch == args.num_epochs-1) or (epoch % args.ckpt_interval == 0):
                    if args.blurry_recon:
                        image_enc = autoenc.encode(2*image[:4]-1).latent_dist.mode() * 0.18215
                        # transform blurry recon latents to images and plot it
                        fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                        jj=-1
                        for j in [0,1,2,3]:
                            jj+=1
                            axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                            axes[jj].axis('off')
                            jj+=1
                            axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                            axes[jj].axis('off')

                        if args.wandb_log:
                            logs[f"test/blur_recons"] = wandb.Image(fig, caption=f"epoch{epoch:03d}")
                            plt.close()
                        else:
                            try:
                                plt.savefig(f'{outdir}/blur_recons_epoch{epoch:03d}.png')
                                plt.show()
                            except OSError as exc:
                                print(f"Warning: failed to save blur recon preview at epoch {epoch}: {exc}", flush=True)
                            finally:
                                plt.close()

                progress_bar.set_postfix(**logs)

                if args.wandb_log: wandb.log(logs)

                save_logs(outdir, logs, epoch)
        # Save model checkpoint and reconstruct
        if (args.ckpt_saving) and (epoch % args.ckpt_interval == 0):
            save_ckpt(outdir, f'last', epoch)

        # wait for other GPUs to catch up if needed
        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

    print("\n===Finished!===\n")
    if args.ckpt_saving:
        save_ckpt(outdir, f'last', epoch)


plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(outdir + '/training_loss.png')  # 保存为文件
plt.close()  # 关闭当前图形

plt.plot(test_losses)
plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.title('Test Loss')
plt.savefig(outdir + '/test_loss.png')  # 保存为文件
plt.close()  # 关闭当前图形
