import torch
import torch.nn as nn
import math
import torch.nn.functional as F

try:
    from MindTuner_modules import LoRALinear as _LoRALinear
    _LINEAR_TYPES = (nn.Linear, _LoRALinear)
except Exception:
    _LINEAR_TYPES = (nn.Linear,)


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapter: W is frozen, deltaW = B @ A is trainable.
    Initialization: B ~ N(0, 0.01), A = 0 so initial delta output is zero.
    Output: y = x @ W^T + scale * x @ A^T @ B^T + bias
    """

    def __init__(self, in_features, out_features, bias=True, rank=8, alpha=8, base_weight=None, base_bias=None,
                 dropout_p=0., tie_dropout_scale=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(1, rank)

        self.dropout_p = float(dropout_p)
        self.tie_dropout_scale = tie_dropout_scale

        # Base weight (frozen)
        if base_weight is None:
            base = nn.Linear(in_features, out_features, bias=bias)
            self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False) if bias else None
        else:
            self.weight = nn.Parameter(base_weight.data.clone(), requires_grad=False)
            self.bias = nn.Parameter(base_bias.data.clone(),
                                     requires_grad=False) if bias and base_bias is not None else None

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        # init: A = 0, B ~ N(0, 0.01)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.01)
        # nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # nn.init.zeros_(self.A)

        self.is_lora = True

    @torch.no_grad()
    def _sample_masks(self, device, dtype):
        """Bernoulli(1-p) masks with inverted-dropout scaling."""
        if self.dropout_p <= 0.0:
            mA = torch.ones(self.in_features, device=device, dtype=dtype)
            mB = torch.ones(self.out_features, device=device, dtype=dtype)
            return mA, mB

        keep = 1.0 - self.dropout_p
        scaleA = 1.0 / keep if self.tie_dropout_scale == 1 else 1.0
        scaleB = 1.0 / keep if self.tie_dropout_scale == 1 else 1.0

        mA = torch.bernoulli(torch.full((self.in_features,), keep, device=device, dtype=dtype))
        mB = torch.bernoulli(torch.full((self.out_features,), keep, device=device, dtype=dtype))

        mA.mul_(scaleA)
        mB.mul_(scaleB)
        return mA, mB


    def forward(self, x):
        base = x.matmul(self.weight.t())
        if self.bias is not None:
            base = base + self.bias

        if self.training and self.dropout_p > 0.0:
            mA, mB = self._sample_masks(device=self.lora_A.device, dtype=self.lora_A.dtype)
            A_hat = self.lora_A * mA
            B_hat = self.lora_B * mB.unsqueeze(1)
        else:
            A_hat, B_hat = self.lora_A, self.lora_B

        delta = x.matmul(A_hat.t()).matmul(B_hat.t()) * self.scale
        return base + delta

class SkipLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=8, activation='gelu',
                 dropout_p=0., tie_dropout_scale=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / max(1, rank)

        self.dropout_p = dropout_p
        self.tie_dropout_scale = tie_dropout_scale

        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.B, mean=0.0, std=0.01)
        # nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # nn.init.zeros_(self.A)


        # nn.init.normal_(self.B, mean=0.0, std=0.01)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.GELU()

        self.is_lora = True  # tag for filtering trainable params

    @torch.no_grad()
    def _sample_masks(self, device, dtype):
        """Bernoulli(1-p) masks with inverted-dropout scaling."""
        if self.dropout_p <= 0.0:
            mA = torch.ones(self.in_features, device=device, dtype=dtype)
            mB = torch.ones(self.out_features, device=device, dtype=dtype)
            return mA, mB

        keep = 1.0 - self.dropout_p
        scaleA = 1.0 / keep if self.tie_dropout_scale == 1 else 1.0
        scaleB = 1.0 / keep if self.tie_dropout_scale == 1 else 1.0

        mA = torch.bernoulli(torch.full((self.in_features,), keep, device=device, dtype=dtype))
        mB = torch.bernoulli(torch.full((self.out_features,), keep, device=device, dtype=dtype))

        mA.mul_(scaleA)
        mB.mul_(scaleB)
        return mA, mB

    def forward(self, v_raw):
        # v_raw: (B, in_features) or (B, 1, in_features)
        if self.training and self.dropout_p > 0.0:
            mA, mB = self._sample_masks(device=self.A.device, dtype=self.A.dtype)
            A_hat = self.A * mA
            B_hat = self.B * mB.unsqueeze(1)
        else:
            A_hat, B_hat = self.A, self.B

        if v_raw.dim() == 3:
            v = v_raw[:, 0, :]
        else:
            v = v_raw

        out = v.matmul(A_hat.t()).matmul(B_hat.t()) * self.scale
        return self.act(out)

# class SkipLoRAParam(nn.Module):
#     """
#     y_skip = (alpha/rank) * activation( x @ A^T @ B^T )
#     ——激活在 BA 之后，符合 activate(BA)。
#     """
#     def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 1.0, act: str = "gelu",
#                  dropout_p=0., tie_dropout_scale=False):
#         super().__init__()
#         self.rank = rank
#         self.scale = alpha / rank
#         self.A = nn.Parameter(torch.zeros(rank, in_dim))      # A_new  (r x in)
#         self.B = nn.Parameter(torch.zeros(out_dim, rank))     # B_new  (out x r)
#         self.act = nn.GELU() if act == "gelu" else nn.ReLU(inplace=True)
#
#         self.dropout_p = float(dropout_p)
#         self.tie_dropout_scale = tie_dropout_scale
#
#         nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
#         nn.init.zeros_(self.A)
#
#     @torch.no_grad()
#     def _sample_masks(self, device, dtype):
#         """Bernoulli(1-p) masks with inverted-dropout scaling."""
#         if self.dropout_p <= 0.0:
#             mA = torch.ones(self.in_features, device=device, dtype=dtype)
#             mB = torch.ones(self.out_features, device=device, dtype=dtype)
#             return mA, mB
#
#         keep = 1.0 - self.dropout_p
#         scaleA = 1.0 / keep if self.tie_dropout_scale == 1 else 1.0
#         scaleB = 1.0 / keep if self.tie_dropout_scale == 1 else 1.0
#
#         mA = torch.bernoulli(torch.full((self.in_features,), keep, device=device, dtype=dtype))
#         mB = torch.bernoulli(torch.full((self.out_features,), keep, device=device, dtype=dtype))
#
#         mA.mul_(scaleA)
#         mB.mul_(scaleB)
#         return mA, mB
#
#     def forward(self, x):                  # x: (B, in_dim)
#         if self.training and self.dropout_p > 0.0:
#             mA, mB = self._sample_masks(device=self.A.device, dtype=self.A.dtype)
#             A_hat = self.A * mA
#             B_hat = self.B * mB.unsqueeze(1)
#         else:
#             A_hat, B_hat = self.A, self.B
#
#         y_lin = (x @ A_hat.t()) @ B_hat.t()    # (B, out_dim)
#         return self.scale * self.act(y_lin)
#

# def replace_linear_with_lora(module: nn.Module, rank: int = 8, alpha: float = 1.0, filter_fn=None, replaced_names=None):
#     if replaced_names is None:
#         replaced_names = []
#     for name, child in list(module.named_children()):
#         if isinstance(child, nn.Linear):
#             if (filter_fn is None) or filter_fn(name, child):
#                 setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
#                 replaced_names.append(name)
#         else:
#             replace_linear_with_lora(child, rank=rank, alpha=alpha, filter_fn=filter_fn, replaced_names=replaced_names)
#     return replaced_names

def _replace_linear_with_lora(module: nn.Module, rank: int, alpha: int, name_filter=None, replaced_names=None,
                              dropout_p=0., tie_dropout_scale=0,
                              ):
    if replaced_names is None:
        replaced_names = []
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if (name_filter is None) or name_filter(child_name, child):
                lora = LoRALinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    rank=rank,
                    alpha=alpha,
                    base_weight=child.weight,
                    base_bias=child.bias,
                    dropout_p=dropout_p,
                    tie_dropout_scale=tie_dropout_scale,
                )
                setattr(module, child_name, lora)
                replaced_names.append(child_name)
        else:
            _replace_linear_with_lora(child, rank, alpha, name_filter, replaced_names, dropout_p, tie_dropout_scale)
    return replaced_names


def enable_lora_params(module: nn.Module):
    if hasattr(module, "lora_parameters") and callable(getattr(module, "lora_parameters")):
        for p in module.lora_parameters():
            p.requires_grad = True
    for name, p in module.named_parameters(recurse=True):
        low = name.lower()
        if any(tok in low for tok in ["lora_", ".lora", "loraa", "lorab", "lora_a", "lora_b"]) \
                or low.endswith(".a") or low.endswith(".b") \
                or any(tok in low for tok in ["loraalpha", "alpha", "lorascale", "scale"]):
            p.requires_grad = True  # LoRA 增量/缩放

# ==== MindTuner: Adaptive Projector (Pivot) ====
class AdaptiveProjector(nn.Module):
    def __init__(self, in_dim=1664, out_dim=1280):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(in_dim, out_dim))

    def forward(self, img_tokens):  # img_tokens: (B, 256, 1664)
        x = img_tokens.mean(dim=1)  # (B, 1664)
        return x @ self.proj  # (B, 1280)


def clip_like_contrastive(p, t, tau=0.006):
    """
    标准 InfoNCE / CLIP 对比损失（双向可选，这里用单向即可，论文式(11)为 image->text）。
    p: (B, D), t: (B, D)
    """
    p = nn.functional.normalize(p, dim=-1)
    t = nn.functional.normalize(t, dim=-1)
    logits = p @ t.t() / tau
    labels = torch.arange(p.size(0), device=p.device)
    return nn.functional.cross_entropy(logits, labels)





# class MindTunerSkipConnector(nn.Module):
#     def __init__(self, in_dim_vox: int, hidden_dim: int, token_dim: int, token_len: int,
#                  rank: int = 8, alpha: float = 1.0, act: str = "gelu"):
#         super().__init__()
#         self.in_dim = in_dim_vox
#         self.hdim = hidden_dim       # 4096
#         self.tdim = token_dim        # 1664
#         self.tlen = token_len        # 256
#         self.rank, self.alpha, self.act = rank, alpha, act
#         self.adapters = nn.ModuleList()
#         self.hooks = []
#         self.vox_raw = None
#         self.pairs = []
#
#         self.align_done = False
#         self.res_done = 0
#         self.tokens_done = False
#
#     def reset_adapters(self):
#         # 丢弃已注册的 ModuleList，彻底清空已创建的 adapters
#         self.adapters = nn.ModuleList()
#         self.pairs.clear()
#         self.align_done = False
#         self.res_done = 0
#         self.tokens_done = False
#
#     @staticmethod
#     def _pearson_abs_mean(x, y):
#         x = x.flatten(1); y = y.flatten(1)
#         x = x - x.mean(0, keepdim=True); y = y - y.mean(0, keepdim=True)
#         xr = x / (x.std(0, keepdim=True) + 1e-6)
#         yr = y / (y.std(0, keepdim=True) + 1e-6)
#         return (xr*yr).mean(0).abs().mean()
#
#     def set_vox_raw(self, vox_raw: torch.Tensor):
#         self.vox_raw = vox_raw
#
#     def _want_inject(self, out: torch.Tensor):
#         if out.dim() == 3 and out.shape[1] == self.tlen and out.shape[-1] == self.tdim:
#             return (not self.tokens_done), "tokens", out.shape[-1], True
#
#         candidate_hdims = {self.hdim, 1024, 2048, 4096}
#         if out.shape[-1] in candidate_hdims:
#             if not self.align_done:
#                 return True, "align", out.shape[-1], (out.dim() == 3)
#             if self.res_done < 4:
#                 return True, "res", out.shape[-1], (out.dim() == 3)
#
#         return False, "", 0, False
#
#     def _make_hook(self, slot: dict):
#         def hook(_mod, _inp, out):
#             if self.vox_raw is None:
#                 return out
#
#             inject, kind, odim, need_bc = self._want_inject(out)
#             if not inject:
#                 return out
#
#             if "adapter" not in slot or slot["adapter"] is None:
#                 ad = SkipLoRAParam(self.in_dim, odim, self.rank, self.alpha, self.act).to(out.device)
#                 slot["adapter"] = ad
#                 self.adapters.append(ad)
#             base = out
#             skip_vec = slot["adapter"](self.vox_raw)                   # (B, odim)
#             if need_bc and out.dim() == 3:
#                 B, L, D = out.shape
#                 skip = skip_vec.unsqueeze(1).expand(B, L, D)
#             else:
#                 skip = skip_vec
#             self.pairs.append((base.detach(), skip.detach()))          # 相加前记录
#             if kind == "align":
#                 self.align_done = True
#             elif kind == "res":
#                 self.res_done += 1
#             elif kind == "tokens":
#                 self.tokens_done = True
#             return base + skip
#         return hook
#
#     def attach_modules(self, modules: list[nn.Module]):
#         LinearTypes = (nn.Linear,)
#         try:
#             # from MindTuner_modules import LoRALinear
#             LinearTypes = (nn.Linear, LoRALinear)
#         except Exception:
#             pass
#         for m in modules:
#             for name, mod in m.named_modules():
#                 if isinstance(mod, LinearTypes):
#                     h = mod.register_forward_hook(self._make_hook({}))
#                     self.hooks.append(h)
#
#     def compute_lskip_and_clear(self, device=None):
#         if not self.pairs:
#             return torch.tensor(0.0, device=device) if device else 0.0
#         vals = [self._pearson_abs_mean(b, s) for (b, s) in self.pairs]
#         self.pairs.clear()
#         out = torch.stack(vals).mean()
#         return out.to(device) if device is not None else out

def batch_pearson_abs(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x, y: (B, D)
    """
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (x * y).sum(dim=1)
    den = torch.sqrt((x.pow(2).sum(dim=1) + eps) * (y.pow(2).sum(dim=1) + eps))
    r = num / den
    return r.abs().mean()

def compute_skip_loss(skip_stats: list, device) -> torch.Tensor:
    """
    skip_stats: list[(M_h-linear(B,D), M_h-skip(B,D))]
    """
    if (skip_stats is None) or (len(skip_stats) == 0):
        return torch.zeros((), device=device, dtype=torch.float32)
    vals = []
    for lin, sk in skip_stats:
        vals.append(batch_pearson_abs(lin, sk))
    return torch.stack(vals).mean()


import numpy as np
from torch.nn import functional as F

class DSU(nn.Module):

    def __init__(self, p=0.5, eps=1e-6):
        super(DSU, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()  # 1xKxCx1
        # t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x
        mean = x.mean(dim=-1, keepdim=True)  # Bx1xC
        std = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean) / std
        x = x * gamma + beta

        return x

import random
class MixStyle(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)
        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed * sig_mix + mu_mix

class FourierMixStyle(nn.Module):
    """
    Fourier域的 MixStyle:
      - amp_mode: 'global' | 'band'
      - phase_mode: 'none' | 'shift' | 'slerp'
      - p: 以概率 p 应用增强（训练时有效，eval 时自动关闭）
      - alpha: Beta(alpha, alpha) 的混合系数
      - n_bands: band 模式下的频带数量
    输入: x [B, L]  (L 可以是时间长度或特征长度；最后一维做FFT)
    """
    def __init__(self, p=0.5, alpha=0.1, amp_mode='global', n_bands=3):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.amp_mode = amp_mode
        self.n_bands = n_bands

    def _beta(self, bsz, device):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((bsz, 1, 1)).to(device)
        return lam

    def _band_slices(self, n_freq):
        edges = torch.linspace(0, n_freq, self.n_bands + 1, dtype=torch.long)
        return [(int(edges[i].item()), int(edges[i+1].item())) for i in range(self.n_bands)]

    def forward(self, x: torch.Tensor):
        if (not self.training) or (torch.rand(()) > self.p):
            return x
        B, _, L = x.shape
        device = x.device

        F = torch.fft.rfft(x, dim=-1)            # [B, Lf]
        A = torch.abs(F)                         # 非负
        P = torch.angle(F)                       # [-pi, pi]
        Lf = A.shape[-1]

        idx = torch.randperm(B, device=device)
        A2, P2 = A[idx], P[idx]
        lam = self._beta(B, device)              # [B,1,1] 广播到频率维

        if self.amp_mode in ('global', 'beta_mix'):
            mu1  = A.mean(dim=-1, keepdim=True)
            std1 = A.std (dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            z1   = (A - mu1) / std1

            mu2  = A2.mean(dim=-1, keepdim=True)
            std2 = A2.std (dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

            mu_mix  = lam * mu1 + (1 - lam) * mu2
            std_mix = lam * std1 + (1 - lam) * std2

            A_new = z1 * std_mix + mu_mix

        elif self.amp_mode == 'band':
            A_new = torch.empty_like(A)
            for lo, hi in self._band_slices(Lf):
                Ai, Aj = A[..., lo:hi], A2[..., lo:hi]

                mu1  = Ai.mean(dim=-1, keepdim=True)
                std1 = Ai.std (dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                z1   = (Ai - mu1) / std1

                mu2  = Aj.mean(dim=-1, keepdim=True)
                std2 = Aj.std (dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

                mu_mix  = lam * mu1 + (1 - lam) * mu2
                std_mix = lam * std1 + (1 - lam) * std2

                A_new[..., lo:hi] = z1 * std_mix + mu_mix
        elif self.amp_mode == 'swap':
            A_new = A2
        else:
            A_new = A

        F_new = A_new * torch.exp(1j * P)
        x_new = torch.fft.irfft(F_new, n=L, dim=-1)
        return x_new


class FourierModelStyle(nn.Module):
    def __init__(self, p=0.5, mode=0, noise_alpha=1.0, eps=1e-6):
        super().__init__()
        # 0: statistic; 1: element
        self.p = p
        self.mode = mode
        self.eps = eps
        self.noise_alpha = noise_alpha

    def forward(self, x: torch.Tensor):
        if (not self.training) or (torch.rand(()) > self.p):
            return x

        B, _, L = x.shape
        device = x.device

        F = torch.fft.rfft(x, dim=-1)            # [B, Lf]
        A = torch.abs(F)                         # B, L, N
        P = torch.angle(F)                       # [-pi, pi]

        A = torch.nan_to_num(A, nan=0.0, posinf=1e3, neginf=0.0)
        A = A.clamp(max=1e3)

        if self.mode == 0:
            mu_i = A.mean(dim=-1, keepdim=True)
            std_i = A.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)

            mu_batch_std = mu_i.std(dim=0, unbiased=False, keepdim=True)    # 1, L, N
            std_batch_std = std_i.std(dim=0, unbiased=False, keepdim=True)

            noise_mu = torch.randn_like(mu_i) * mu_batch_std    # (0, mu_std)
            noise_std = torch.randn_like(std_i) * std_batch_std  # (0, std_std)

            mu_new = mu_i + self.noise_alpha * noise_mu
            std_new = (std_i + self.noise_alpha * noise_std).clamp_min(self.eps)

            z = (A - mu_i) / std_i
            A_new = z * std_new + mu_new

        elif self.mode == 1:
            A_std = A.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
            noise = torch.randn_like(A) * A_std
            A_new = A + self.noise_alpha * noise
        elif self.mode == 2:
            a_min = A.amin(dim=0, keepdim=True)
            a_max = A.amax(dim=0, keepdim=True)
            rand_u = torch.rand_like(A)
            A_new = a_min + rand_u * (a_max - a_min).clamp_min(self.eps)
        else:
            raise ValueError(f"Unsupported FourierModelStyle mode: {self.mode}")

        norm_factor = (A.norm(dim=(-2, -1), keepdim=True) /
                       (A_new.norm(dim=(-2, -1), keepdim=True) + self.eps))
        A_new = A_new * norm_factor

        F_new = A_new * torch.exp(1j * P)
        x_new = torch.fft.irfft(F_new, n=L, dim=-1)
        x_new = torch.nan_to_num(x_new, nan=0.0, posinf=1e3, neginf=-1e3)
        return x_new
