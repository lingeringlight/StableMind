import torch, os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFilter
import logging
import open_clip
import pickle

try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from scipy.optimize import fsolve
import torchvision.transforms.functional as TF
import math
import kornia


def _rgb_to_bgr(img_np):
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        return img_np[..., ::-1]
    return img_np


def _bgr_to_rgb(img_np):
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        return img_np[..., ::-1]
    return img_np


def _gaussian_blur_np(img_np, blur_kernel_size):
    if cv2 is not None:
        return cv2.GaussianBlur(img_np, (blur_kernel_size, blur_kernel_size), 0)
    radius = max((int(blur_kernel_size) - 1) / 6.0, 0.0)
    return np.array(Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(radius=radius)))


def _blend_with_mask(img1, img2, mask):
    if cv2 is not None:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.convertScaleAbs(img1 * (1 - alpha) + img2 * alpha)
    alpha = np.asarray(mask, dtype=np.float32)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    blended = np.clip(
        np.asarray(img1, dtype=np.float32) * (1.0 - alpha) +
        np.asarray(img2, dtype=np.float32) * alpha,
        0,
        255,
    )
    return blended.astype(np.uint8)


class DirectT:
    def __init__(self):
        pass

    def __call__(self, x, U=None):
        return x

class UniformBlur:
    def __init__(self, blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        img_np = _rgb_to_bgr(img_np)
        img_blur = _gaussian_blur_np(img_np, self.blur_kernel_size)
        img_blur = _bgr_to_rgb(img_blur)
        return Image.fromarray(img_blur)


class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h, w), np.float32)

        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1 - c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]

        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]

        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1, distance / max_distance)
                y0 = fun_degrade(x0, **kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        return _blend_with_mask(img1, img2, mask)

    def __call__(self, img, blur_kernel_size=None):
        if blur_kernel_size == None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        img = _rgb_to_bgr(img)
        blured = _gaussian_blur_np(img, blur_kernel_size)
        blended = self.alphaBlend(img, blured, 1 - self.mask)
        blended = _bgr_to_rgb(blended)
        return Image.fromarray(blended)

    def linear(self, x, **kwargs):
        return 1 - x

    def exp(self, x, **kwargs):
        system_g = kwargs.get('system_g', 4)
        return np.exp(-system_g * x)

    def quadratic(self, x, **kwargs):
        return 1 - x ** 2

    def log(self, x, **kwargs):
        b = 1 / (np.e - 1)
        a = np.log(b) + 1
        return a - np.log(x + b)

    def brachistochrone(self, x, **kwargs):

        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return y0


class FoveaBlurTorch_adaptive(nn.Module):
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        super().__init__()
        assert blur_kernel_size % 2 == 1 and blur_kernel_size >= 3, "blur_kernel_size 必须为 >=3 的奇数"
        self.blur_kernel_size = int(blur_kernel_size)
        self.h, self.w = int(h), int(w)
        self.curve_type = curve_type
        self.kwargs = kwargs

        self.mask = np.zeros((h, w), np.float32)

        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1 - c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]

        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]

        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        if fun_degrade is None:
            raise ValueError(f"Unknown curve_type: {curve_type}")

        self._center_resolution = center_resolution
        self._edge_resolution = edge_resolution
        self._fun_degrade = fun_degrade

        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1, distance / max_distance)
                y0 = fun_degrade(x0, **kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

        self.register_buffer("mask_t", torch.from_numpy(self.mask).float())  # [H,W]

    def alphaBlend(self, img1, img2, mask):
        return _blend_with_mask(img1, img2, mask)

    # def __call__(self, img, blur_kernel_size=None):
    #     return self._process_single(img, blur_kernel_size)

    def forward(self, x, center=None):
        """
        x: torch.Tensor
           - (B, C, H, W) 或 (C, H, W)，通常范围[0,1]或[0,255]
        返回: 同形状 torch.Tensor，dtype/设备与输入一致
        """
        orig_device = x.device
        orig_dtype = x.dtype

        if x.dim() == 3:
            x = x.unsqueeze(0)  # -> (1, C, H, W)

        B, C, H, W = x.shape
        if (H != self.h) or (W != self.w):
            raise ValueError(f"输入尺寸 {H}x{W} 与 FoveaBlur 初始化的 {self.h}x{self.w} 不一致。")

        per_sample_centers = None
        if center is None:
            # 使用默认中心（走旧逻辑，不重算 mask）
            pass
        else:
            if isinstance(center, (tuple, list)) and len(center) == 2:
                cx, cy = float(center[0]), float(center[1])
                per_sample_centers = np.tile(np.array([[cx, cy]], dtype=np.float32), (B, 1))
            elif isinstance(center, torch.Tensor):
                assert center.shape == (B, 2), "center Tensor 需为 (B,2)"
                per_sample_centers = center.detach().cpu().numpy().astype(np.float32)
            elif isinstance(center, np.ndarray):
                assert center.shape == (B, 2), "center ndarray 需为 (B,2)"
                per_sample_centers = center.astype(np.float32)
            else:
                raise TypeError("center 类型不支持，应为 None / (cx,cy) / Tensor(B,2) / ndarray(B,2)")

        x_cpu = x.detach().to("cpu")
        out_list = []
        for b in range(B):
            img_bchw = x_cpu[b]
            img_pil = TF.to_pil_image(img_bchw.clamp(0, 1) if img_bchw.max() <= 1.0 else (img_bchw / 255.0).clamp(0,1))

            mask_override = None
            if per_sample_centers is not None:
                cx_b = float(per_sample_centers[b, 0])
                cy_b = float(per_sample_centers[b, 1])
                mask_override = self._build_mask_at_center(cx_b, cy_b)

            out_pil = self._process_single(img_pil, blur_kernel_size=None, mask_override=mask_override)
            # out_pil = self._process_single(img_pil, blur_kernel_size=None)
            out_tensor = TF.to_tensor(out_pil)  # float32, [0,1]
            if x_cpu.max() > 1.0:
                out_tensor = (out_tensor * 255.0).clamp(0, 255)
            out_list.append(out_tensor)

        out = torch.stack(out_list, dim=0).to(orig_device).to(orig_dtype)
        return out

    def _build_mask_at_center(self, cx, cy):
        h, w = self.h, self.w
        mask = np.zeros((h, w), np.float32)
        max_distance = np.sqrt((h - cy - 1) ** 2 + (w - cx - 1) ** 2)
        if max_distance < 1e-6:
            max_distance = 1.0  # 避免除零

        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
                x0 = min(1.0, distance / max_distance)
                y0 = self._fun_degrade(x0, **self.kwargs)
                mask[i, j] = self._edge_resolution + (self._center_resolution - self._edge_resolution) * y0
        return mask

    def _process_single(self, img, blur_kernel_size=None, mask_override=None):
        if blur_kernel_size is None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        img = _rgb_to_bgr(img)
        blured = _gaussian_blur_np(img, blur_kernel_size)
        mask_used = self.mask if mask_override is None else mask_override
        blended = self.alphaBlend(img, blured, 1 - mask_used)
        blended = _bgr_to_rgb(blended)
        return Image.fromarray(blended)

    def linear(self, x, **kwargs):
        return 1 - x

    def exp(self, x, **kwargs):
        system_g = kwargs.get('system_g', 4)
        return np.exp(-system_g * x)

    def quadratic(self, x, **kwargs):
        return 1 - x ** 2

    def log(self, x, **kwargs):
        b = 1 / (np.e - 1)
        a = np.log(b) + 1
        return a - np.log(x + b)

    def brachistochrone(self, x, **kwargs):
        def equation(t):
            return t - np.sin(t) - (x / self.r)
        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return y0


class FoveaBlurTorch(nn.Module):
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        super().__init__()
        assert blur_kernel_size % 2 == 1 and blur_kernel_size >= 3, "blur_kernel_size 必须为 >=3 的奇数"
        self.blur_kernel_size = int(blur_kernel_size)
        self.h, self.w = int(h), int(w)
        self.curve_type = curve_type
        self.kwargs = kwargs

        self.mask = np.zeros((h, w), np.float32)

        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1 - c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]

        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]

        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        if fun_degrade is None:
            raise ValueError(f"Unknown curve_type: {curve_type}")

        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1, distance / max_distance)
                y0 = fun_degrade(x0, **kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

        self.register_buffer("mask_t", torch.from_numpy(self.mask).float())  # [H,W]

    def alphaBlend(self, img1, img2, mask):
        return _blend_with_mask(img1, img2, mask)

    # def __call__(self, img, blur_kernel_size=None):
    #     return self._process_single(img, blur_kernel_size)

    def forward(self, x):
        """
        x: torch.Tensor
           - (B, C, H, W) 或 (C, H, W)，通常范围[0,1]或[0,255]
        返回: 同形状 torch.Tensor，dtype/设备与输入一致
        """
        orig_device = x.device
        orig_dtype = x.dtype

        if x.dim() == 3:
            x = x.unsqueeze(0)  # -> (1, C, H, W)

        B, C, H, W = x.shape
        if (H != self.h) or (W != self.w):
            raise ValueError(f"输入尺寸 {H}x{W} 与 FoveaBlur 初始化的 {self.h}x{self.w} 不一致。")

        x_cpu = x.detach().to("cpu")
        out_list = []
        for b in range(B):
            img_bchw = x_cpu[b]
            img_pil = TF.to_pil_image(img_bchw.clamp(0, 1) if img_bchw.max() <= 1.0 else (img_bchw / 255.0).clamp(0,1))
            out_pil = self._process_single(img_pil, blur_kernel_size=None)
            out_tensor = TF.to_tensor(out_pil)  # float32, [0,1]
            if x_cpu.max() > 1.0:
                out_tensor = (out_tensor * 255.0).clamp(0, 255)
            out_list.append(out_tensor)

        out = torch.stack(out_list, dim=0).to(orig_device).to(orig_dtype)
        return out

    def _process_single(self, img, blur_kernel_size=None):
        if blur_kernel_size is None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        img = _rgb_to_bgr(img)
        blured = _gaussian_blur_np(img, blur_kernel_size)
        blended = self.alphaBlend(img, blured, 1 - self.mask)
        blended = _bgr_to_rgb(blended)
        return Image.fromarray(blended)

    def linear(self, x, **kwargs):
        return 1 - x

    def exp(self, x, **kwargs):
        system_g = kwargs.get('system_g', 4)
        return np.exp(-system_g * x)

    def quadratic(self, x, **kwargs):
        return 1 - x ** 2

    def log(self, x, **kwargs):
        b = 1 / (np.e - 1)
        a = np.log(b) + 1
        return a - np.log(x + b)

    def brachistochrone(self, x, **kwargs):
        def equation(t):
            return t - np.sin(t) - (x / self.r)
        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return y0


class FoveaBlurTorch_Fast(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        blur_kernel_size: int = 51,
        curve_type: str = 'exp',
        system_g: float = 3.0,
        center: tuple | None = None,
    ):
        super().__init__()
        assert blur_kernel_size % 2 == 1 and blur_kernel_size >= 3, \
            "blur_kernel_size 必须为 >=3 的奇数"
        self.h, self.w = int(h), int(w)
        self.kernel_size = int(blur_kernel_size)
        self.curve_type = curve_type
        self.system_g = float(system_g)
        self.center = center  # (cx, cy) in pixels or None

        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32),
            torch.arange(self.w, dtype=torch.float32),
            indexing='ij'
        )
        if center is None:
            cx, cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
        else:
            cx, cy = float(center[0]), float(center[1])

        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        max_dist = torch.sqrt(torch.tensor(((self.h - 1 - cy) ** 2) + ((self.w - 1 - cx) ** 2))) + 1e-8
        x0 = torch.clamp(dist / max_dist, 0.0, 1.0)  # 半径归一化

        y = self._curve(x0)              # [H,W]
        alpha = (1.0 - y).clamp(0.0, 1.0)  # 中心 0 边缘 1
        self.register_buffer("alpha", alpha[None, None, ...])  # [1,1,H,W]

        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        self.kernel_hw = (self.kernel_size, self.kernel_size)
        self.sigma_hw = (float(sigma), float(sigma))

    def _curve(self, x: torch.Tensor) -> torch.Tensor:
        if self.curve_type == "linear":
            return 1.0 - x
        elif self.curve_type == "exp":
            return torch.exp(-self.system_g * x)
        elif self.curve_type == "quadratic":
            return 1.0 - x ** 2
        elif self.curve_type == "log":
            b = 1.0 / (math.e - 1.0)
            a = math.log(b) + 1.0
            return (a - torch.log(x + b)).clamp(0.0, 1.0)
        elif self.curve_type == "brachistochrone":
            return 0.5 * (1.0 + torch.cos(math.pi * x))
        else:
            raise ValueError(f"Unknown curve_type: {self.curve_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True

        B, C, H, W = x.shape
        if not x.dtype.is_floating_point:
            x = x.float()

        blurred = kornia.filters.gaussian_blur2d(
            x, kernel_size=self.kernel_hw, sigma=self.sigma_hw, border_type='reflect'
        )

        alpha = self.alpha
        if alpha.shape[-2:] != (H, W):
            alpha = F.interpolate(alpha, size=(H, W), mode='bilinear', align_corners=False)
        alpha = alpha.expand(B, C, H, W).to(x.device)

        out = x * (1.0 - alpha) + blurred * alpha
        out = out.clamp(0.0, 1.0)
        if squeeze:
            out = out.squeeze(0)
        return out


class FoveaBlurTorch_Fast_adaptive(nn.Module):
    def __init__(
        self,
        h: int,
        w: int,
        blur_kernel_size: int = 51,
        curve_type: str = 'exp',
        system_g: float = 3.0,
        center: tuple | None = None,   # 仅用于默认值（不再固化到 buffer）
    ):
        super().__init__()
        assert blur_kernel_size % 2 == 1 and blur_kernel_size >= 3, \
            "blur_kernel_size 必须为 >=3 的奇数"
        self.h, self.w = int(h), int(w)
        self.kernel_size = int(blur_kernel_size)
        self.curve_type = curve_type
        self.system_g = float(system_g)

        if center is None:
            self.default_center = None
        else:
            self.default_center = (float(center[0]), float(center[1]))

        sigma = 0.3 * ((self.kernel_size - 1) * 0.5 - 1) + 0.8
        self.kernel_hw = (self.kernel_size, self.kernel_size)
        self.sigma_hw = (float(sigma), float(sigma))

    def _curve(self, x: torch.Tensor) -> torch.Tensor:
        if self.curve_type == "linear":
            return 1.0 - x
        elif self.curve_type == "exp":
            return torch.exp(-self.system_g * x)
        elif self.curve_type == "quadratic":
            return 1.0 - x ** 2
        elif self.curve_type == "log":
            b = 1.0 / (math.e - 1.0)
            a = math.log(b) + 1.0
            return (a - torch.log(x + b)).clamp(0.0, 1.0)
        elif self.curve_type == "brachistochrone":
            # 平滑的近似：cos 曲线
            return 0.5 * (1.0 + torch.cos(math.pi * x))
        else:
            raise ValueError(f"Unknown curve_type: {self.curve_type}")

    @staticmethod
    def _to_bchw(x: torch.Tensor):
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        return x, squeeze

    def _build_alpha(
        self,
        H: int, W: int,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
        center,
        r_px,
    ) -> torch.Tensor:
        """
        返回 alpha: (B,1,H,W), 中心 0、边缘 1
        - center: None / (cx,cy) / Tensor[B,2]
        - r_px  : None / float / Tensor[B]
        """
        # 网格 [1,1,H,W]
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        yy = yy.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        xx = xx.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # 处理 center
        if center is None and self.default_center is None:
            # 几何中心
            cx = (W - 1) * 0.5
            cy = (H - 1) * 0.5
            cx = torch.full((B, 1, 1, 1), cx, device=device, dtype=dtype)
            cy = torch.full((B, 1, 1, 1), cy, device=device, dtype=dtype)
        elif center is None and self.default_center is not None:
            cx = torch.full((B, 1, 1, 1), self.default_center[0], device=device, dtype=dtype)
            cy = torch.full((B, 1, 1, 1), self.default_center[1], device=device, dtype=dtype)
        else:
            if isinstance(center, (tuple, list)):
                cx = torch.full((B, 1, 1, 1), float(center[0]), device=device, dtype=dtype)
                cy = torch.full((B, 1, 1, 1), float(center[1]), device=device, dtype=dtype)
            elif isinstance(center, torch.Tensor):
                assert center.shape == (B, 2), "center 张量应为 (B,2)"
                cx = center[:, 0].view(B, 1, 1, 1).to(device=device, dtype=dtype)
                cy = center[:, 1].view(B, 1, 1, 1).to(device=device, dtype=dtype)
            else:
                raise TypeError("center 必须是 None / (cx,cy) / Tensor[B,2]")

        # 处理 r_px
        if r_px is None:
            # r = torch.full((B, 1, 1, 1), float(self.kernel_size), device=device, dtype=dtype)
            d00 = torch.sqrt((cx - 0) ** 2 + (cy - 0) ** 2)
            d10 = torch.sqrt((cx - (W - 1)) ** 2 + (cy - 0) ** 2)
            d01 = torch.sqrt((cx - 0) ** 2 + (cy - (H - 1)) ** 2)
            d11 = torch.sqrt((cx - (W - 1)) ** 2 + (cy - (H - 1)) ** 2)
            r = torch.max(torch.max(d00, d10), torch.max(d01, d11))  # (B,1,1,1)
        elif isinstance(r_px, (int, float)):
            r = torch.full((B, 1, 1, 1), float(r_px), device=device, dtype=dtype)
        elif isinstance(r_px, torch.Tensor):
            assert r_px.shape == (B,), "r_px 张量应为 (B,)"
            r = r_px.view(B, 1, 1, 1).to(device=device, dtype=dtype)
        else:
            raise TypeError("r_px 必须是 None / float / Tensor[B]")

        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # (1,1,H,W) 与 (B,1,1,1) 广播 -> (B,1,H,W)
        x0 = (dist / torch.clamp(r, min=1.0)).clamp(0.0, 1.0)  # dist / r_px

        y = self._curve(x0)          # (B,1,H,W) ∈ [0,1]
        alpha = (1.0 - y).clamp(0.0, 1.0)
        return alpha  # (B,1,H,W)

    def forward(
        self,
        x: torch.Tensor,
        center: torch.Tensor | None = None,
        r_px: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:  (C,H,W) 或 (B,C,H,W), 值域[0,1]或其他浮点都可
        center: None / (cx,cy) / Tensor[B,2]
        r_px  : None / float / Tensor[B]
        """
        # center = keypoints
        # print("received center:", center.shape)
        x, squeeze = self._to_bchw(x)
        B, C, H, W = x.shape
        if not x.dtype.is_floating_point:
            x = x.float()

        # 模糊图
        blurred = kornia.filters.gaussian_blur2d(
            x, kernel_size=self.kernel_hw, sigma=self.sigma_hw, border_type='reflect'
        )

        # 按需构建 alpha（中心 0，边缘 1）
        alpha = self._build_alpha(
            H=H, W=W, B=B, device=x.device, dtype=x.dtype,
            center=center, r_px=r_px
        )  # (B,1,H,W)
        alpha = alpha.expand(B, C, H, W)

        # alpha = self.alpha
        # if alpha.shape[-2:] != (H, W):
        #     alpha = F.interpolate(alpha, size=(H, W), mode='bilinear', align_corners=False)
        # alpha = alpha.expand(B, C, H, W).to(x.device)

        out = x * (1.0 - alpha) + blurred * alpha
        out = out.clamp(0.0, 1.0)
        if squeeze:
            out = out.squeeze(0)
        return out
