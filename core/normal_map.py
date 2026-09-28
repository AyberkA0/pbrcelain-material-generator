"""Height-map -> tangent-space normal-map conversion.

Independent of any specific depth model or UI: operates on a plain float32
HxW height array in [0, 1], as produced by core.height_map.build_height_map.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class NormalMapOptions:
    strength: float = 1.0
    invert_y: bool = False
    pre_blur_radius: int = 0
    seamless: bool = True
    multi_frequency: bool = False
    macro_strength: float = 1.0
    medium_strength: float = 1.0
    micro_strength: float = 1.0
    inject_albedo: bool = False
    albedo_strength: float = 0.5
    albedo_source: Image.Image | np.ndarray | None = None


_REFERENCE_DIM = 64.0


def generate_normal_map(height: np.ndarray, options: NormalMapOptions | None = None) -> np.ndarray:
    """Derive a tangent-space normal map from a [0,1] float32 height array.

    Returns a float32 HxWx3 array in [0, 1] (ready for to_image/RGB encoding).
    """
    options = options or NormalMapOptions()
    working = height.astype(np.float32)
    h, w = working.shape

    margin = 0
    if options.seamless:
        margin = max(options.pre_blur_radius, 12 if options.multi_frequency else 0) + 3
        working = np.pad(working, margin, mode="wrap")

    if options.pre_blur_radius > 0:
        k = options.pre_blur_radius * 2 + 1
        working = cv2.GaussianBlur(working, (k, k), 0)

    scale = options.strength * (max(h, w) / _REFERENCE_DIM)

    if options.multi_frequency:
        k_macro = max(9, int(min(h, w) * 0.04) * 2 + 1)
        h_macro = cv2.GaussianBlur(working, (k_macro, k_macro), 0)
        gx_macro = cv2.Sobel(h_macro, cv2.CV_32F, 1, 0, ksize=3) * scale * options.macro_strength
        gy_macro = cv2.Sobel(h_macro, cv2.CV_32F, 0, 1, ksize=3) * scale * options.macro_strength

        k_med = max(3, int(min(h, w) * 0.01) * 2 + 1)
        h_med = cv2.GaussianBlur(working, (k_med, k_med), 0)
        gx_med = cv2.Sobel(h_med - h_macro, cv2.CV_32F, 1, 0, ksize=3) * scale * options.medium_strength
        gy_med = cv2.Sobel(h_med - h_macro, cv2.CV_32F, 0, 1, ksize=3) * scale * options.medium_strength

        h_micro = working - h_med
        gx_micro = cv2.Sobel(h_micro, cv2.CV_32F, 1, 0, ksize=3) * scale * options.micro_strength
        gy_micro = cv2.Sobel(h_micro, cv2.CV_32F, 0, 1, ksize=3) * scale * options.micro_strength

        gx = gx_macro + gx_med + gx_micro
        gy = gy_macro + gy_med + gy_micro
    else:
        gx = cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3) * scale
        gy = cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3) * scale

    if options.inject_albedo and options.albedo_source is not None and options.albedo_strength > 1e-4:
        alb = options.albedo_source
        if isinstance(alb, Image.Image):
            alb_arr = np.asarray(alb.convert("L"), dtype=np.float32) / 255.0
        elif alb.ndim == 3:
            alb_arr = (0.2126 * alb[..., 0] + 0.7152 * alb[..., 1] + 0.0722 * alb[..., 2]).astype(np.float32) / 255.0
        else:
            alb_arr = alb.astype(np.float32) / 255.0

        if alb_arr.shape[0] != h or alb_arr.shape[1] != w:
            alb_arr = cv2.resize(alb_arr, (w, h), interpolation=cv2.INTER_LINEAR)

        if margin:
            alb_arr = np.pad(alb_arr, margin, mode="wrap" if options.seamless else "reflect")

        alb_blur = cv2.GaussianBlur(alb_arr, (5, 5), 0)
        alb_hp = alb_arr - alb_blur
        gx += cv2.Sobel(alb_hp, cv2.CV_32F, 1, 0, ksize=3) * scale * (options.albedo_strength * 1.5)
        gy += cv2.Sobel(alb_hp, cv2.CV_32F, 0, 1, ksize=3) * scale * (options.albedo_strength * 1.5)

    if options.invert_y:
        gy = -gy

    nz = np.ones_like(working)
    normal = np.stack([-gx, -gy, nz], axis=-1)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(length, 1e-8)

    if margin:
        normal = normal[margin : margin + h, margin : margin + w]

    return (normal * 0.5 + 0.5).astype(np.float32)


def to_image(normal: np.ndarray) -> Image.Image:
    """Convert a [0,1] float32 HxWx3 normal array into an 8-bit RGB PIL image."""
    arr = (np.clip(normal, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_normal_map(normal: np.ndarray, path: str) -> None:
    to_image(normal).save(path)
