from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


ROUGHNESS_PRESETS: dict[str, dict] = {
    "Custom": {
        "base_roughness": 0.3,
        "strength": 1.0,
        "height_cavity": 0.0,
        "albedo_contrast": 0.0,
    },
    "Stone / Concrete": {
        "base_roughness": 0.75,
        "strength": 1.2,
        "height_cavity": 0.35,
        "albedo_contrast": 0.25,
    },
    "Wood": {
        "base_roughness": 0.65,
        "strength": 0.8,
        "height_cavity": 0.20,
        "albedo_contrast": 0.40,
    },
    "Varnished Wood": {
        "base_roughness": 0.35,
        "strength": 0.5,
        "height_cavity": 0.25,
        "albedo_contrast": 0.20,
    },
    "Polished Metal": {
        "base_roughness": 0.15,
        "strength": 0.4,
        "height_cavity": 0.10,
        "albedo_contrast": 0.05,
    },
    "Ceramic / Tile": {
        "base_roughness": 0.20,
        "strength": 0.6,
        "height_cavity": 0.50,
        "albedo_contrast": 0.10,
    },
    "Wet / Muddy": {
        "base_roughness": 0.25,
        "strength": 1.0,
        "height_cavity": -0.60,
        "albedo_contrast": 0.20,
    },
    "Fabric / Leather": {
        "base_roughness": 0.85,
        "strength": 0.5,
        "height_cavity": 0.15,
        "albedo_contrast": 0.30,
    },
}


@dataclass
class RoughnessMapOptions:
    base_roughness: float = 0.3
    strength: float = 1.0
    window_pct: float = 2.0
    invert: bool = False
    imperfection_mask: np.ndarray | None = None
    imperfection_strength: float = 1.0
    preset_name: str = "Custom"
    height_cavity_factor: float = 0.0
    albedo_contrast_factor: float = 0.0
    height_source: np.ndarray | None = None
    albedo_source: Image.Image | np.ndarray | None = None


def generate_roughness_map(normal: np.ndarray, options: RoughnessMapOptions | None = None) -> np.ndarray:
    options = options or RoughnessMapOptions()
    h, w = normal.shape[:2]

    n = normal.astype(np.float32) * 2.0 - 1.0
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(length, 1e-8)

    window_px = max(3, int(round(min(h, w) * options.window_pct / 100.0)))
    if window_px % 2 == 0:
        window_px += 1
    mean_n = cv2.boxFilter(n, ddepth=-1, ksize=(window_px, window_px))
    avg_len = np.clip(np.linalg.norm(mean_n, axis=-1), 0.0, 1.0)

    variance_signal = 1.0 - avg_len
    roughness = options.base_roughness + options.strength * variance_signal

    if options.height_source is not None and abs(options.height_cavity_factor) > 1e-3:
        h_arr = options.height_source.astype(np.float32)
        if h_arr.shape[0] != h or h_arr.shape[1] != w:
            h_arr = cv2.resize(h_arr, (w, h), interpolation=cv2.INTER_LINEAR)
        h_smooth = cv2.GaussianBlur(h_arr, (9, 9), 0)
        cavity = np.clip((h_smooth - h_arr) * 4.0, -1.0, 1.0)
        if options.height_cavity_factor > 0:
            roughness += options.height_cavity_factor * np.maximum(0.0, cavity)
        else:
            puddle_mask = np.maximum(0.0, 1.0 - h_arr)
            roughness += options.height_cavity_factor * puddle_mask

    if options.albedo_source is not None and options.albedo_contrast_factor > 1e-3:
        alb = options.albedo_source
        if isinstance(alb, Image.Image):
            lum = np.asarray(alb.convert("L"), dtype=np.float32) / 255.0
        elif alb.ndim == 3:
            lum = (0.2126 * alb[..., 0] + 0.7152 * alb[..., 1] + 0.0722 * alb[..., 2]).astype(np.float32) / 255.0
        else:
            lum = alb.astype(np.float32) / 255.0

        if lum.shape[0] != h or lum.shape[1] != w:
            lum = cv2.resize(lum, (w, h), interpolation=cv2.INTER_LINEAR)

        lum_var = np.abs(lum - cv2.GaussianBlur(lum, (7, 7), 0))
        roughness += options.albedo_contrast_factor * (lum_var * 2.5)

    if options.invert:
        roughness = 1.0 - roughness

    if options.imperfection_mask is not None and options.imperfection_strength > 1e-4:
        mask = options.imperfection_mask
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask.ndim == 3 and mask.shape[2] >= 4:
            if mask.dtype == np.uint8:
                target_r = mask[..., 0].astype(np.float32) / 255.0
                alpha = (mask[..., 3].astype(np.float32) / 255.0) * float(options.imperfection_strength)
            else:
                target_r = mask[..., 0].astype(np.float32)
                alpha = mask[..., 3].astype(np.float32) * float(options.imperfection_strength)
        elif mask.ndim == 3 and mask.shape[2] >= 2:
            target_r = mask[..., 0].astype(np.float32)
            alpha = mask[..., 1].astype(np.float32) * float(options.imperfection_strength)
        else:
            if mask.dtype == np.uint8:
                target_r = mask.astype(np.float32) / 255.0
            else:
                target_r = mask.astype(np.float32)
            alpha = np.ones_like(target_r) * float(options.imperfection_strength)

        alpha = np.clip(alpha, 0.0, 1.0)
        roughness = (1.0 - alpha) * roughness + alpha * target_r

    return np.clip(roughness, 0.0, 1.0).astype(np.float32)


def to_image(roughness: np.ndarray) -> Image.Image:
    arr = (np.clip(roughness, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def save_roughness_map(roughness: np.ndarray, path: str) -> None:
    to_image(roughness).save(path)
