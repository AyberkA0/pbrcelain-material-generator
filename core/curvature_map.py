from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_SCALES = (0.002, 0.006, 0.018)


def generate_curvature_map(height: np.ndarray) -> np.ndarray:
    h_arr = height.astype(np.float32)
    w = h_arr.shape[1]
    sigmas = [max(0.6, s * w) for s in _SCALES]
    pad = int(np.ceil(max(sigmas) * 4)) + 1
    padded = np.pad(h_arr, pad, mode="wrap")

    curvature = np.zeros_like(padded)
    for sigma in sigmas:
        curvature += padded - cv2.GaussianBlur(padded, (0, 0), sigma)
    curvature = curvature[pad:-pad, pad:-pad]

    scale = float(np.percentile(np.abs(curvature), 99.5))
    if scale < 1e-8:
        return np.full_like(h_arr, 0.5)
    return np.clip(0.5 + 0.5 * curvature / scale, 0.0, 1.0).astype(np.float32)


def to_image(curvature: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(curvature, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L")
