from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


AO_METHOD_RAYTRACED = "raytraced"
AO_METHOD_CLASSIC = "classic"


@dataclass
class AOMapOptions:
    strength: float = 1.0
    radius: int = 16
    samples: int = 8
    contrast: float = 1.5
    blur_radius: int = 1
    invert: bool = False
    bit_depth: int = 8
    method: str = AO_METHOD_CLASSIC
    surface_width_cm: float = 50.0
    max_depth_mm: float = 10.0
    search_percent: float = 3.0


def generate_ao_map(
    height: np.ndarray,
    options: AOMapOptions | None = None,
) -> np.ndarray:
    options = options or AOMapOptions()
    h_arr = height.astype(np.float32)
    if options.method == AO_METHOD_RAYTRACED:
        occlusion = _raytraced_occlusion(h_arr, options)
    else:
        occlusion = _classic_occlusion(h_arr, options)

    ao = 1.0 - np.clip(occlusion * options.strength, 0.0, 1.0)
    contrast = max(0.1, options.contrast)
    ao = np.power(ao, contrast)

    if options.blur_radius > 0:
        k = options.blur_radius * 2 + 1
        ao = cv2.GaussianBlur(ao, (k, k), 0)

    if options.invert:
        ao = 1.0 - ao

    return np.clip(ao, 0.0, 1.0).astype(np.float32)


def _classic_occlusion(h_arr: np.ndarray, options: AOMapOptions) -> np.ndarray:
    h, w = h_arr.shape[:2]
    rad_factors = [0.25, 0.50, 0.75, 1.0]
    radii = [max(1, int(round(options.radius * f))) for f in rad_factors]
    angles = np.linspace(0, 2 * np.pi, max(4, options.samples), endpoint=False)

    total_occ = np.zeros((h, w), dtype=np.float32)
    sample_count = 0
    for r in radii:
        for theta in angles:
            dx = int(round(r * np.cos(theta)))
            dy = int(round(r * np.sin(theta)))
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(h_arr, dy, axis=0), dx, axis=1)
            dist = np.hypot(dx, dy)
            diff = np.maximum(0.0, shifted - h_arr)
            slope = diff / (dist * 0.04 + 1e-2)
            total_occ += slope
            sample_count += 1
    if sample_count > 0:
        total_occ /= sample_count
    return total_occ * 2.2


def _raytraced_occlusion(h_arr: np.ndarray, options: AOMapOptions) -> np.ndarray:
    h, w = h_arr.shape[:2]
    depth_ratio = (options.max_depth_mm / 10.0) / max(options.surface_width_cm, 1e-3)
    height_px = h_arr * (depth_ratio * w)

    max_dist = max(2.0, options.search_percent / 100.0 * w)
    steps = np.unique(np.round(np.geomspace(1.0, max_dist, num=12)).astype(int))
    directions = max(4, options.samples)

    occlusion = np.zeros((h, w), dtype=np.float32)
    for k in range(directions):
        theta = 2.0 * np.pi * (k + 0.5) / directions
        cx, cy = np.cos(theta), np.sin(theta)

        fwd = _shifted(height_px, cx, cy)
        back = _shifted(height_px, -cx, -cy)
        tan_t = (fwd - back) * 0.5
        sin_t = tan_t / np.sqrt(1.0 + tan_t * tan_t)

        best = np.zeros((h, w), dtype=np.float32)
        for d in steps:
            sample = _shifted(height_px, cx * d, cy * d)
            tan_h = (sample - height_px) / float(d)
            sin_h = tan_h / np.sqrt(1.0 + tan_h * tan_h)
            falloff = 1.0 - (float(d) / max_dist) ** 2
            np.maximum(best, (sin_h - sin_t) * falloff, out=best)
        occlusion += best

    return occlusion / directions


def _shifted(arr: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.roll(np.roll(arr, -int(round(dy)), axis=0), -int(round(dx)), axis=1)


def to_image(ao: np.ndarray, bit_depth: int = 8) -> Image.Image:
    ao_clipped = np.clip(ao, 0.0, 1.0)
    if bit_depth == 32:
        return Image.fromarray(ao_clipped.astype(np.float32), mode="F")
    if bit_depth == 16:
        arr16 = (ao_clipped * 65535.0).round().astype(np.uint16)
        return Image.fromarray(arr16)
    arr8 = (ao_clipped * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr8, mode="L")


def save_ao_map(ao: np.ndarray, path: str, bit_depth: int = 8) -> None:
    to_image(ao, bit_depth=bit_depth).save(path)
