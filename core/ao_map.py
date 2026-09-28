"""Ambient Occlusion (AO) map generation.

Derives a high-fidelity contact shadow and crevice occlusion map from
height field relief using multi-directional horizon ray sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass
class AOMapOptions:
    strength: float = 1.0
    radius: int = 16
    samples: int = 8
    contrast: float = 1.5
    blur_radius: int = 1
    invert: bool = False
    bit_depth: int = 8


def generate_ao_map(
    height: np.ndarray,
    options: AOMapOptions | None = None,
) -> np.ndarray:
    """Derive an Ambient Occlusion array in [0, 1] from a float32 height array in [0, 1].

    Returns float32 HxW array in [0, 1] where 1.0 = fully illuminated surface,
    0.0 = deeply occluded crevice.
    """
    options = options or AOMapOptions()
    h_arr = height.astype(np.float32)
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

    occ_scaled = np.clip(total_occ * options.strength * 2.2, 0.0, 1.0)
    ao = 1.0 - occ_scaled

    contrast = max(0.1, options.contrast)
    ao = np.power(ao, contrast)

    if options.blur_radius > 0:
        k = options.blur_radius * 2 + 1
        ao = cv2.GaussianBlur(ao, (k, k), 0)

    if options.invert:
        ao = 1.0 - ao

    return np.clip(ao, 0.0, 1.0).astype(np.float32)


def to_image(ao: np.ndarray, bit_depth: int = 8) -> Image.Image:
    """Convert float32 [0, 1] AO array to PIL Grayscale Image (8, 16, or 32-bit)."""
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
