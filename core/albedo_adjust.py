from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class ColorShiftRule:
    enabled: bool = False
    source_rgb: tuple[int, int, int] = (200, 200, 200)
    target_rgb: tuple[int, int, int] = (120, 120, 120)
    tolerance: float = 40.0
    falloff: float = 1.0


@dataclass
class AlbedoAdjustOptions:
    white_balance_enabled: bool = False
    white_balance_strength: float = 0.7
    seamless_enabled: bool = False
    seamless_overlap_pct: float = 10.0


AlbedoGenerateOptions = AlbedoAdjustOptions


def make_seamless_image(
    image: Image.Image | np.ndarray,
    overlap_pct: float = 10.0,
) -> np.ndarray:
    """Make any arbitrary image seamlessly tileable using smoothstep border cross-blending."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    else:
        arr = image.astype(np.float32).copy()

    h, w = arr.shape[:2]
    mx = max(2, int(round(w * overlap_pct / 100.0)))
    my = max(2, int(round(h * overlap_pct / 100.0)))

    for d in range(mx):
        t = d / float(mx)
        wt = t * t * (3.0 - 2.0 * t)
        left_col = arr[:, d].copy()
        right_col = arr[:, w - 1 - d].copy()
        s = 0.5 * (left_col + right_col)
        arr[:, d] = (1.0 - wt) * s + wt * left_col
        arr[:, w - 1 - d] = (1.0 - wt) * s + wt * right_col

    for d in range(my):
        t = d / float(my)
        wt = t * t * (3.0 - 2.0 * t)
        top_row = arr[d, :].copy()
        bot_row = arr[h - 1 - d, :].copy()
        s = 0.5 * (top_row + bot_row)
        arr[d, :] = (1.0 - wt) * s + wt * top_row
        arr[h - 1 - d, :] = (1.0 - wt) * s + wt * bot_row

    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def auto_white_balance(
    image: Image.Image | np.ndarray,
    strength: float = 0.7,
) -> np.ndarray:
    """Neutralize color cast using the Gray-World illuminant normalization."""
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        arr = image.astype(np.float32)

    mean_r = float(np.mean(arr[..., 0]))
    mean_g = float(np.mean(arr[..., 1]))
    mean_b = float(np.mean(arr[..., 2]))
    gray = (mean_r + mean_g + mean_b) / 3.0

    kr = (gray / max(mean_r, 1e-4) - 1.0) * strength + 1.0
    kg = (gray / max(mean_g, 1e-4) - 1.0) * strength + 1.0
    kb = (gray / max(mean_b, 1e-4) - 1.0) * strength + 1.0

    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0] * kr
    out[..., 1] = arr[..., 1] * kg
    out[..., 2] = arr[..., 2] * kb
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def apply_albedo_adjustments(
    image: Image.Image | np.ndarray,
    options: AlbedoAdjustOptions | None = None,
    rules: list[ColorShiftRule] | None = None,
) -> Image.Image:
    """Apply complete Albedo Adjustment pipeline: White Balance -> Color Shifts -> Seamless Tiling."""
    options = options or AlbedoAdjustOptions()
    rules = rules or []

    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = image.astype(np.uint8)

    if options.white_balance_enabled and options.white_balance_strength > 1e-3:
        arr = auto_white_balance(arr, strength=options.white_balance_strength)

    img = Image.fromarray(arr, mode="RGB")
    if rules:
        img = apply_color_shifts(img, rules)
        arr = np.asarray(img, dtype=np.uint8)

    if options.seamless_enabled and options.seamless_overlap_pct > 1e-3:
        arr = make_seamless_image(arr, overlap_pct=options.seamless_overlap_pct)

    return Image.fromarray(arr, mode="RGB")


def generate_albedo_map(
    image: Image.Image | np.ndarray,
    options: AlbedoAdjustOptions | None = None,
) -> Image.Image:
    return apply_albedo_adjustments(image, options=options)


def apply_color_shifts(
    image: Image.Image | np.ndarray,
    rules: list[ColorShiftRule],
) -> Image.Image:
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        arr = image.astype(np.float32)

    active_rules = [r for r in rules if r.enabled and r.tolerance > 0.1]
    if not active_rules:
        if isinstance(image, Image.Image):
            return image
        return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), mode="RGB")

    out = np.copy(arr)

    for rule in active_rules:
        src = np.array(rule.source_rgb, dtype=np.float32)
        dst = np.array(rule.target_rgb, dtype=np.float32)
        delta = dst - src

        diff = out - src[np.newaxis, np.newaxis, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))

        tol = max(rule.tolerance, 1e-4)
        norm_dist = np.clip(dist / tol, 0.0, 1.0)
        weight = np.power(1.0 - norm_dist, max(rule.falloff, 0.1))[:, :, np.newaxis]

        out = out + delta[np.newaxis, np.newaxis, :] * weight
        out = np.clip(out, 0.0, 255.0)

    res = np.clip(out, 0, 255).round().astype(np.uint8)
    return Image.fromarray(res, mode="RGB")
