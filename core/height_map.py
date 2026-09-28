"""Post-processing of raw depth-model output into a usable height map.

Kept independent of any specific depth model: everything here operates on
a plain float32 HxW numpy array.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os
import cv2
import numpy as np
from PIL import Image


DETREND_NONE = "none"
DETREND_PLANE = "plane"
DETREND_QUADRATIC = "quadratic"
DETREND_HIGHPASS = "highpass"

PREVIEW_MAX_DIM = 1024


def downsample_for_preview(depth: np.ndarray, max_dim: int = PREVIEW_MAX_DIM) -> np.ndarray:
    """Shrink a depth array before running the pipeline for on-screen preview.

    normalize()'s percentile computation and the polynomial-surface detrend
    scale with pixel count, so running them on a full 4K+ array on every
    slider tweak or live chunk-estimation update is what makes the UI
    stutter. Only the actual save path should use full resolution.
    """
    h, w = depth.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return depth
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)


DEFAULT_BOWL_CURVE: list[tuple[float, float]] = [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]


@dataclass
class HeightMapOptions:
    invert: bool = False
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    gamma: float = 1.0
    blur_radius: int = 0
    bit_depth: int = 16
    detrend_mode: str = DETREND_QUADRATIC
    detrend_radius_pct: float = 8.0
    bowl_curve_x: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_BOWL_CURVE))
    bowl_curve_y: list[tuple[float, float]] = field(default_factory=lambda: list(DEFAULT_BOWL_CURVE))
    bowl_curve: list[tuple[float, float]] | None = None
    seamless: bool = False
    seamless_feather_pct: float = 6.0
    local_equalization: float = 0.0
    crevice_suppression: float = 0.0
    albedo_guidance: float = 0.0

    def __post_init__(self):
        if self.bowl_curve is not None:
            if not self.bowl_curve_x or self.bowl_curve_x == DEFAULT_BOWL_CURVE:
                self.bowl_curve_x = list(self.bowl_curve)
            if not self.bowl_curve_y or self.bowl_curve_y == DEFAULT_BOWL_CURVE:
                self.bowl_curve_y = list(self.bowl_curve)


def fit_polynomial_surface(depth: np.ndarray, order: int) -> np.ndarray:
    """Least-squares fit of a low-order 2D polynomial to `depth`.

    Used to model the large-scale "bowl"/tilt shape that monocular depth
    models impose even on flat, orthographic-style material photos (they are
    trained on perspective scenes and learn that image edges recede).
    order=1 fits a tilted plane, order=2 adds the symmetric bowl/dome term.
    """
    h, w = depth.shape
    step = max(1, max(h, w) // 200)
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    zz = depth[yy, xx].astype(np.float64)

    ny = (yy.astype(np.float64) / max(h - 1, 1)) * 2 - 1
    nx = (xx.astype(np.float64) / max(w - 1, 1)) * 2 - 1

    terms = [np.ones_like(nx.ravel())]
    if order >= 1:
        terms += [nx.ravel(), ny.ravel()]
    if order >= 2:
        terms += [nx.ravel() ** 2, ny.ravel() ** 2, (nx * ny).ravel()]
    a_matrix = np.stack(terms, axis=1)
    coeffs, *_ = np.linalg.lstsq(a_matrix, zz.ravel(), rcond=None)

    full_y = (np.arange(h, dtype=np.float32) / max(h - 1, 1)) * 2 - 1
    full_x = (np.arange(w, dtype=np.float32) / max(w - 1, 1)) * 2 - 1
    surface = np.full((h, w), np.float32(coeffs[0]), dtype=np.float32)
    if order >= 1:
        surface += np.float32(coeffs[1]) * full_x[np.newaxis, :]
        surface += np.float32(coeffs[2]) * full_y[:, np.newaxis]
    if order >= 2:
        surface += np.float32(coeffs[3]) * (full_x * full_x)[np.newaxis, :]
        surface += np.float32(coeffs[4]) * (full_y * full_y)[:, np.newaxis]
        surface += np.float32(coeffs[5]) * full_y[:, np.newaxis] * full_x[np.newaxis, :]
    return surface


def radial_distance_grid(h: int, w: int) -> np.ndarray:
    """Normalized distance from image center: 0 at center, 1 at the corners."""
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ny = (np.arange(h, dtype=np.float32) - cy) / max(cy, 1.0)
    nx = (np.arange(w, dtype=np.float32) - cx) / max(cx, 1.0)
    dist = np.sqrt((ny[:, np.newaxis] ** 2 + nx[np.newaxis, :] ** 2) / 2.0)
    return np.minimum(dist, 1.0, out=dist)


def compute_monotone_cubic_spline_slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Compute slopes for Monotone Cubic Hermite/Bezier Spline (Fritsch-Carlson).

    Guarantees monotonicity on each monotonic subinterval, eliminating
    overshoot, oscillations, and ringing artifacts when dragging points.
    """
    n = len(xs)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    if n == 2:
        m = (ys[1] - ys[0]) / max(xs[1] - xs[0], 1e-9)
        return np.array([m, m], dtype=np.float64)

    dx = np.diff(xs)
    dy = np.diff(ys)
    dx = np.where(dx == 0, 1e-9, dx)
    delta = dy / dx

    m = np.zeros(n, dtype=np.float64)
    m[0] = delta[0]
    m[-1] = delta[-1]

    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            m[i] = (delta[i - 1] + delta[i]) / 2.0

    for i in range(n - 1):
        if abs(delta[i]) < 1e-9:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            alpha = m[i] / delta[i]
            beta = m[i + 1] / delta[i]
            hypot = alpha * alpha + beta * beta
            if hypot > 9.0:
                tau = 3.0 / np.sqrt(hypot)
                m[i] = tau * alpha * delta[i]
                m[i + 1] = tau * beta * delta[i]

    return m


def compute_bezier_segments(
    points: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """Derive cubic Bezier control points (P0, C1, C2, P1) for each interval.

    Matches Monotone Hermite Spline slopes at each knot point.
    """
    if len(points) < 2:
        return []
    pts = sorted(points, key=lambda p: p[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    slopes = compute_monotone_cubic_spline_slopes(xs, ys)

    segments = []
    for i in range(len(pts) - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        dx = x1 - x0
        c1 = (float(x0 + dx / 3.0), float(y0 + slopes[i] * dx / 3.0))
        c2 = (float(x1 - dx / 3.0), float(y1 - slopes[i + 1] * dx / 3.0))
        segments.append(((float(x0), float(y0)), c1, c2, (float(x1), float(y1))))
    return segments


def evaluate_bezier_curve(points: list[tuple[float, float]], t: np.ndarray | float) -> np.ndarray:
    """Evaluate smooth cubic Bezier curve defined by control points at `t`.

    Uses analytic Monotone Hermite/Bezier formulation. Fast, C1 continuous,
    and guaranteed free of spurious local extrema or ringing.
    """
    is_scalar = np.isscalar(t)
    t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if not points:
        res = np.ones_like(t_arr, dtype=np.float32)
        return float(res[0]) if is_scalar else res
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) == 1:
        res = np.full_like(t_arr, pts[0][1], dtype=np.float32)
        return float(res[0]) if is_scalar else res

    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    slopes = compute_monotone_cubic_spline_slopes(xs, ys)

    idx = np.searchsorted(xs, t_arr, side="right")
    idx = np.clip(idx, 1, len(xs) - 1)
    i = idx - 1

    x0 = xs[i]
    x1 = xs[i + 1]
    y0 = ys[i]
    y1 = ys[i + 1]
    dx = np.maximum(x1 - x0, 1e-9)

    s = np.clip((t_arr - x0) / dx, 0.0, 1.0)
    s2 = s * s
    s3 = s2 * s
    one_minus_s = 1.0 - s
    one_minus_s2 = one_minus_s * one_minus_s
    one_minus_s3 = one_minus_s2 * one_minus_s

    c1y = y0 + slopes[i] * dx / 3.0
    c2y = y1 - slopes[i + 1] * dx / 3.0

    y = one_minus_s3 * y0 + 3.0 * one_minus_s2 * s * c1y + 3.0 * one_minus_s * s2 * c2y + s3 * y1

    y = np.where(t_arr < xs[0], ys[0], y)
    y = np.where(t_arr > xs[-1], ys[-1], y)

    out = y.astype(np.float32)
    return float(out[0]) if is_scalar else out


evaluate_curve = evaluate_bezier_curve


def apply_bowl_curve_2d(
    depth: np.ndarray,
    curve_x: list[tuple[float, float]] | None = None,
    curve_y: list[tuple[float, float]] | None = None,
    order: int = 2,
) -> np.ndarray:
    """Remove the fitted bowl/dome shape scaled independently across X and Y axes.

    Lenses are naturally anisotropic (horizontal vs vertical field of view,
    aspect ratio, or anamorphic distortion). This applies separable 2D
    multipliers M(y, x) = M_x(u) * M_y(v) so each axis can be tuned without
    unintended cross-talk.
    """
    surface = fit_polynomial_surface(depth, order=order)
    h, w = depth.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    nx = np.abs(np.arange(w, dtype=np.float32) - cx) / max(cx, 1.0)
    np.clip(nx, 0.0, 1.0, out=nx)

    ny = np.abs(np.arange(h, dtype=np.float32) - cy) / max(cy, 1.0)
    np.clip(ny, 0.0, 1.0, out=ny)

    cx_curve = curve_x or DEFAULT_BOWL_CURVE
    cy_curve = curve_y or DEFAULT_BOWL_CURVE

    mx = evaluate_bezier_curve(cx_curve, nx)
    my = evaluate_bezier_curve(cy_curve, ny)

    multiplier = my[:, np.newaxis] * mx[np.newaxis, :]
    return depth - surface * multiplier


def apply_bowl_curve(depth: np.ndarray, curve: list[tuple[float, float]], order: int = 2) -> np.ndarray:
    """Legacy radial bowl curve helper."""
    return apply_bowl_curve_2d(depth, curve_x=curve, curve_y=curve, order=order)


def detrend(
    depth: np.ndarray,
    mode: str,
    radius_pct: float = 20.0,
    bowl_curve: list[tuple[float, float]] | None = None,
    bowl_curve_x: list[tuple[float, float]] | None = None,
    bowl_curve_y: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Remove the large-scale depth shape, keeping only local relief."""
    if mode == DETREND_NONE:
        return depth
    if mode == DETREND_PLANE:
        return depth - fit_polynomial_surface(depth, order=1)
    if mode == DETREND_QUADRATIC:
        cx = bowl_curve_x or bowl_curve or DEFAULT_BOWL_CURVE
        cy = bowl_curve_y or bowl_curve or DEFAULT_BOWL_CURVE
        return apply_bowl_curve_2d(depth, curve_x=cx, curve_y=cy, order=2)
    if mode == DETREND_HIGHPASS:
        h, w = depth.shape
        radius_px = max(2, int(round((radius_pct / 100.0) * max(h, w) / 2)))
        k = radius_px * 2 + 1
        low_freq = cv2.GaussianBlur(depth, (k, k), 0)
        return depth - low_freq
    raise ValueError(f"Unknown detrend mode: {mode}")


def normalize(depth: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    """Min-max normalize using percentile clipping to ignore outliers."""
    lo, hi = np.percentile(depth, [low_percentile, high_percentile])
    if hi <= lo:
        hi = lo + 1e-6
    normed = (depth.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0)


def apply_gamma(normed: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return normed
    gamma = max(gamma, 1e-3)
    return np.power(normed, 1.0 / gamma)


def apply_smoothing(normed: np.ndarray, blur_radius: int) -> np.ndarray:
    if blur_radius <= 0:
        return normed
    k = blur_radius * 2 + 1
    return cv2.GaussianBlur(normed, (k, k), 0)


def _find_best_cut(diff_profile: np.ndarray, lo_frac: float = 0.4, hi_frac: float = 0.6) -> int:
    """Pick which column/row boundary becomes the new tile edge.

    `diff_profile[k]` is how much column/row k+1 differs from column/row k
    in the source image. Any roll amount gives naturally-continuous new
    edges *on average* (they come from originally-adjacent pixels) - but if
    the blindly-chosen boundary happens to land on a real sharp feature
    (a crack, a hard shadow edge), that mismatch gets baked straight into
    the tile edge. Searching a centered window for the smallest actual jump
    avoids that unlucky coincidence, while staying near the middle so the
    relocated original seam still has room to be blended away from the edges.
    """
    n = len(diff_profile)
    lo = max(1, int(round(n * lo_frac)))
    hi = min(n, int(round(n * hi_frac)))
    if hi <= lo:
        return n // 2 + 1
    window = diff_profile[lo:hi]
    return lo + int(np.argmin(window)) + 1


def seam_mismatch(arr: np.ndarray) -> dict:
    """Quantify how well `arr` would tile edge-to-edge, for verification.

    Reports the actual left/right and top/bottom edge jump alongside the
    image's typical local (pixel-to-pixel) jump, so "does the seam match"
    can be judged relative to the content instead of an arbitrary threshold.
    """
    typical = float(np.mean(np.abs(np.diff(arr, axis=1)))) + float(np.mean(np.abs(np.diff(arr, axis=0))))
    typical /= 2.0
    lr = float(np.mean(np.abs(arr[:, 0] - arr[:, -1])))
    tb = float(np.mean(np.abs(arr[0, :] - arr[-1, :])))
    return {
        "left_right_diff": lr,
        "top_bottom_diff": tb,
        "typical_local_diff": typical,
        "left_right_ratio": lr / typical if typical > 1e-9 else float("inf"),
        "top_bottom_ratio": tb / typical if typical > 1e-9 else float("inf"),
    }


def make_seamless(normed: np.ndarray, feather_pct: float = 6.0) -> np.ndarray:
    """Make a height map wrap seamlessly when tiled edge-to-edge.

    Offset-and-heal technique, with the cut location chosen by measurement
    rather than assumed: for each axis, the actual column-to-column (and
    row-to-row) jump is measured everywhere in a centered window, and the
    smallest one is used as the new tile edge - so the boundary that will be
    seen when tiled is verified to be a low-mismatch one, not just "the
    middle, hopefully". Rolling to put that boundary at position 0 moves the
    image's original (generally mismatched, since it wasn't seamless to
    begin with) edge into the interior as a single cross-shaped seam, which
    is then hidden by fading it into a locally blurred version - acceptable
    for a height map, where avoiding a visible seam matters more than
    preserving crisp detail in that one thin band.
    """
    h, w = normed.shape

    col_diff = np.mean(np.abs(np.diff(normed, axis=1)), axis=0)
    row_diff = np.mean(np.abs(np.diff(normed, axis=0)), axis=1)
    cut_x = _find_best_cut(col_diff)
    cut_y = _find_best_cut(row_diff)
    shift_x = (w - cut_x) % w
    shift_y = (h - cut_y) % h

    rolled = np.roll(normed, shift=(shift_y, shift_x), axis=(0, 1))

    feather_px = max(2, int(round(min(h, w) * feather_pct / 100.0)))
    k = feather_px * 2 + 1
    blurred = cv2.GaussianBlur(rolled, (k, k), 0)

    dist_x = np.abs(np.arange(w) - shift_x)
    dist_y = np.abs(np.arange(h) - shift_y)
    weight_x = np.clip(1.0 - dist_x / feather_px, 0.0, 1.0)
    weight_y = np.clip(1.0 - dist_y / feather_px, 0.0, 1.0)
    weight = np.maximum(weight_x[np.newaxis, :], weight_y[:, np.newaxis])

    res = (rolled * (1.0 - weight) + blurred * weight).astype(np.float32)
    try:
        res = apply_seamless_corner_heal(res, box_ratio=0.05)
    except Exception as e:
        print(f"[WARN] Failed to apply seamless corner heal: {e}")
    return res


def apply_local_equalization(normed: np.ndarray, strength: float, radius_pct: float = 6.0) -> np.ndarray:
    """Local Contrast Normalization / Baseline Leveling.

    Evens out wide regional lighting and macro-tilt differences so that brick
    reliefs throughout the entire texture lie on the same horizontal plane.
    """
    if strength <= 0.0:
        return normed

    h, w = normed.shape
    radius_px = max(3, int(round((radius_pct / 100.0) * max(h, w))))
    k = radius_px * 2 + 1

    local_mean = cv2.GaussianBlur(normed, (k, k), 0)

    diff_sq = (normed - local_mean) ** 2
    local_var = cv2.GaussianBlur(diff_sq, (k, k), 0)
    local_std = np.sqrt(np.maximum(local_var, 1e-5))

    normalized_relief = (normed - local_mean) / (local_std * 2.5 + 1e-4)
    leveled = np.clip(normalized_relief * 0.5 + 0.5, 0.0, 1.0)

    return np.clip((1.0 - strength) * normed + strength * leveled, 0.0, 1.0).astype(np.float32)


def apply_crevice_suppression(normed: np.ndarray, strength: float, radius_pct: float = 2.5) -> np.ndarray:
    """Morphological crevice & brick relief calibration.

    Dual-action relief enhancement:
    1. Deepens mortar lines/crevices (suppresses bright inverted joints).
    2. Elevates and inflates brick bodies (peaks) relative to joints.
    3. Renormalizes so bricks stand tall and crevices sit deep.
    """
    if abs(strength) < 1e-4:
        return normed

    h, w = normed.shape
    kernel_size = max(3, int(round((radius_pct / 100.0) * max(h, w))))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    blackhat = cv2.morphologyEx(normed, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(normed, cv2.MORPH_TOPHAT, kernel)

    out = np.copy(normed)
    if strength > 0.0:
        joint_suppression = tophat * (strength * 1.2) + blackhat * (strength * 0.8)
        brick_boost = (1.0 - np.clip(tophat + blackhat, 0.0, 1.0)) * (strength * 0.4)
        out = out - joint_suppression + brick_boost
    else:
        s = abs(strength)
        out = out + blackhat * (s * 1.0) - tophat * (s * 0.8)

    lo = float(np.percentile(out, 0.5))
    hi = float(np.percentile(out, 99.5))
    if hi > lo + 1e-5:
        out = (out - lo) / (hi - lo)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_albedo_guidance(
    normed: np.ndarray,
    guide: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Anchor joint / crevice depths using color/contrast from source Albedo.

    Mortar / joints typically have distinctive darker or sharp boundary features
    in the photo. Albedo guidance ensures that wherever a joint is detected in the
    source photo, it is firmly forced into the lower height range.
    """
    if strength <= 0.0 or guide is None:
        return normed

    h, w = normed.shape
    if guide.shape[:2] != (h, w):
        guide_resized = cv2.resize(guide, (w, h), interpolation=cv2.INTER_AREA)
    else:
        guide_resized = guide

    if guide_resized.ndim == 3:
        lum = cv2.cvtColor(guide_resized.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        lum = guide_resized.astype(np.float32)
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    grad_x = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_mag = np.clip(edge_mag / (np.percentile(edge_mag, 95) + 1e-5), 0.0, 1.0)

    target = np.minimum(normed, 1.0 - edge_mag * 0.6)
    guided = (1.0 - strength) * normed + strength * target
    return np.clip(guided, 0.0, 1.0).astype(np.float32)


def build_height_map(
    depth: np.ndarray,
    options: HeightMapOptions,
    guide: np.ndarray | None = None,
) -> np.ndarray:
    """Run the full pipeline and return a float32 array in [0, 1]."""
    working = detrend(
        depth,
        options.detrend_mode,
        options.detrend_radius_pct,
        bowl_curve_x=options.bowl_curve_x,
        bowl_curve_y=options.bowl_curve_y,
    )
    normed = normalize(working, options.low_percentile, options.high_percentile)
    if options.invert:
        normed = 1.0 - normed

    if options.local_equalization > 0.0:
        normed = apply_local_equalization(normed, options.local_equalization)

    if abs(options.crevice_suppression) > 1e-4:
        normed = apply_crevice_suppression(normed, options.crevice_suppression)

    if options.albedo_guidance > 0.0 and guide is not None:
        normed = apply_albedo_guidance(normed, guide, options.albedo_guidance)

    normed = apply_gamma(normed, options.gamma)
    normed = apply_smoothing(normed, options.blur_radius)
    if options.seamless:
        normed = make_seamless(normed, options.seamless_feather_pct)
    return np.clip(normed, 0.0, 1.0)


def to_image(normed: np.ndarray, bit_depth: int = 8) -> Image.Image:
    """Convert a float array into a grayscale PIL image according to bit depth."""
    if bit_depth == 32:
        return Image.fromarray(normed.astype(np.float32), mode="F")
    if bit_depth == 16:
        arr = (np.clip(normed, 0.0, 1.0) * 65535.0).round().astype(np.uint16)
        return Image.fromarray(arr)
    arr = (np.clip(normed, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def save_height_map(normed: np.ndarray, path: str, bit_depth: int = 8) -> None:
    img = to_image(normed, bit_depth)
    safe_save_image(img, path)


def get_desktop_dir() -> str:
    """Resolve the user's desktop directory path."""
    import os

    candidates = [
        os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop"),
        os.path.join(os.path.expanduser("~"), "Desktop"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.expanduser("~")


def build_seamless_extended_tile(
    image: Image.Image, padding_pct: float = 0.10
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Build a 3x3 tiled composite and symmetrically crop to center tile + margin.

    For example, with a 100x100 input and 10% padding per edge, the 3x3 canvas
    (300x300) is cropped to 120x120, preserving the full center tile at [pad_x, pad_x + W].
    Returns (extended_image, (pad_x, pad_y, pad_x + W, pad_y + H)).
    """
    rgb = image.convert("RGB")
    w, h = rgb.size

    pad_x = max(1, int(round(w * padding_pct)))
    pad_y = max(1, int(round(h * padding_pct)))

    tiled = Image.new("RGB", (w * 3, h * 3))
    for r in range(3):
        for c in range(3):
            tiled.paste(rgb, (c * w, r * h))

    crop_box = (w - pad_x, h - pad_y, 2 * w + pad_x, 2 * h + pad_y)
    extended = tiled.crop(crop_box)
    tile_box = (pad_x, pad_y, pad_x + w, pad_y + h)
    return extended, tile_box


def apply_force_seamless_blur(
    extended: Image.Image,
    tile_box: tuple[int, int, int, int],
    blur_pct: float = 0.03,
    max_passes: int = 5,
) -> Image.Image:
    """Apply a smooth band blur along the top and left frame boundary lines of extended.

    The blur band width is proportional to the tile dimensions (default: 3% of tile size).
    Iteratively blends with Gaussian blur along y = pad_y and x = pad_x until the sharp
    seam step discontinuity disappears, preserving crisp detail everywhere else.
    """
    pad_x, pad_y, end_x, end_y = tile_box
    w = end_x - pad_x
    h = end_y - pad_y

    radius = max(1, int(round(min(w, h) * blur_pct)))
    ksize = 2 * radius + 1
    if ksize % 2 == 0:
        ksize += 1

    img_rgb = extended.convert("RGB")
    arr = np.asarray(img_rgb, dtype=np.float32)
    h_ext, w_ext = arr.shape[:2]

    ys = np.arange(h_ext)
    dy = np.abs(ys - pad_y)
    mask_y = np.clip(1.0 - (dy / float(radius)), 0.0, 1.0)[:, None, None]
    mask_y = mask_y * mask_y * (3.0 - 2.0 * mask_y)

    xs = np.arange(w_ext)
    dx = np.abs(xs - pad_x)
    mask_x = np.clip(1.0 - (dx / float(radius)), 0.0, 1.0)[None, :, None]
    mask_x = mask_x * mask_x * (3.0 - 2.0 * mask_x)

    mask = np.maximum(mask_y, mask_x)

    res = arr.copy()
    prev_jump = float("inf")
    for _ in range(max_passes):
        blurred = cv2.GaussianBlur(res, (ksize, ksize), sigmaX=radius * 0.6, sigmaY=radius * 0.6)
        res = (1.0 - mask) * res + mask * blurred

        top_jump = float(np.mean(np.abs(res[pad_y, :, :] - res[pad_y - 1, :, :])))
        left_jump = float(np.mean(np.abs(res[:, pad_x, :] - res[:, pad_x - 1, :])))
        max_jump = max(top_jump, left_jump)

        if max_jump < 2.0 or abs(prev_jump - max_jump) < 0.2:
            break
        prev_jump = max_jump

    res_uint8 = np.clip(res, 0, 255).astype(np.uint8)
    return Image.fromarray(res_uint8, mode="RGB")


def save_exr(output_path: str, arr: np.ndarray, as_rgb: bool = True) -> str:
    """Save a 32-bit float array as a standard OpenEXR (.exr) file in pure Python.

    Compatible with Unreal Engine 5, Blender, Houdini, Maya, Nuke, Photoshop.
    Channel names are lexicographically sorted per OpenEXR specification.
    """
    import struct

    h, w = arr.shape[:2]
    magic = struct.pack("<I", 0x01312f76)
    version = struct.pack("<I", 2)

    def attr(name: str, type_name: str, value_bytes: bytes) -> bytes:
        return name.encode("ascii") + b"\x00" + type_name.encode("ascii") + b"\x00" + struct.pack("<I", len(value_bytes)) + value_bytes

    if as_rgb:
        ch_b = b"B\x00" + struct.pack("<IB3sII", 2, 0, b"\x00\x00\x00", 1, 1)
        ch_g = b"G\x00" + struct.pack("<IB3sII", 2, 0, b"\x00\x00\x00", 1, 1)
        ch_r = b"R\x00" + struct.pack("<IB3sII", 2, 0, b"\x00\x00\x00", 1, 1)
        channels_attr = attr("channels", "chlist", ch_b + ch_g + ch_r + b"\x00")
    else:
        ch_y = b"Y\x00" + struct.pack("<IB3sII", 2, 0, b"\x00\x00\x00", 1, 1)
        channels_attr = attr("channels", "chlist", ch_y + b"\x00")

    comp_attr = attr("compression", "compression", b"\x00")
    data_win_attr = attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
    disp_win_attr = attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
    line_order_attr = attr("lineOrder", "lineOrder", b"\x00")
    aspect_attr = attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    center_attr = attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    width_attr = attr("screenWindowWidth", "float", struct.pack("<f", 1.0))

    header = magic + version + channels_attr + comp_attr + data_win_attr + disp_win_attr + line_order_attr + aspect_attr + center_attr + width_attr + b"\x00"
    num_channels = 3 if as_rgb else 1
    bytes_per_row = w * 4 * num_channels
    scanline_chunk_size = 4 + 4 + bytes_per_row
    first_offset = len(header) + h * 8
    offsets = [first_offset + i * scanline_chunk_size for i in range(h)]
    offset_table = struct.pack(f"<{h}Q", *offsets)

    float_data = arr.astype("<f4")
    if as_rgb and float_data.ndim == 2:
        b_data = float_data
        g_data = float_data
        r_data = float_data
    elif as_rgb and float_data.ndim == 3:
        r_data = float_data[:, :, 0]
        g_data = float_data[:, :, 1]
        b_data = float_data[:, :, 2]
    else:
        y_data = float_data

    try:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        with open(output_path, "wb") as f:
            f.write(header)
            f.write(offset_table)
            for y in range(h):
                f.write(struct.pack("<ii", y, bytes_per_row))
                if as_rgb:
                    f.write(b_data[y].tobytes())
                    f.write(g_data[y].tobytes())
                    f.write(r_data[y].tobytes())
                else:
                    f.write(y_data[y].tobytes())
        return output_path
    except Exception:
        import tempfile
        dirname, _ = os.path.split(output_path)
        if not dirname:
            dirname = "."
        with tempfile.NamedTemporaryFile(dir=dirname, delete=False, suffix=".exr") as tmp:
            tmp_path = tmp.name
        with open(tmp_path, "wb") as f:
            f.write(header)
            f.write(offset_table)
            for y in range(h):
                f.write(struct.pack("<ii", y, bytes_per_row))
                if as_rgb:
                    f.write(b_data[y].tobytes())
                    f.write(g_data[y].tobytes())
                    f.write(r_data[y].tobytes())
                else:
                    f.write(y_data[y].tobytes())
        try:
            os.replace(tmp_path, output_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        return output_path


def safe_save_image(image: Image.Image, output_path: str) -> str:
    """Safely save a PIL Image or EXR to disk on Windows/OneDrive.

    Handles file lock issues (like OSError: [Errno 22] Invalid argument)
    by removing the destination file if it exists, or writing to a temporary
    file and atomically replacing it.
    """
    if image.mode == "F" and output_path.lower().endswith(".png"):
        output_path = os.path.splitext(output_path)[0] + ".exr"

    if output_path.lower().endswith(".exr"):
        if image.mode in ("RGBA", "LA", "P", "1", "CMYK", "YCbCr"):
            image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32)
        if image.mode in ("I;16", "I"):
            arr = arr / 65535.0
        elif image.mode in ("L", "RGB"):
            arr = arr / 255.0
        return save_exr(output_path, arr, as_rgb=True)

    img_to_save = image

    try:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        img_to_save.save(output_path)
        return output_path
    except Exception:
        import tempfile
        dirname, _ = os.path.split(output_path)
        if not dirname:
            dirname = "."
        suffix = os.path.splitext(output_path)[1] or ".png"
        with tempfile.NamedTemporaryFile(dir=dirname, delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
        img_to_save.save(tmp_path)
        try:
            os.replace(tmp_path, output_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
        return output_path


def save_extended_tile_preview(
    extended: Image.Image,
    tile_box: tuple[int, int, int, int],
    output_path: str,
    border_color: tuple[int, int, int] = (255, 0, 0),
    border_width: int = 2,
) -> str:
    """Save a copy of `extended` with the center tile frame highlighted to `output_path`."""
    from PIL import ImageDraw

    preview = extended.copy()
    draw = ImageDraw.Draw(preview)
    x0, y0, x1, y1 = tile_box
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=border_color, width=border_width)
    return safe_save_image(preview, output_path)


def apply_seamless_strip_copy(
    ehm: np.ndarray | Image.Image,
    orig_size: tuple[int, int],
    padding_pct: float = 0.10,
) -> np.ndarray | Image.Image:
    """Apply horizontal and vertical seamless boundary strip copies on ehm."""
    w, h = orig_size
    pad_x = max(1, int(round(w * padding_pct)))
    pad_y = max(1, int(round(h * padding_pct)))

    corner_y = pad_y
    corner_x = pad_x

    if isinstance(ehm, Image.Image):
        from PIL import ImageDraw

        out_img = ehm.copy()
        strip_h = out_img.crop((0, pad_y, pad_x, pad_y + h))
        out_img.paste(strip_h, (w, pad_y))
        strip_v = out_img.crop((pad_x, 0, pad_x + w, pad_y))
        out_img.paste(strip_v, (pad_x, h))

        draw = ImageDraw.Draw(out_img)
        draw.rectangle([w, pad_y, w + corner_x - 1, pad_y + corner_y - 1], fill=0)
        draw.rectangle([w, h, w + corner_x - 1, h + corner_y - 1], fill=0)
        draw.rectangle([pad_x, h, pad_x + corner_x - 1, h + corner_y - 1], fill=0)
        return out_img

    out = np.copy(ehm)
    source_strip_h = out[pad_y : pad_y + h, 0 : pad_x]
    out[pad_y : pad_y + h, w : w + pad_x] = source_strip_h

    source_strip_v = out[0 : pad_y, pad_x : pad_x + w]
    out[h : h + pad_y, pad_x : pad_x + w] = source_strip_v

    out[pad_y : pad_y + corner_y, w : w + corner_x] = 0.0
    out[h : h + corner_y, w : w + corner_x] = 0.0
    out[h : h + corner_y, pad_x : pad_x + corner_x] = 0.0
    return out


def save_extended_height_preview(
    ehm: np.ndarray | Image.Image,
    output_path: str,
    bit_depth: int = 8,
) -> str:
    """Save the extended height map (ehm) to disk as a PNG preview."""
    if isinstance(ehm, np.ndarray):
        img = to_image(ehm, bit_depth=bit_depth)
    else:
        img = ehm
    return safe_save_image(img, output_path)


def crop_center_tile(
    ehm: np.ndarray | Image.Image,
    orig_size: tuple[int, int],
    padding_pct: float = 0.10,
) -> np.ndarray | Image.Image:
    """Crop the center tile (W x H) from the extended image or array (termed 'raw1').

    For example, with a 100x100 original and 10% padding (120x120 extended):
    - pad_x = 10, pad_y = 10
    - Crops [y: 10:110, x: 10:110], returning a 100x100 array or image.
    """
    w, h = orig_size
    pad_x = max(1, int(round(w * padding_pct)))
    pad_y = max(1, int(round(h * padding_pct)))

    if isinstance(ehm, Image.Image):
        return ehm.crop((pad_x, pad_y, pad_x + w, pad_y + h))

    return np.copy(ehm[pad_y : pad_y + h, pad_x : pad_x + w])


def apply_seamless_linear_mask(
    ehm: np.ndarray,
    orig_size: tuple[int, int],
    padding_pct: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply linear gradient masking f(x)=x on the boundary strips to construct raw1.

    Linear blend behavior:
    - Right strip (x in [w - pad_x, w], y in [pad_y, h - pad_y]):
      From right to left (outer to inner), transparency starts at 0% (patch opaque)
      at outer edge (x = w - 1) and linearly increases to 100% (patch transparent)
      at innermost pixel (x = w - pad_x).
    - Bottom strip (y in [h - pad_y, h], x in [pad_x, w - pad_x]):
      From bottom to top (outer to inner), transparency starts at 0% (patch opaque)
      at outer edge (y = h - 1) and linearly increases to 100% (patch transparent)
      at innermost pixel (y = h - pad_y).
    - Corner squares (top-right, bottom-left, bottom-right) remain black (0.0)
      for the upcoming corner processing step.

    Returns:
    - (ehm_modified, raw1, mask_map)
    """
    w, h = orig_size
    pad_x = max(1, int(round(w * padding_pct)))
    pad_y = max(1, int(round(h * padding_pct)))
    expected_h = h + 2 * pad_y
    expected_w = w + 2 * pad_x

    if ehm.shape[0] != expected_h or ehm.shape[1] != expected_w:
        if ehm.shape[0] == h and ehm.shape[1] == w:
            raw1 = np.copy(ehm)
            mask_map = np.zeros((h, w), dtype=np.float32)
            return ehm, raw1, mask_map
        ehm = cv2.resize(ehm, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

    raw1 = np.copy(ehm[pad_y : pad_y + h, pad_x : pad_x + w])

    patch_v = ehm[pad_y : pad_y + h, 0 : pad_x]
    patch_h = ehm[0 : pad_y, pad_x : pad_x + w]

    corner_y = pad_y
    corner_x = pad_x

    y_start = corner_y
    y_end = h - corner_y
    if pad_x > 1:
        alpha_x = np.linspace(0.0, 1.0, pad_x, dtype=np.float32)[np.newaxis, :]
    else:
        alpha_x = np.array([[1.0]], dtype=np.float32)

    raw1[y_start : y_end, w - pad_x : w] = (
        (1.0 - alpha_x) * raw1[y_start : y_end, w - pad_x : w]
        + alpha_x * patch_v[y_start : y_end, :]
    )

    x_start = corner_x
    x_end = w - corner_x
    if pad_y > 1:
        alpha_y = np.linspace(0.0, 1.0, pad_y, dtype=np.float32)[:, np.newaxis]
    else:
        alpha_y = np.array([[1.0]], dtype=np.float32)

    raw1[h - pad_y : h, x_start : x_end] = (
        (1.0 - alpha_y) * raw1[h - pad_y : h, x_start : x_end]
        + alpha_y * patch_h[:, x_start : x_end]
    )

    orig_tr = np.copy(raw1[0 : corner_y, w - corner_x : w])

    ref_left_tr = ehm[pad_y : pad_y + corner_y, 0 : pad_x]

    ys_idx, xs_idx = np.indices((corner_y, corner_x), dtype=np.float32)
    u = xs_idx / max(1.0, float(corner_x - 1))
    v = ys_idx / max(1.0, float(corner_y - 1))

    overlap_ratio = 0.25
    max_half_width = max(2.0, float(min(corner_y, corner_x) * overlap_ratio))

    tr_signed_dist = (u + v - 1.0) / np.sqrt(2.0)
    dist_to_end1_tr = np.hypot(u - 0.0, v - 1.0)
    dist_to_end2_tr = np.hypot(u - 1.0, v - 0.0)
    taper_ends_tr = np.clip(np.minimum(dist_to_end1_tr, dist_to_end2_tr) / 0.06, 0.0, 1.0)
    taper_border_tr = np.clip(np.minimum(v, 1.0 - u) / 0.05, 0.0, 1.0)
    taper_tr = taper_ends_tr * taper_border_tr
    taper_tr = taper_tr * taper_tr * (3.0 - 2.0 * taper_tr)

    eff_hw_tr = np.maximum(0.5, max_half_width * taper_tr)
    norm_coord_tr = np.clip(tr_signed_dist * min(corner_y, corner_x) / (2.0 * eff_hw_tr) + 0.5, 0.0, 1.0)
    sharp_mask_tr = np.where((u + v) <= 1.0, 0.0, 1.0)
    blend_weight_tr = norm_coord_tr * norm_coord_tr * (3.0 - 2.0 * norm_coord_tr)
    w_weight_tr = np.where(taper_tr <= 1e-4, sharp_mask_tr, blend_weight_tr)

    ref_box_tr = (1.0 - w_weight_tr) * orig_tr + w_weight_tr * ref_left_tr

    mask_tr = np.flipud(create_diagonal_corner_mask((corner_y, corner_x)))
    raw1[0 : corner_y, w - corner_x : w] = (
        (1.0 - mask_tr) * orig_tr + mask_tr * ref_box_tr
    )

    orig_bl = np.copy(raw1[h - corner_y : h, 0 : corner_x])
    ref_top_bl = ehm[0 : pad_y, pad_x : pad_x + corner_x]

    bl_signed_dist = (u + v - 1.0) / np.sqrt(2.0)
    dist_to_end1_bl = np.hypot(u - 0.0, v - 1.0)
    dist_to_end2_bl = np.hypot(u - 1.0, v - 0.0)
    taper_ends_bl = np.clip(np.minimum(dist_to_end1_bl, dist_to_end2_bl) / 0.06, 0.0, 1.0)
    taper_border_bl = np.clip(np.minimum(u, 1.0 - v) / 0.05, 0.0, 1.0)
    taper_bl = taper_ends_bl * taper_border_bl
    taper_bl = taper_bl * taper_bl * (3.0 - 2.0 * taper_bl)

    eff_hw_bl = np.maximum(0.5, max_half_width * taper_bl)
    norm_coord_bl = np.clip(bl_signed_dist * min(corner_y, corner_x) / (2.0 * eff_hw_bl) + 0.5, 0.0, 1.0)
    sharp_mask_bl = np.where((u + v) <= 1.0, 0.0, 1.0)
    blend_weight_bl = norm_coord_bl * norm_coord_bl * (3.0 - 2.0 * norm_coord_bl)
    w_weight_bl = np.where(taper_bl <= 1e-4, sharp_mask_bl, blend_weight_bl)

    ref_box_bl = (1.0 - w_weight_bl) * orig_bl + w_weight_bl * ref_top_bl

    mask_bl = np.fliplr(create_diagonal_corner_mask((corner_y, corner_x)))
    raw1[h - corner_y : h, 0 : corner_x] = (
        (1.0 - mask_bl) * orig_bl + mask_bl * ref_box_bl
    )

    orig_corner = np.copy(raw1[h - corner_y : h, w - corner_x : w])
    ref_bottom_left = ehm[h : h + pad_y, 0 : pad_x]
    ref_top_right = ehm[0 : pad_y, w : w + pad_x]

    br_signed_dist = (u - v) / np.sqrt(2.0)
    dist_to_end1_br = np.hypot(u - 0.0, v - 0.0)
    dist_to_end2_br = np.hypot(u - 1.0, v - 1.0)
    taper_ends_br = np.clip(np.minimum(dist_to_end1_br, dist_to_end2_br) / 0.06, 0.0, 1.0)
    taper_border_br = np.clip(np.minimum(1.0 - u, 1.0 - v) / 0.05, 0.0, 1.0)
    taper_br = taper_ends_br * taper_border_br
    taper_br = taper_br * taper_br * (3.0 - 2.0 * taper_br)

    eff_hw_br = np.maximum(0.5, max_half_width * taper_br)
    norm_coord_br = np.clip(br_signed_dist * min(corner_y, corner_x) / (2.0 * eff_hw_br) + 0.5, 0.0, 1.0)
    sharp_mask_br = np.where(u >= v, 1.0, 0.0)
    blend_weight_br = norm_coord_br * norm_coord_br * (3.0 - 2.0 * norm_coord_br)
    w_weight_br = np.where(taper_br <= 1e-4, sharp_mask_br, blend_weight_br)

    ref_box = w_weight_br * ref_bottom_left + (1.0 - w_weight_br) * ref_top_right

    diag_mask = create_diagonal_corner_mask((corner_y, corner_x))
    raw1[h - corner_y : h, w - corner_x : w] = (
        (1.0 - diag_mask) * orig_corner + diag_mask * ref_box
    )

    mask_map = np.zeros((h, w), dtype=np.float32)
    mask_map[y_start : y_end, w - pad_x : w] = alpha_x
    mask_map[h - pad_y : h, x_start : x_end] = alpha_y
    mask_map[h - corner_y : h, w - corner_x : w] = diag_mask
    mask_map[h - corner_y : h, 0 : corner_x] = mask_bl
    mask_map[0 : corner_y, w - corner_x : w] = mask_tr

    try:
        raw1 = apply_seamless_corner_heal(raw1, box_ratio=0.05)
    except Exception as e:
        print(f"[WARN] Failed to apply seamless corner heal: {e}")

    ehm_out = np.copy(ehm)
    ehm_out[pad_y : pad_y + h, pad_x : pad_x + w] = raw1

    return ehm_out, raw1, mask_map


def save_raw1_mask_debug_preview(
    raw1: np.ndarray,
    mask_map: np.ndarray,
    output_path: str,
    corner_size: tuple[int, int] | None = None,
) -> str:
    """Save raw1 with all applied masks visualized in shades of green,
    with corner triangle diagonals/edges clearly marked in red/cyan.
    """
    h, w = raw1.shape
    base_gray = (np.clip(raw1, 0.0, 1.0) * 255.0).astype(np.float32)

    alpha = np.clip(mask_map, 0.0, 1.0)

    r = base_gray * (1.0 - alpha * 0.85)
    g = np.clip(base_gray * (1.0 - alpha * 0.2) + alpha * 255.0 * 0.7, 0.0, 255.0)
    b = base_gray * (1.0 - alpha * 0.85)

    cy_len, cx_len = corner_size if corner_size is not None else (int(round(h * 0.10)), int(round(w * 0.10)))
    if cy_len > 1 and cx_len > 1:
        y0, x0 = h - cy_len, w - cx_len
        cv2.line(r, (x0, y0), (x0 + cx_len - 1, y0 + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0, y0), (x0 + cx_len - 1, y0 + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0, y0), (x0 + cx_len - 1, y0 + cy_len - 1), 30.0, thickness=1)
        cv2.line(r, (x0 + cx_len - 1, y0), (x0, y0 + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0 + cx_len - 1, y0), (x0, y0 + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0 + cx_len - 1, y0), (x0, y0 + cy_len - 1), 30.0, thickness=1)

        y0_bl, x0_bl = h - cy_len, 0
        cv2.line(r, (x0_bl, y0_bl), (x0_bl + cx_len - 1, y0_bl + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_bl, y0_bl), (x0_bl + cx_len - 1, y0_bl + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_bl, y0_bl), (x0_bl + cx_len - 1, y0_bl + cy_len - 1), 30.0, thickness=1)
        cv2.line(r, (x0_bl + cx_len - 1, y0_bl), (x0_bl, y0_bl + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_bl + cx_len - 1, y0_bl), (x0_bl, y0_bl + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_bl + cx_len - 1, y0_bl), (x0_bl, y0_bl + cy_len - 1), 30.0, thickness=1)

        y0_tr, x0_tr = 0, w - cx_len
        cv2.line(r, (x0_tr, y0_tr), (x0_tr + cx_len - 1, y0_tr + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_tr, y0_tr), (x0_tr + cx_len - 1, y0_tr + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_tr, y0_tr), (x0_tr + cx_len - 1, y0_tr + cy_len - 1), 30.0, thickness=1)
        cv2.line(r, (x0_tr + cx_len - 1, y0_tr), (x0_tr, y0_tr + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_tr + cx_len - 1, y0_tr), (x0_tr, y0_tr + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_tr + cx_len - 1, y0_tr), (x0_tr, y0_tr + cy_len - 1), 30.0, thickness=1)

    rgb = np.stack([r, g, b], axis=-1).round().astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    return safe_save_image(img, output_path)


def save_mask_debug_preview(
    mask_map: np.ndarray,
    output_path: str,
    corner_size: tuple[int, int] | None = None,
) -> str:
    """Save mask map as green shades on pure black background,
    with corner triangle diagonals/edges clearly marked in bright red.
    """
    h, w = mask_map.shape
    alpha = np.clip(mask_map, 0.0, 1.0)

    r = np.zeros_like(alpha, dtype=np.float32)
    g = (alpha * 255.0).astype(np.float32)
    b = np.zeros_like(alpha, dtype=np.float32)

    cy_len, cx_len = corner_size if corner_size is not None else (int(round(h * 0.10)), int(round(w * 0.10)))
    if cy_len > 1 and cx_len > 1:
        y0, x0 = h - cy_len, w - cx_len
        cv2.line(r, (x0, y0), (x0 + cx_len - 1, y0 + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0, y0), (x0 + cx_len - 1, y0 + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0, y0), (x0 + cx_len - 1, y0 + cy_len - 1), 30.0, thickness=1)
        cv2.line(r, (x0 + cx_len - 1, y0), (x0, y0 + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0 + cx_len - 1, y0), (x0, y0 + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0 + cx_len - 1, y0), (x0, y0 + cy_len - 1), 30.0, thickness=1)

        y0_bl, x0_bl = h - cy_len, 0
        cv2.line(r, (x0_bl, y0_bl), (x0_bl + cx_len - 1, y0_bl + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_bl, y0_bl), (x0_bl + cx_len - 1, y0_bl + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_bl, y0_bl), (x0_bl + cx_len - 1, y0_bl + cy_len - 1), 30.0, thickness=1)
        cv2.line(r, (x0_bl + cx_len - 1, y0_bl), (x0_bl, y0_bl + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_bl + cx_len - 1, y0_bl), (x0_bl, y0_bl + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_bl + cx_len - 1, y0_bl), (x0_bl, y0_bl + cy_len - 1), 30.0, thickness=1)

        y0_tr, x0_tr = 0, w - cx_len
        cv2.line(r, (x0_tr, y0_tr), (x0_tr + cx_len - 1, y0_tr + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_tr, y0_tr), (x0_tr + cx_len - 1, y0_tr + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_tr, y0_tr), (x0_tr + cx_len - 1, y0_tr + cy_len - 1), 30.0, thickness=1)
        cv2.line(r, (x0_tr + cx_len - 1, y0_tr), (x0_tr, y0_tr + cy_len - 1), 255.0, thickness=1)
        cv2.line(g, (x0_tr + cx_len - 1, y0_tr), (x0_tr, y0_tr + cy_len - 1), 30.0, thickness=1)
        cv2.line(b, (x0_tr + cx_len - 1, y0_tr), (x0_tr, y0_tr + cy_len - 1), 30.0, thickness=1)

    rgb = np.stack([r, g, b], axis=-1).round().astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    return safe_save_image(img, output_path)


save_corner_mask_debug_preview = save_mask_debug_preview


def create_diagonal_corner_mask(size: int | tuple[int, int]) -> np.ndarray:
    """Create a 2D corner mask using the max(x, y) formulation.

    For example for N=6:
    [[0, 1, 2, 3, 4, 5],
     [1, 1, 2, 3, 4, 5],
     [2, 2, 2, 3, 4, 5],
     [3, 3, 3, 3, 4, 5],
     [4, 4, 4, 4, 4, 5],
     [5, 5, 5, 5, 5, 5]] / 5

    This seamlessly aligns with the right strip (varying along x)
    and the bottom strip (varying along y) with zero discontinuity at the edges!
    """
    if isinstance(size, int):
        h, w = size, size
    else:
        h, w = size

    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)

    ys = (np.arange(h, dtype=np.float32) / (h - 1))[:, np.newaxis]
    xs = (np.arange(w, dtype=np.float32) / (w - 1))[np.newaxis, :]
    mask = xs + ys - (xs * ys)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def build_seamless_corner_debug_grid(
    image_or_array: np.ndarray | Image.Image,
    box_ratio: float = 0.05,
) -> np.ndarray:
    """Extract 4 corner boxes from the image and assemble them in a 2x2 grid: [[4, 3], [2, 1]].

    Box numbering:
      1: Sol ust (Top-Left)
      2: Sag ust (Top-Right)
      3: Sol alt (Bottom-Left)
      4: Sag alt (Bottom-Right)

    Grid arrangement:
      Row 0: [Box 4, Box 3]
      Row 1: [Box 2, Box 1]
    """
    if isinstance(image_or_array, Image.Image):
        arr = np.asarray(image_or_array)
    else:
        arr = np.asarray(image_or_array)

    h, w = arr.shape[:2]
    box_h = max(1, int(round(h * box_ratio)))
    box_w = max(1, int(round(w * box_ratio)))

    box1 = arr[0:box_h, 0:box_w]
    box2 = arr[0:box_h, w - box_w : w]
    box3 = arr[h - box_h : h, 0:box_w]
    box4 = arr[h - box_h : h, w - box_w : w]

    row0 = np.concatenate([box4, box3], axis=1)
    row1 = np.concatenate([box2, box1], axis=1)
    return np.concatenate([row0, row1], axis=0)


def blur_seamless_corner_grid(
    grid: np.ndarray,
    blur_units: float | None = None,
    max_passes: int = 10,
) -> np.ndarray:
    """Apply a smooth blur of specified units (default: half grid size, e.g. 5 units on a 10x10 grid)
    centered at the middle intersection point of the 2x2 corner grid until the central discontinuity ends.
    """
    arr = grid.astype(np.float32).copy()
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2

    radius = float(blur_units) if blur_units is not None else float(min(cy, cx))
    radius = max(1.0, radius)

    ksize = int(round(radius)) * 2 + 1
    if ksize % 2 == 0:
        ksize += 1

    ys = np.arange(h, dtype=np.float32) - (cy - 0.5)
    xs = np.arange(w, dtype=np.float32) - (cx - 0.5)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)

    t = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    mask = t * t * (3.0 - 2.0 * t)
    if arr.ndim == 3:
        mask = mask[..., np.newaxis]

    res = arr.copy()
    prev_discontinuity = float("inf")
    sigma = max(0.5, radius * 0.6)

    for _ in range(max_passes):
        blurred = cv2.GaussianBlur(res, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        res = (1.0 - mask) * res + mask * blurred

        jump_h = np.abs(res[cy, :] - res[cy - 1, :])
        jump_v = np.abs(res[:, cx] - res[:, cx - 1])
        c_jump = max(
            float(np.max(jump_h[max(0, cx - 1) : min(w, cx + 1)])),
            float(np.max(jump_v[max(0, cy - 1) : min(h, cy + 1)])),
        )

        if abs(prev_discontinuity - c_jump) < 0.002 or c_jump < 0.01:
            break
        prev_discontinuity = c_jump

    return res


def save_seamless_blurred_debug_image(
    image_or_array: np.ndarray | Image.Image,
    output_path: str | None = None,
) -> str:
    """Save the center-blurred corner debug image to desktop."""
    if isinstance(image_or_array, Image.Image):
        grid = np.asarray(image_or_array)
    else:
        grid = np.asarray(image_or_array)

    if np.issubdtype(grid.dtype, np.floating):
        if float(grid.max()) <= 1.0 + 1e-5:
            grid_uint8 = np.clip(grid * 255.0, 0.0, 255.0).round().astype(np.uint8)
        else:
            grid_uint8 = np.clip(grid, 0.0, 255.0).round().astype(np.uint8)
    else:
        grid_uint8 = np.clip(grid, 0, 255).astype(np.uint8)

    if grid_uint8.ndim == 2:
        img = Image.fromarray(grid_uint8, mode="L")
    elif grid_uint8.ndim == 3 and grid_uint8.shape[2] == 3:
        img = Image.fromarray(grid_uint8, mode="RGB")
    elif grid_uint8.ndim == 3 and grid_uint8.shape[2] == 4:
        img = Image.fromarray(grid_uint8, mode="RGBA")
    else:
        img = Image.fromarray(grid_uint8)

    if output_path is None:
        desktop = get_desktop_dir()
        output_path = os.path.join(desktop, "seamless_blurred_debug.png")

    saved_path = safe_save_image(img, output_path)
    if output_path.endswith("seamless_blurred_debug.png"):
        alt_path = os.path.join(os.path.dirname(output_path), "seamless_corner_blurred_debug.png")
        try:
            safe_save_image(img, alt_path)
        except Exception:
            pass
    return saved_path


def save_seamless_corner_debug_image(
    image_or_array: np.ndarray | Image.Image,
    output_path: str | None = None,
    box_ratio: float = 0.05,
) -> str:
    """Save the [[4, 3], [2, 1]] assembled corner debug image and its center-blurred counterpart to desktop."""
    grid = build_seamless_corner_debug_grid(image_or_array, box_ratio=box_ratio)

    if np.issubdtype(grid.dtype, np.floating):
        if float(grid.max()) <= 1.0 + 1e-5:
            grid_uint8 = np.clip(grid * 255.0, 0.0, 255.0).round().astype(np.uint8)
        else:
            grid_uint8 = np.clip(grid, 0.0, 255.0).round().astype(np.uint8)
    else:
        grid_uint8 = np.clip(grid, 0, 255).astype(np.uint8)

    if grid_uint8.ndim == 2:
        img = Image.fromarray(grid_uint8, mode="L")
    elif grid_uint8.ndim == 3 and grid_uint8.shape[2] == 3:
        img = Image.fromarray(grid_uint8, mode="RGB")
    elif grid_uint8.ndim == 3 and grid_uint8.shape[2] == 4:
        img = Image.fromarray(grid_uint8, mode="RGBA")
    else:
        img = Image.fromarray(grid_uint8)

    if output_path is None:
        desktop = get_desktop_dir()
        output_path = os.path.join(desktop, "seamless_debug.png")

    saved_path = safe_save_image(img, output_path)
    if output_path.endswith("seamless_debug.png"):
        alt_path = os.path.join(os.path.dirname(output_path), "seamless_corner_debug.png")
        try:
            safe_save_image(img, alt_path)
        except Exception:
            pass

    try:
        blurred_grid = blur_seamless_corner_grid(grid)
        save_seamless_blurred_debug_image(blurred_grid)
    except Exception as e:
        print(f"[WARN] Failed to save seamless blurred debug image: {e}")

    return saved_path


def apply_seamless_corner_heal(
    image_or_array: np.ndarray | Image.Image,
    box_ratio: float = 0.05,
    blur_units: float | None = None,
    max_passes: int = 10,
    save_debug: bool = False,
) -> np.ndarray:
    """Heal the 4 corners of the main seamless result using the center-blurred 2x2 corner grid.

    Grid mapping [[4, 3], [2, 1]]:
      - Region 4 (Top-Left of grid) -> Sag Alt (Bottom-Right) of result
      - Region 3 (Top-Right of grid) -> Sol Alt (Bottom-Left) of result
      - Region 2 (Bottom-Left of grid) -> Sag Ust (Top-Right) of result
      - Region 1 (Bottom-Right of grid) -> Sol Ust (Top-Left) of result
    """
    is_pil = isinstance(image_or_array, Image.Image)
    if is_pil:
        arr = np.asarray(image_or_array).copy()
    else:
        arr = np.asarray(image_or_array).copy()

    h, w = arr.shape[:2]
    box_h = max(1, int(round(h * box_ratio)))
    box_w = max(1, int(round(w * box_ratio)))

    grid = build_seamless_corner_debug_grid(arr, box_ratio=box_ratio)

    blurred_grid = blur_seamless_corner_grid(grid, blur_units=blur_units, max_passes=max_passes)

    if save_debug:
        try:
            save_seamless_corner_debug_image(arr, box_ratio=box_ratio)
            save_seamless_blurred_debug_image(blurred_grid)
        except Exception as e:
            print(f"[WARN] Failed to save seamless corner debug files: {e}")

    region4 = blurred_grid[0 : box_h, 0 : box_w]
    region3 = blurred_grid[0 : box_h, box_w : 2 * box_w]
    region2 = blurred_grid[box_h : 2 * box_h, 0 : box_w]
    region1 = blurred_grid[box_h : 2 * box_h, box_w : 2 * box_w]

    if not np.issubdtype(arr.dtype, np.floating):
        region4 = np.clip(region4, 0, 255).round().astype(arr.dtype)
        region3 = np.clip(region3, 0, 255).round().astype(arr.dtype)
        region2 = np.clip(region2, 0, 255).round().astype(arr.dtype)
        region1 = np.clip(region1, 0, 255).round().astype(arr.dtype)
    else:
        region4 = region4.astype(arr.dtype)
        region3 = region3.astype(arr.dtype)
        region2 = region2.astype(arr.dtype)
        region1 = region1.astype(arr.dtype)

    out_arr = arr.copy()
    out_arr[h - box_h : h, w - box_w : w] = region4
    out_arr[h - box_h : h, 0 : box_w] = region3
    out_arr[0 : box_h, w - box_w : w] = region2
    out_arr[0 : box_h, 0 : box_w] = region1

    if is_pil:
        return Image.fromarray(out_arr)
    return out_arr
