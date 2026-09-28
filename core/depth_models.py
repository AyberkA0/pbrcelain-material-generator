"""Depth-estimation backends used to turn a material photo into a height map.

Depth Anything V2 is loaded through Hugging Face `transformers` with the
generic `AutoModelForDepthEstimation` API; Marigold goes through the
`diffusers` MarigoldDepthPipeline. Adding a new checkpoint of an existing
family is just a new entry in MODEL_CATALOG.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image

MODEL_CATALOG: dict[str, dict[str, str]] = {
    "Depth Anything V2": {
        "Small (fast)": "depth-anything/Depth-Anything-V2-Small-hf",
        "Base": "depth-anything/Depth-Anything-V2-Base-hf",
        "Large (accurate)": "depth-anything/Depth-Anything-V2-Large-hf",
    },
    "Marigold": {
        "LCM (Fast & Ultra Detail)": "prs-eth/marigold-depth-lcm-v1-0",
        "v1.1 (Standard Quality)": "prs-eth/marigold-depth-v1-1",
    },
}

FAMILY_NEAR_IS_LARGE: dict[str, bool] = {
    "Depth Anything V2": True,
    "Marigold": True,
}

ProgressCB = Optional[Callable[[str], None]]
IsCancelledCB = Optional[Callable[[], bool]]


class InferenceCancelled(Exception):
    """Raised (and caught by the caller) when a user-requested cancel lands mid-run."""


@dataclass
class DepthResult:
    depth: np.ndarray
    near_is_large: bool
    guide: np.ndarray
    seamless_orig_size: Optional[tuple[int, int]] = None


class DepthEstimator:
    """Loads one depth-estimation checkpoint and runs inference on PIL images."""

    def __init__(self, family: str, variant: str, device: str = "auto"):
        if family not in MODEL_CATALOG or variant not in MODEL_CATALOG[family]:
            raise ValueError(f"Unknown model {family}/{variant}")
        self.family = family
        self.variant = variant
        self.checkpoint = MODEL_CATALOG[family][variant]
        self.requested_device = device
        self.device = device
        self.processor = None
        self.model = None
        self.pipe = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if device == "auto":
            if has_cuda:
                return "cuda"
            if has_mps:
                return "mps"
            return "cpu"
        if device == "cuda" and not has_cuda:
            import sys

            build = torch.__version__
            hint = (
                "The installed PyTorch is a CPU-only build"
                if torch.version.cuda is None
                else "PyTorch was built with CUDA, but no usable NVIDIA GPU/driver was found"
            )
            raise RuntimeError(
                f"GPU (CUDA) was selected but CUDA is not available.\n\n"
                f"{hint} (torch {build}).\n"
                f"Python interpreter: {sys.executable}\n\n"
                "Make sure the app is started from the project's venv "
                "(venv\\Scripts\\python.exe main.py) and that torch was installed with "
                "the CUDA index URL (see README), or select 'Auto' / 'CPU'."
            )
        if device == "mps" and not has_mps:
            raise RuntimeError("Apple Silicon (MPS) was selected but MPS is not available. Select 'Auto' or 'CPU'.")
        return device

    def key(self) -> str:
        """Cache key identifying this exact loaded configuration."""
        return f"{self.checkpoint}|{self.device}"

    def load(self, progress_cb: ProgressCB = None) -> None:
        import torch

        self.device = self._resolve_device(self.requested_device)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if progress_cb:
            progress_cb(f"Loading {self.family} ({self.variant}) on {self.device.upper()}…\n{self.checkpoint}")

        if self.family == "Marigold":
            from diffusers import MarigoldDepthPipeline

            pipe_dtype = torch.float16 if self.device.startswith("cuda") or self.device == "mps" else torch.float32
            pipe_kwargs = {"dtype": pipe_dtype}
            if pipe_dtype == torch.float16:
                pipe_kwargs["variant"] = "fp16"
            print(f"[DepthEstimator] Loading Marigold {self.checkpoint} on {self.device} with {pipe_kwargs}")
            self.pipe = MarigoldDepthPipeline.from_pretrained(self.checkpoint, **pipe_kwargs)
            self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.model = self.pipe.unet
            self.processor = None
            actual = next(self.pipe.unet.parameters()).device.type
            if actual != self.device.split(":")[0]:
                raise RuntimeError(
                    f"Marigold weights ended up on '{actual}' instead of '{self.device}'."
                )
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            kwargs = {}
            if self.device.startswith("cuda") or self.device == "mps":
                kwargs["dtype"] = torch.float16

            print(f"[DepthEstimator] Loading {self.checkpoint} on {self.device} with options: {kwargs}")
            self.model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint, **kwargs)
            self.model.to(self.device)
            self.model.eval()
            self.pipe = None

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.max_memory_reserved() / (1024**2)
            msg = f"Ready on {self.device.upper()} (VRAM: {allocated:.0f}MB allocated / {reserved:.0f}MB reserved)"
            print(f"[DepthEstimator] {msg}")
        else:
            msg = f"Ready on {self.device.upper()}."
            print(f"[DepthEstimator] {msg}")

        if progress_cb:
            progress_cb(msg)

    def is_loaded(self) -> bool:
        return self.pipe is not None or self.model is not None

    def infer(self, image: Image.Image, progress_cb: ProgressCB = None) -> DepthResult:
        import torch

        if not self.is_loaded():
            raise RuntimeError("Model is not loaded yet")

        t_start = time.time()
        device_str = self.device.upper()
        vram_info = ""
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            alloc_mb = torch.cuda.memory_allocated() / (1024**2)
            vram_info = f" | VRAM: {alloc_mb:.0f}MB"

        rgb = image.convert("RGB")

        if self.family == "Marigold":
            if self.pipe is None:
                raise RuntimeError("Marigold pipeline is not loaded yet")
            status = f"Running Marigold diffusion inference [{device_str}{vram_info}]…"
            if progress_cb:
                progress_cb(status)
            print(f"[DepthEstimator] {status}")

            steps = 4 if "lcm" in self.checkpoint.lower() else 10
            t_inf_start = time.time()
            try:
                res = self.pipe(
                    rgb,
                    num_inference_steps=steps,
                    processing_resolution=768,
                    match_input_resolution=True,
                    resample_method_output="bicubic",
                    batch_size=1,
                )
                pred = np.squeeze(res.prediction)
                depth = (1.0 - pred).astype(np.float32)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"Out of GPU memory during {self.family} inference. "
                        "Close other GPU applications or use a smaller resolution image."
                    ) from exc
                raise

            if not np.all(np.isfinite(depth)):
                depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)

            guide = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

            elapsed = time.time() - t_inf_start
            total_time = time.time() - t_start
            print(f"[DepthEstimator] {self.family} inference finished in {elapsed:.2f}s (total: {total_time:.2f}s)")
            return DepthResult(
                depth=depth,
                near_is_large=True,
                guide=guide,
            )

        status = f"Preparing image for {self.family} [{device_str}{vram_info}]…"
        if progress_cb:
            progress_cb(status)
        print(f"[DepthEstimator] {status}")

        inputs = self.processor(images=rgb, return_tensors="pt")

        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None and self.device != "cpu":
            inputs = {
                k: v.to(self.device, dtype=model_dtype) if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        status = f"Running {self.family} inference [{device_str}{vram_info}]…"
        if progress_cb:
            progress_cb(status)
        print(f"[DepthEstimator] {status}")

        t_inf_start = time.time()
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predicted = outputs.predicted_depth
                if predicted.ndim == 2:
                    predicted = predicted.unsqueeze(0)
                resized = torch.nn.functional.interpolate(
                    predicted.unsqueeze(1).float(),
                    size=rgb.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1).squeeze(0)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Out of GPU memory during {self.family} inference. "
                    "Close other GPU applications or use a smaller resolution image."
                ) from exc
            raise

        depth = resized.detach().cpu().numpy().astype(np.float32)
        guide = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        elapsed = time.time() - t_inf_start
        total_time = time.time() - t_start
        print(f"[DepthEstimator] {self.family} inference finished in {elapsed:.2f}s (total: {total_time:.2f}s)")

        return DepthResult(
            depth=depth,
            near_is_large=FAMILY_NEAR_IS_LARGE.get(self.family, True),
            guide=guide,
        )

    def infer_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Run several same-purpose images through a single forward pass.

        Batching amortizes per-call overhead across images, which on a GPU
        can be significantly faster than looping one-at-a-time (as
        `infer_tiled` otherwise would for every chunk). Images may differ in
        size - each is resized back to its own size individually after the
        shared forward pass. Returns raw depth arrays (no guide/near_is_large,
        since chunk tiles don't need those - only the full-image `infer` call
        does).
        """
        import torch

        if not self.is_loaded():
            raise RuntimeError("Model is not loaded yet")
        if not images:
            return []

        if self.family == "Marigold":
            if self.pipe is None:
                raise RuntimeError("Marigold pipeline is not loaded yet")
            steps = 4 if "lcm" in self.checkpoint.lower() else 10
            rgbs = [img.convert("RGB") for img in images]
            try:
                res = self.pipe(
                    rgbs,
                    num_inference_steps=steps,
                    processing_resolution=768,
                    match_input_resolution=True,
                    resample_method_output="bicubic",
                    batch_size=min(len(rgbs), 4),
                )
                preds = res.prediction
                results: list[np.ndarray] = []
                for p in preds:
                    p_sq = np.squeeze(p)
                    d = (1.0 - p_sq).astype(np.float32)
                    if not np.all(np.isfinite(d)):
                        d = np.nan_to_num(d, nan=0.5, posinf=1.0, neginf=0.0)
                    results.append(d)
                return results
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"Out of GPU memory processing a batch of {len(images)} chunk(s). "
                        "Lower the chunk batch size (or chunk size) and try again."
                    ) from exc
                raise

        rgbs = [img.convert("RGB") for img in images]
        inputs = self.processor(images=rgbs, return_tensors="pt")

        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None and self.device != "cpu":
            inputs = {
                k: v.to(self.device, dtype=model_dtype) if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predicted = outputs.predicted_depth
                if predicted.ndim == 2:
                    predicted = predicted.unsqueeze(0)
                results: list[np.ndarray] = []
                for i, rgb in enumerate(rgbs):
                    resized = torch.nn.functional.interpolate(
                        predicted[i : i + 1].unsqueeze(1).float(),
                        size=rgb.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze(1).squeeze(0)
                    results.append(resized.detach().cpu().numpy().astype(np.float32))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"Out of GPU memory processing a batch of {len(images)} chunk(s). "
                    "Lower the chunk batch size (or chunk size) and try again."
                ) from exc
            raise
        return results

    def infer_tiled(
        self,
        image: Image.Image,
        tile_size: int = 768,
        overlap_pct: float = 25.0,
        progress_cb: ProgressCB = None,
        partial_cb: Optional[Callable[[np.ndarray], None]] = None,
        is_cancelled: IsCancelledCB = None,
        batch_size: int = 1,
    ) -> "DepthResult":
        """High-cost, high-detail alternative to `infer`.

        A single forward pass squeezes the whole photo into the model's fixed
        (small) input resolution, which is why fine per-object detail gets
        smeared. This instead:
          1. runs one full-image pass to get a trustworthy large-scale
             ("global") relative depth field,
          2. runs the model again on many overlapping crops, each much closer
             to the model's native resolution so it can resolve fine detail,
          3. calibrates each crop's arbitrary relative scale/offset to match
             the global depth in that same region (least-squares fit),
          4. blends the calibrated crops back together with linear cross-fade
             feathering in the overlaps.

        `partial_cb`, if given, is called periodically with the best
        composite depth built so far (unprocessed regions fall back to the
        global pass), so a caller can show a live-refining preview.
        """
        return infer_tiled(
            self,
            image,
            tile_size=tile_size,
            overlap_pct=overlap_pct,
            progress_cb=progress_cb,
            partial_cb=partial_cb,
            is_cancelled=is_cancelled,
            batch_size=batch_size,
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.pipe = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass


def _tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """Top-left offsets of tiles covering [0, total) with a flush final tile."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _calibrate_to_reference(tile_depth: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Least-squares scale+offset fit of `tile_depth` onto `reference`.

    Relative depth models have an arbitrary per-image scale/shift, so a tile
    run independently can't be pasted directly next to the global estimate
    or its neighbours - it must first be recalibrated to agree with the
    trusted low-frequency global depth covering the same pixels.
    """
    x = tile_depth.ravel().astype(np.float64)
    y = reference.ravel().astype(np.float64)
    var_x = x.var()
    if var_x < 1e-8:
        scale, offset = 1.0, float(y.mean() - x.mean())
    else:
        design = np.stack([x, np.ones_like(x)], axis=1)
        (scale, offset), *_ = np.linalg.lstsq(design, y, rcond=None)
        if scale <= 0.0:
            scale, offset = 1.0, float(y.mean() - x.mean())
    return tile_depth.astype(np.float64) * scale + offset


def _axis_weights(length: int, taper_start: int, taper_end: int) -> np.ndarray:
    """1D blend weights: 1.0 in the core, linearly ramping in shared overlaps.

    `taper_start`/`taper_end` are 0 on edges that touch the image border
    (nothing to blend with there) and the overlap width on edges shared with
    a neighboring tile. Matching ramps on both sides of a shared overlap sum
    to exactly 1 everywhere (a linear cross-fade partition of unity).
    """
    w = np.ones(length, dtype=np.float64)
    if taper_start > 0:
        w[:taper_start] = np.arange(1, taper_start + 1) / (taper_start + 1)
    if taper_end > 0:
        ramp = np.arange(1, taper_end + 1) / (taper_end + 1)
        w[length - taper_end :] = ramp[::-1]
    return w


def infer_tiled(
    estimator: DepthEstimator,
    image: Image.Image,
    tile_size: int = 768,
    overlap_pct: float = 25.0,
    progress_cb: ProgressCB = None,
    partial_cb: Optional[Callable[[np.ndarray], None]] = None,
    partial_min_interval: float = 0.25,
    is_cancelled: IsCancelledCB = None,
    batch_size: int = 1,
) -> DepthResult:
    rgb = image.convert("RGB")
    w, h = rgb.size
    tile_size = max(64, min(tile_size, w, h))
    overlap_px = max(1, int(round(tile_size * overlap_pct / 100.0)))
    stride = max(1, tile_size - overlap_px)
    batch_size = max(1, batch_size)

    if progress_cb:
        progress_cb("Chunk estimation: running full-image reference pass…")
    global_result = estimator.infer(rgb, progress_cb=progress_cb)
    global_depth = global_result.depth

    if is_cancelled and is_cancelled():
        raise InferenceCancelled()

    xs = _tile_starts(w, tile_size, stride)
    ys = _tile_starts(h, tile_size, stride)
    n_cols, n_rows = len(xs), len(ys)
    positions = [(row, col, y0, x0) for row, y0 in enumerate(ys) for col, x0 in enumerate(xs)]
    total_tiles = len(positions)
    last_partial_emit = 0.0

    accum = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    tile_idx = 0
    for batch_start in range(0, total_tiles, batch_size):
        if is_cancelled and is_cancelled():
            raise InferenceCancelled()

        batch_positions = positions[batch_start : batch_start + batch_size]
        crops = [rgb.crop((x0, y0, x0 + tile_size, y0 + tile_size)) for (_, _, y0, x0) in batch_positions]
        batch_depths = estimator.infer_batch(crops)

        for (row, col, y0, x0), tile_depth in zip(batch_positions, batch_depths):
            tile_idx += 1
            x1, y1 = x0 + tile_size, y0 + tile_size
            if progress_cb:
                progress_cb(f"Chunk estimation: tile {tile_idx}/{total_tiles}…")

            calibrated = _calibrate_to_reference(tile_depth, global_depth[y0:y1, x0:x1])

            wx = _axis_weights(x1 - x0, overlap_px if col > 0 else 0, overlap_px if col < n_cols - 1 else 0)
            wy = _axis_weights(y1 - y0, overlap_px if row > 0 else 0, overlap_px if row < n_rows - 1 else 0)
            weight = np.outer(wy, wx)

            accum[y0:y1, x0:x1] += calibrated * weight
            weight_sum[y0:y1, x0:x1] += weight

            if partial_cb is not None:
                now = time.monotonic()
                is_last_tile = tile_idx == total_tiles
                if is_last_tile or (now - last_partial_emit) >= partial_min_interval:
                    last_partial_emit = now
                    covered = weight_sum > 1e-6
                    partial = np.where(covered, accum / np.maximum(weight_sum, 1e-8), 0.0)
                    partial_cb(partial.astype(np.float32))

    composite = (accum / np.maximum(weight_sum, 1e-8)).astype(np.float32)
    return DepthResult(depth=composite, near_is_large=global_result.near_is_large, guide=global_result.guide)
