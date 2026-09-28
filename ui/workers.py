"""Background QThread workers so model loading / inference never block the UI."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PIL import Image

from core.depth_models import DepthEstimator, InferenceCancelled
from core.height_map import apply_seamless_linear_mask, build_height_map, downsample_for_preview
from core.height_map import to_image as height_map_to_image
from core.normal_map import generate_normal_map
from core.normal_map import to_image as normal_map_to_image
from core.roughness_map import generate_roughness_map
from core.roughness_map import to_image as roughness_map_to_image


class ModelLoadWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, estimator: DepthEstimator, parent=None):
        super().__init__(parent)
        self.estimator = estimator

    def run(self) -> None:
        try:
            self.estimator.load(progress_cb=self.progress.emit)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """Runs depth estimation for the project's single current Height source image."""

    result_ready = pyqtSignal(object)
    tile_progress = pyqtSignal(str)
    tile_preview = pyqtSignal(object, object)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        estimator: DepthEstimator,
        image: Image.Image,
        parent=None,
        tiled: bool = False,
        tile_size: int = 768,
        overlap_pct: float = 25.0,
        batch_size: int = 1,
    ):
        super().__init__(parent)
        self.estimator = estimator
        self.image = image
        self.tiled = tiled
        self.tile_size = tile_size
        self.overlap_pct = overlap_pct
        self.batch_size = batch_size

    def run(self) -> None:
        try:
            if self.tiled:
                guide = np.asarray(self.image.convert("L"), dtype=np.float32) / 255.0
                result = self.estimator.infer_tiled(
                    self.image,
                    tile_size=self.tile_size,
                    overlap_pct=self.overlap_pct,
                    progress_cb=self.tile_progress.emit,
                    partial_cb=lambda depth: self.tile_preview.emit(depth, guide),
                    is_cancelled=self.isInterruptionRequested,
                    batch_size=self.batch_size,
                )
            else:
                result = self.estimator.infer(self.image, progress_cb=self.tile_progress.emit)
            self.result_ready.emit(result)
            self.finished_ok.emit()
        except InferenceCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class PropertiesPreviewWorker(QThread):
    """Build lightweight, disposable map previews away from the GUI thread.

    Property edits must remain responsive even when the source texture is 4K
    or larger.  This worker always operates on a max-1024px representation;
    the full-resolution pipeline is deliberately reserved for Save.
    """

    result_ready = pyqtSignal(int, str, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        revision: int,
        map_type: str,
        *,
        height_depth: np.ndarray | None = None,
        height_guide: np.ndarray | None = None,
        height_seamless_orig_size: tuple[int, int] | None = None,
        height_options=None,
        height_image: Image.Image | None = None,
        normal_image: Image.Image | None = None,
        normal_options=None,
        roughness_options=None,
        albedo_image: Image.Image | None = None,
        albedo_shift_rules: list | None = None,
        albedo_options=None,
        ao_options=None,
        make_normal: bool = False,
        make_roughness: bool = False,
        make_ao: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.revision = revision
        self.map_type = map_type
        self.height_depth = height_depth
        self.height_guide = height_guide
        self.height_seamless_orig_size = height_seamless_orig_size
        self.height_options = height_options
        self.height_image = height_image
        self.normal_image = normal_image
        self.normal_options = normal_options
        self.roughness_options = roughness_options
        self.albedo_image = albedo_image
        self.albedo_shift_rules = albedo_shift_rules or []
        self.albedo_options = albedo_options
        self.ao_options = ao_options
        self.make_normal = make_normal
        self.make_roughness = make_roughness
        self.make_ao = make_ao

    @staticmethod
    def _image_array(image: Image.Image, mode: str) -> np.ndarray:
        """Decode an image and limit it before any expensive local filters."""
        preview = image.convert(mode)
        preview.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return np.asarray(preview, dtype=np.float32) / 255.0

    @staticmethod
    def _height_array(image: Image.Image) -> np.ndarray:
        """Preserve 16-bit and 32-bit float Height values while producing a compact preview."""
        if image.mode == "I;16":
            return downsample_for_preview(np.asarray(image, dtype=np.float32) / 65535.0)
        if image.mode == "F":
            return downsample_for_preview(np.asarray(image, dtype=np.float32).clip(0.0, 1.0))
        return PropertiesPreviewWorker._image_array(image, "L")

    def run(self) -> None:
        try:
            images: dict[str, Image.Image] = {}
            normal_arr = None

            if self.map_type == "albedo":
                if self.albedo_image is None:
                    return
                from core.albedo_adjust import apply_albedo_adjustments
                preview_alb = self.albedo_image.copy()
                preview_alb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                adjusted = apply_albedo_adjustments(
                    preview_alb,
                    options=self.albedo_options,
                    rules=self.albedo_shift_rules,
                )
                images["albedo"] = adjusted
            elif self.map_type == "height":
                if self.height_depth is None:
                    return
                guide_down = downsample_for_preview(self.height_guide) if self.height_guide is not None else None
                depth_down = downsample_for_preview(self.height_depth)
                height_arr = build_height_map(depth_down, self.height_options, guide=guide_down)
                if self.height_seamless_orig_size is not None:
                    scale = depth_down.shape[1] / self.height_depth.shape[1]
                    ow, oh = self.height_seamless_orig_size
                    small_size = (max(1, int(round(ow * scale))), max(1, int(round(oh * scale))))
                    _, height_arr, _ = apply_seamless_linear_mask(height_arr, orig_size=small_size, padding_pct=0.10)
                images["height"] = height_map_to_image(height_arr, bit_depth=8)
                if self.make_normal:
                    normal_arr = generate_normal_map(height_arr, self.normal_options)
                    images["normal"] = normal_map_to_image(normal_arr)
            elif self.map_type == "normal":
                if self.height_image is None:
                    return
                height_arr = self._height_array(self.height_image)
                normal_arr = generate_normal_map(height_arr, self.normal_options)
                images["normal"] = normal_map_to_image(normal_arr)
            elif self.map_type == "roughness":
                if self.normal_image is None:
                    return
                normal_arr = self._image_array(self.normal_image, "RGB")
            elif self.map_type == "ao":
                if self.height_image is None:
                    return
                from core.ao_map import generate_ao_map, to_image as ao_to_image
                height_arr = self._height_array(self.height_image)
                ao_arr = generate_ao_map(height_arr, self.ao_options)
                images["ao"] = ao_to_image(ao_arr, bit_depth=8)

            if self.isInterruptionRequested():
                return
            if self.make_roughness and normal_arr is not None:
                roughness_arr = generate_roughness_map(normal_arr, self.roughness_options)
                images["roughness"] = roughness_map_to_image(roughness_arr)

            if self.make_ao and "height" in images and self.height_depth is not None:
                from core.ao_map import generate_ao_map, to_image as ao_to_image
                ao_arr = generate_ao_map(height_arr, self.ao_options)
                images["ao"] = ao_to_image(ao_arr, bit_depth=8)

            if not self.isInterruptionRequested():
                self.result_ready.emit(self.revision, self.map_type, images)
        except Exception as exc:
            self.failed.emit(self.revision, str(exc))


class ProjectSaveWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        path: str,
        project,
        source_images: dict,
        generated_images: dict,
        height_raw=None,
        parent=None,
    ):
        super().__init__(parent)
        self.path = path
        self.project = project
        self.source_images = source_images
        self.generated_images = generated_images
        self.height_raw = height_raw

    def run(self) -> None:
        try:
            from core.project import save_project
            save_project(
                self.path,
                self.project,
                self.source_images,
                self.generated_images,
                height_raw=self.height_raw,
                progress_cb=self.progress.emit,
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class ProjectLoadWorker(QThread):
    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(object, object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            from core.project import load_project
            project, source_images, cache_images, height_raw = load_project(
                self.path, progress_cb=self.progress.emit
            )
            self.finished_ok.emit(project, source_images, cache_images, height_raw)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        out_dir: str,
        items: list[tuple[str, Image.Image, str]],
        parent=None,
    ):
        super().__init__(parent)
        self.out_dir = out_dir
        self.items = items

    def run(self) -> None:
        import os
        from core.height_map import safe_save_image

        exported = []
        total = len(self.items)
        try:
            for idx, (label, img, fname) in enumerate(self.items, 1):
                path = os.path.join(self.out_dir, fname)
                self.progress.emit(idx, total, f"Saving {label} ({fname})…")
                safe_save_image(img, path)
                exported.append(f"{label} -> {fname}")
            self.finished_ok.emit(exported)
        except Exception as exc:
            self.failed.emit(str(exc))


class SingleImageSaveWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, image: Image.Image, path: str, parent=None):
        super().__init__(parent)
        self.image = image
        self.path = path

    def run(self) -> None:
        try:
            from core.height_map import safe_save_image
            safe_save_image(self.image, self.path)
            self.finished_ok.emit(self.path)
        except Exception as exc:
            self.failed.emit(str(exc))
