"""PBRCELAIN project file (.pcln): a zip container of metadata + images.

Layout inside the zip:
    project.json                  - ProjectData serialized to JSON
    images/<slot>_source.png      - uploaded source image for a map slot (if any)
    images/<slot>_cache.png       - last-generated output for a map slot (if any)
    images/<slot>_cache.tiff      - same, for 32-bit float (mode "F") outputs,
                                    which PNG cannot store
    data/height_raw_depth.npy     - raw depth-model output for Height (if generated this session)
    data/height_raw_guide.npy     - matching source-luminance guide array
    data/height_raw_meta.json     - near_is_large flag for the raw depth array

Keeping generated outputs (height/normal) cached alongside sources means
reopening a project doesn't require re-running the (potentially very slow)
depth-estimation model just to see the material again. The raw depth array
is *also* kept (not just the final processed Height PNG) because Height's
Properties (detrend/clip/gamma/...) are post-processing applied to that raw
field — without it, editing Properties after reopening a project would have
nothing to recompute from and could only re-process the already-processed
PNG, giving different (wrong) results.
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable, Optional

import numpy as np
from PIL import Image

from core.depth_models import DepthResult
from core.maps import MapType

PROJECT_VERSION = 1
PROJECT_EXTENSION = ".pcln"
PROJECT_OPEN_EXTENSIONS = (PROJECT_EXTENSION, ".zip")
PROJECT_FILE_FILTER = "PBRCELAIN Project (" + " ".join(f"*{ext}" for ext in PROJECT_OPEN_EXTENSIONS) + ")"


def is_project_path(path: str) -> bool:
    """True for files that can be opened as a project: .pcln, or a .zip with
    the same layout (a .pcln is a zip archive)."""
    return path.lower().endswith(PROJECT_OPEN_EXTENSIONS)


@dataclass
class MapSlotData:
    has_source: bool = False
    has_cache: bool = False
    options: dict = field(default_factory=dict)


@dataclass
class ProjectData:
    version: int = PROJECT_VERSION
    map_slots: dict[str, MapSlotData] = field(default_factory=dict)
    depth_model: dict = field(default_factory=dict)
    preview_settings: dict = field(default_factory=dict)

    @staticmethod
    def new() -> "ProjectData":
        return ProjectData(map_slots={mt.value: MapSlotData() for mt in MapType})


def _slot_source_name(map_type: MapType) -> str:
    return f"images/{map_type.value}_source.png"


def _slot_cache_name(map_type: MapType, float32: bool = False) -> str:
    ext = "tiff" if float32 else "png"
    return f"images/{map_type.value}_cache.{ext}"


def _encode_image(image: Image.Image) -> tuple[bytes, bool]:
    """Serialize losslessly: PNG, or TIFF for 32-bit float images PNG can't hold."""
    buf = BytesIO()
    is_float = image.mode == "F"
    image.save(buf, format="TIFF" if is_float else "PNG")
    return buf.getvalue(), is_float


_PRESERVED_SOURCE_MODES = ("L", "I;16", "I")


def save_project(
    path: str,
    project: ProjectData,
    source_images: dict[MapType, Optional[Image.Image]],
    cache_images: dict[MapType, Optional[Image.Image]],
    height_raw: Optional[DepthResult] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Write `project` plus any provided images to `path` as a .pcln zip."""
    if progress_cb:
        progress_cb("Packaging project metadata…")

    payload = {
        "version": project.version,
        "map_slots": {
            key: {"has_source": slot.has_source, "has_cache": slot.has_cache, "options": slot.options}
            for key, slot in project.map_slots.items()
        },
        "depth_model": project.depth_model,
        "preview_settings": project.preview_settings,
    }

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(payload, indent=2))
        for map_type in MapType:
            src = source_images.get(map_type)
            if src is not None:
                if progress_cb:
                    progress_cb(f"Compressing {map_type.value.capitalize()} source…")
                buf = BytesIO()
                src.save(buf, format="PNG")
                zf.writestr(_slot_source_name(map_type), buf.getvalue())
            cache = cache_images.get(map_type)
            if cache is not None:
                if progress_cb:
                    progress_cb(f"Compressing {map_type.value.capitalize()} cache…")
                data, is_float = _encode_image(cache)
                zf.writestr(_slot_cache_name(map_type, float32=is_float), data)

        if height_raw is not None:
            if progress_cb:
                progress_cb("Compressing depth model cache…")
            depth_buf = BytesIO()
            np.save(depth_buf, height_raw.depth)
            zf.writestr("data/height_raw_depth.npy", depth_buf.getvalue())
            if height_raw.guide is not None:
                guide_buf = BytesIO()
                np.save(guide_buf, height_raw.guide)
                zf.writestr("data/height_raw_guide.npy", guide_buf.getvalue())
            meta = {"near_is_large": height_raw.near_is_large}
            if height_raw.seamless_orig_size is not None:
                meta["seamless_orig_size"] = list(height_raw.seamless_orig_size)
            zf.writestr("data/height_raw_meta.json", json.dumps(meta))

    if progress_cb:
        progress_cb("Saving complete.")


def load_project(
    path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[ProjectData, dict[MapType, Image.Image], dict[MapType, Image.Image], Optional[DepthResult]]:
    """Read a .pcln (or .zip) project, returning (project metadata, source
    images, cache images, raw Height depth result if one was saved)."""
    if progress_cb:
        progress_cb("Reading project archive…")

    source_images: dict[MapType, Image.Image] = {}
    cache_images: dict[MapType, Image.Image] = {}
    height_raw: Optional[DepthResult] = None

    if not zipfile.is_zipfile(path):
        raise ValueError(f"{os.path.basename(path)} is not a PBRCELAIN project archive.")
    with zipfile.ZipFile(path, "r") as zf:
        if "project.json" not in zf.namelist():
            raise ValueError(
                f"{os.path.basename(path)} is not a PBRCELAIN project: the archive has no project.json."
            )
        payload = json.loads(zf.read("project.json").decode("utf-8"))
        map_slots = {
            key: MapSlotData(
                has_source=slot_data.get("has_source", False),
                has_cache=slot_data.get("has_cache", False),
                options=slot_data.get("options", {}),
            )
            for key, slot_data in payload.get("map_slots", {}).items()
        }
        project = ProjectData(
            version=payload.get("version", PROJECT_VERSION),
            map_slots=map_slots,
            depth_model=payload.get("depth_model", {}),
            preview_settings=payload.get("preview_settings", {}),
        )

        names = set(zf.namelist())
        for map_type in MapType:
            src_name = _slot_source_name(map_type)
            if src_name in names:
                if progress_cb:
                    progress_cb(f"Loading {map_type.value.capitalize()} source…")
                img = Image.open(BytesIO(zf.read(src_name)))
                img.load()
                if img.mode not in _PRESERVED_SOURCE_MODES:
                    img = img.convert("RGB")
                source_images[map_type] = img
            cache_name = next(
                (n for n in (_slot_cache_name(map_type), _slot_cache_name(map_type, float32=True)) if n in names),
                None,
            )
            if cache_name is not None:
                if progress_cb:
                    progress_cb(f"Loading {map_type.value.capitalize()} cache…")
                img = Image.open(BytesIO(zf.read(cache_name)))
                img.load()
                cache_images[map_type] = img

        if "data/height_raw_depth.npy" in names:
            if progress_cb:
                progress_cb("Decompressing depth model cache…")
            depth = np.load(BytesIO(zf.read("data/height_raw_depth.npy")))
            guide = np.load(BytesIO(zf.read("data/height_raw_guide.npy"))) if "data/height_raw_guide.npy" in names else None
            meta = {}
            if "data/height_raw_meta.json" in names:
                meta = json.loads(zf.read("data/height_raw_meta.json").decode("utf-8"))
            seamless_size = meta.get("seamless_orig_size")
            height_raw = DepthResult(
                depth=depth,
                guide=guide,
                near_is_large=meta.get("near_is_large", True),
                seamless_orig_size=tuple(seamless_size) if seamless_size else None,
            )

    if progress_cb:
        progress_cb("Project loaded.")

    return project, source_images, cache_images, height_raw
