from __future__ import annotations

from enum import Enum


class MapType(str, Enum):
    ALBEDO = "albedo"
    HEIGHT = "height"
    NORMAL = "normal"
    ROUGHNESS = "roughness"
    AO = "ao"


MAP_TYPE_ORDER: list[MapType] = [
    MapType.ALBEDO,
    MapType.HEIGHT,
    MapType.NORMAL,
    MapType.ROUGHNESS,
    MapType.AO,
]

MAP_TYPE_LABELS: dict[MapType, str] = {
    MapType.ALBEDO: "Albedo",
    MapType.HEIGHT: "Height",
    MapType.NORMAL: "Normal",
    MapType.ROUGHNESS: "Roughness",
    MapType.AO: "Ambient Occlusion",
}

GENERATABLE_MAP_TYPES: set[MapType] = {
    MapType.ALBEDO,
    MapType.HEIGHT,
    MapType.NORMAL,
    MapType.ROUGHNESS,
    MapType.AO,
}

UNIMPLEMENTED_MAP_TYPES: set[MapType] = set()
