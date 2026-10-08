from __future__ import annotations

import base64
import io
from typing import Callable

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)
from PIL import Image
import numpy as np

from core.roughness_map import ROUGHNESS_PRESETS, RoughnessMapOptions
from ui.imperfection_painter import ImperfectionPainterDialog
from ui.theme import legacy_style
from ui.panels.base import GeneratePropertiesContainer


class RoughnessPanel(GeneratePropertiesContainer):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._imperfection_mask: np.ndarray | None = None
        self._underlay_provider: Callable[[], tuple[Image.Image | None, Image.Image | None, Image.Image | None]] | None = None

        self.note_label = QLabel(
            "No Normal map yet, so Generate Map has nothing to work from "
            "(you can still upload a Roughness map directly instead)."
        )
        self.note_label.setWordWrap(True)
        self.note_label.setProperty("role", "dim")
        self.generate_page.add_wide(self.note_label)

        self.generate_page.add_section("PBR Material Preset")
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(ROUGHNESS_PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.generate_page.add_param("Material type", self.preset_combo)

        self.generate_page.add_section("Hybrid Source Blending")
        self.height_cavity_spin = QDoubleSpinBox()
        self.height_cavity_spin.setRange(-1.0, 1.0)
        self.height_cavity_spin.setSingleStep(0.05)
        self.height_cavity_spin.setValue(0.0)
        self.height_cavity_spin.setToolTip(
            "Height Cavity Influence:\n"
            "> 0: Crevices/depressions accumulate dust/dirt and become rougher.\n"
            "< 0: Depressions pool water and become mirror-glossy puddles."
        )
        self.height_cavity_spin.valueChanged.connect(self.propertiesChanged.emit)
        self.generate_page.add_param("Crevice / Cavity", self.height_cavity_spin)

        self.albedo_contrast_spin = QDoubleSpinBox()
        self.albedo_contrast_spin.setRange(0.0, 1.0)
        self.albedo_contrast_spin.setSingleStep(0.05)
        self.albedo_contrast_spin.setValue(0.0)
        self.albedo_contrast_spin.setToolTip("Color and grain variation in photo raises local roughness variance.")
        self.albedo_contrast_spin.valueChanged.connect(self.propertiesChanged.emit)
        self.generate_page.add_param("Albedo contrast", self.albedo_contrast_spin)

        reset_gen_btn = QPushButton("Reset Generate Defaults")
        reset_gen_btn.clicked.connect(self.reset_generate_defaults)
        self.generate_page.add_wide(reset_gen_btn)
        self.generate_page.add_stretch()

        page = self.properties_page
        caveat = QLabel(
            "Derived from Normal map detail — a rough approximation, not a "
            "true material measurement. Unreliable for polished, painted, "
            "or metallic surfaces."
        )
        caveat.setWordWrap(True)
        caveat.setProperty("role", "dim")
        page.add_wide(caveat)

        self.base_roughness_spin = QDoubleSpinBox()
        self.base_roughness_spin.setRange(0.0, 1.0)
        self.base_roughness_spin.setSingleStep(0.05)
        self.base_roughness_spin.setValue(0.3)
        self.base_roughness_spin.setToolTip("Roughness of a perfectly flat/coherent area.")
        self.base_roughness_spin.valueChanged.connect(self.propertiesChanged.emit)
        page.add_param("Base Roughness", self.base_roughness_spin)

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 3.0)
        self.strength_spin.setSingleStep(0.1)
        self.strength_spin.setValue(1.0)
        self.strength_spin.setToolTip("How strongly local Normal-map detail raises roughness.")
        self.strength_spin.valueChanged.connect(self.propertiesChanged.emit)
        page.add_param("Strength", self.strength_spin)

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.5, 10.0)
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setValue(2.0)
        self.window_spin.setSuffix(" %")
        self.window_spin.setToolTip("Footprint size (as % of the shorter image side) the local variance is measured over.")
        self.window_spin.valueChanged.connect(self.propertiesChanged.emit)
        page.add_param("Detail Window", self.window_spin)

        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.setToolTip("Flip the convention, for materials where this heuristic runs backwards.")
        self.invert_checkbox.stateChanged.connect(self.propertiesChanged.emit)
        page.add_param("Convention", self.invert_checkbox)

        page.add_section("Surface Imperfections & Wear")

        paint_container = QWidget()
        paint_row = QHBoxLayout(paint_container)
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setSpacing(6)

        self.paint_btn = QPushButton("🖌️ Paint Imperfections…")
        self.paint_btn.setToolTip("Open brush window to paint custom rust, wear, scratches, or polished glossy spots.")
        self.paint_btn.clicked.connect(self._open_imperfection_painter)
        paint_row.addWidget(self.paint_btn, 1)

        self.clear_paint_btn = QPushButton("Clear")
        self.clear_paint_btn.setToolTip("Clear all painted wear and glossy spots.")
        self.clear_paint_btn.setEnabled(False)
        self.clear_paint_btn.clicked.connect(self._clear_paint)
        paint_row.addWidget(self.clear_paint_btn)

        page.add_wide(paint_container)

        self.imperfection_status_label = QLabel("No custom paint layer")
        legacy_style(self.imperfection_status_label, "color: #71717a; font-size: 11px;", "hint")
        page.add_wide(self.imperfection_status_label)

        self.imperfection_strength_spin = QDoubleSpinBox()
        self.imperfection_strength_spin.setRange(0.0, 1.0)
        self.imperfection_strength_spin.setSingleStep(0.05)
        self.imperfection_strength_spin.setValue(1.0)
        self.imperfection_strength_spin.setToolTip("Overall intensity / blend strength of your painted imperfection layer.")
        self.imperfection_strength_spin.valueChanged.connect(self.propertiesChanged.emit)
        page.add_param("Paint Strength", self.imperfection_strength_spin)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        page.add_wide(reset_btn)

        page.add_stretch()

    def set_underlay_provider(
        self,
        provider: Callable[[], tuple[Image.Image | None, Image.Image | None, Image.Image | None]],
    ) -> None:
        self._underlay_provider = provider

    def _open_imperfection_painter(self) -> None:
        albedo, normal, roughness = (None, None, None)
        if self._underlay_provider is not None:
            albedo, normal, roughness = self._underlay_provider()

        dialog = ImperfectionPainterDialog(
            mask=self._imperfection_mask,
            albedo=albedo,
            normal=normal,
            roughness=roughness,
            parent=self.window(),
        )
        if dialog.exec() == ImperfectionPainterDialog.DialogCode.Accepted:
            self._imperfection_mask = dialog.result_mask()
            self._update_paint_status_ui()
            self.propertiesChanged.emit()

    def _clear_paint(self) -> None:
        self._imperfection_mask = None
        self._update_paint_status_ui()
        self.propertiesChanged.emit()

    def _update_paint_status_ui(self) -> None:
        has_paint = bool(self._imperfection_mask is not None and np.any(self._imperfection_mask[..., 3] > 0))
        self.clear_paint_btn.setEnabled(has_paint)
        if has_paint:
            h, w = self._imperfection_mask.shape[:2]
            self.imperfection_status_label.setText(f"● Custom paint active ({w}×{h})")
            legacy_style(self.imperfection_status_label, "color: #38bdf8; font-size: 11px; font-weight: bold;", "hint-active")
        else:
            self.imperfection_status_label.setText("No custom paint layer")
            legacy_style(self.imperfection_status_label, "color: #71717a; font-size: 11px;", "hint")

    def _on_preset_changed(self, preset_name: str) -> None:
        if preset_name not in ROUGHNESS_PRESETS or preset_name == "Custom":
            return
        p = ROUGHNESS_PRESETS[preset_name]
        self.base_roughness_spin.blockSignals(True)
        self.strength_spin.blockSignals(True)
        self.height_cavity_spin.blockSignals(True)
        self.albedo_contrast_spin.blockSignals(True)

        self.base_roughness_spin.setValue(p["base_roughness"])
        self.strength_spin.setValue(p["strength"])
        self.height_cavity_spin.setValue(p["height_cavity"])
        self.albedo_contrast_spin.setValue(p["albedo_contrast"])

        self.base_roughness_spin.blockSignals(False)
        self.strength_spin.blockSignals(False)
        self.height_cavity_spin.blockSignals(False)
        self.albedo_contrast_spin.blockSignals(False)
        self.propertiesChanged.emit()

    def reset_generate_defaults(self) -> None:
        self.preset_combo.setCurrentText("Custom")
        self.height_cavity_spin.setValue(0.0)
        self.albedo_contrast_spin.setValue(0.0)
        self.propertiesChanged.emit()

    def reset_to_defaults(self) -> None:
        self.base_roughness_spin.setValue(0.3)
        self.strength_spin.setValue(1.0)
        self.window_spin.setValue(2.0)
        self.invert_checkbox.setChecked(False)
        self.imperfection_strength_spin.setValue(1.0)
        self._imperfection_mask = None
        self._update_paint_status_ui()
        self.reset_generate_defaults()
        self.propertiesChanged.emit()

    def set_has_normal(self, has_normal: bool) -> None:
        self.note_label.setVisible(not has_normal)

    def current_options(self) -> RoughnessMapOptions:
        return RoughnessMapOptions(
            base_roughness=self.base_roughness_spin.value(),
            strength=self.strength_spin.value(),
            window_pct=self.window_spin.value(),
            invert=self.invert_checkbox.isChecked(),
            imperfection_mask=self._imperfection_mask,
            imperfection_strength=self.imperfection_strength_spin.value(),
            preset_name=self.preset_combo.currentText(),
            height_cavity_factor=self.height_cavity_spin.value(),
            albedo_contrast_factor=self.albedo_contrast_spin.value(),
        )

    def properties_to_dict(self) -> dict:
        d = {
            "base_roughness": self.base_roughness_spin.value(),
            "strength": self.strength_spin.value(),
            "window_pct": self.window_spin.value(),
            "invert": self.invert_checkbox.isChecked(),
            "imperfection_strength": self.imperfection_strength_spin.value(),
            "preset_name": self.preset_combo.currentText(),
            "height_cavity_factor": self.height_cavity_spin.value(),
            "albedo_contrast_factor": self.albedo_contrast_spin.value(),
        }
        if self._imperfection_mask is not None and np.any(self._imperfection_mask[..., 3] > 0):
            pil_mask = Image.fromarray(self._imperfection_mask, mode="RGBA")
            bio = io.BytesIO()
            pil_mask.save(bio, format="PNG")
            d["imperfection_mask_b64"] = base64.b64encode(bio.getvalue()).decode("ascii")
        return d

    def properties_apply_dict(self, data: dict) -> None:
        if not data:
            return
        self.base_roughness_spin.setValue(data.get("base_roughness", self.base_roughness_spin.value()))
        self.strength_spin.setValue(data.get("strength", self.strength_spin.value()))
        self.window_spin.setValue(data.get("window_pct", self.window_spin.value()))
        self.invert_checkbox.setChecked(data.get("invert", False))
        self.imperfection_strength_spin.setValue(data.get("imperfection_strength", 1.0))

        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentText(data.get("preset_name", "Custom"))
        self.preset_combo.blockSignals(False)
        self.height_cavity_spin.setValue(data.get("height_cavity_factor", 0.0))
        self.albedo_contrast_spin.setValue(data.get("albedo_contrast_factor", 0.0))

        if "imperfection_mask_b64" in data:
            try:
                raw_bytes = base64.b64decode(data["imperfection_mask_b64"])
                pil_mask = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
                self._imperfection_mask = np.asarray(pil_mask, dtype=np.uint8)
            except Exception:
                self._imperfection_mask = None
        else:
            self._imperfection_mask = None

        self._update_paint_status_ui()

    to_options_dict = properties_to_dict
    apply_options_dict = properties_apply_dict
