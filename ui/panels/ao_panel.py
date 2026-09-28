"""Ambient Occlusion (AO) map panel — controls for deriving contact shadows from Height.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QPushButton,
    QSpinBox,
)

from core.ao_map import AOMapOptions
from ui.panels.base import GeneratePropertiesContainer


class AOPanel(GeneratePropertiesContainer):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.note_label = QLabel(
            "No Height map yet, so Generate Map has nothing to work from "
            "(you can still upload an Ambient Occlusion map directly instead)."
        )
        self.note_label.setWordWrap(True)
        self.note_label.setProperty("role", "dim")
        self.generate_page.add_wide(self.note_label)

        self.generate_page.add_section("Horizon Ray Sampling")

        self.samples_combo = QComboBox()
        self.samples_combo.addItems(["4 rays (Fast)", "8 rays (Balanced)", "12 rays (High)", "16 rays (Ultra)"])
        self.samples_combo.setCurrentIndex(1)
        self.samples_combo.currentIndexChanged.connect(self.propertiesChanged.emit)
        self.generate_page.add_param("Sample rays", self.samples_combo)

        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(2, 64)
        self.radius_spin.setValue(16)
        self.radius_spin.setToolTip("Search distance in pixels for neighboring occluding structures.")
        self.radius_spin.valueChanged.connect(self.propertiesChanged.emit)
        self.generate_page.add_param("Search radius (px)", self.radius_spin)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 10)
        self.blur_spin.setValue(1)
        self.blur_spin.setToolTip("Gentle post-softening to eliminate stepped sampling artifacts.")
        self.blur_spin.valueChanged.connect(self.propertiesChanged.emit)
        self.generate_page.add_param("Pre-smooth (px)", self.blur_spin)

        reset_gen_btn = QPushButton("Reset Generate Defaults")
        reset_gen_btn.clicked.connect(self.reset_generate_defaults)
        self.generate_page.add_wide(reset_gen_btn)
        self.generate_page.add_stretch()

        page = self.properties_page
        page.add_section("Intensity & Curve")

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.1, 5.0)
        self.strength_spin.setSingleStep(0.1)
        self.strength_spin.setValue(1.0)
        self.strength_spin.setToolTip("Multiplier for contact shadow darkness in crevices.")
        self.strength_spin.valueChanged.connect(self.propertiesChanged.emit)
        page.add_param("Strength", self.strength_spin)

        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.2, 4.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.5)
        self.contrast_spin.setToolTip("Occlusion gamma power: >1.0 deepens cracks while keeping flat areas bright.")
        self.contrast_spin.valueChanged.connect(self.propertiesChanged.emit)
        page.add_param("Contrast curve", self.contrast_spin)

        self.invert_checkbox = QCheckBox("Invert (Dark highlights)")
        self.invert_checkbox.stateChanged.connect(self.propertiesChanged.emit)
        page.add_param("Invert", self.invert_checkbox)

        page.add_section("Format")
        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["8-bit", "16-bit"])
        self.bit_depth_combo.currentTextChanged.connect(self.propertiesChanged.emit)
        page.add_param("Export bit depth", self.bit_depth_combo)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_to_defaults)
        page.add_wide(reset_btn)
        page.add_stretch()

    def set_has_height(self, has_height: bool) -> None:
        self.note_label.setVisible(not has_height)

    def _get_samples_count(self) -> int:
        idx = self.samples_combo.currentIndex()
        return [4, 8, 12, 16][idx] if 0 <= idx < 4 else 8

    def reset_generate_defaults(self) -> None:
        self.samples_combo.setCurrentIndex(1)
        self.radius_spin.setValue(16)
        self.blur_spin.setValue(1)
        self.propertiesChanged.emit()

    def reset_to_defaults(self) -> None:
        self.strength_spin.setValue(1.0)
        self.contrast_spin.setValue(1.5)
        self.invert_checkbox.setChecked(False)
        self.bit_depth_combo.setCurrentText("8-bit")
        self.reset_generate_defaults()
        self.propertiesChanged.emit()

    def current_options(self) -> AOMapOptions:
        bit_depth = 16 if self.bit_depth_combo.currentText() == "16-bit" else 8
        return AOMapOptions(
            strength=self.strength_spin.value(),
            radius=self.radius_spin.value(),
            samples=self._get_samples_count(),
            contrast=self.contrast_spin.value(),
            blur_radius=self.blur_spin.value(),
            invert=self.invert_checkbox.isChecked(),
            bit_depth=bit_depth,
        )

    def properties_to_dict(self) -> dict:
        return {
            "strength": self.strength_spin.value(),
            "radius": self.radius_spin.value(),
            "samples_idx": self.samples_combo.currentIndex(),
            "contrast": self.contrast_spin.value(),
            "blur_radius": self.blur_spin.value(),
            "invert": self.invert_checkbox.isChecked(),
            "bit_depth": self.bit_depth_combo.currentText(),
        }

    def properties_apply_dict(self, data: dict) -> None:
        if not data:
            return
        self.strength_spin.setValue(data.get("strength", 1.0))
        self.radius_spin.setValue(data.get("radius", 16))
        self.samples_combo.setCurrentIndex(data.get("samples_idx", 1))
        self.contrast_spin.setValue(data.get("contrast", 1.5))
        self.blur_spin.setValue(data.get("blur_radius", 1))
        self.invert_checkbox.setChecked(data.get("invert", False))
        self.bit_depth_combo.setCurrentText(data.get("bit_depth", "8-bit"))

    to_options_dict = properties_to_dict
    apply_options_dict = properties_apply_dict
