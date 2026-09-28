"""Preview-only display settings: camera, lighting, grid/rotate/wireframe,
and environment. Fully independent from material-generation parameters —
these only affect how the 3D preview is rendered, not the exported maps.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

MESH_LABELS = {"sphere": "Sphere", "cube": "Cube", "plane": "Plane"}
MESH_VALUES = {v: k for k, v in MESH_LABELS.items()}


class PreviewPropertiesPanel(QWidget):
    settingsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PreviewPropertiesPanel")
        self._hdri_path: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        col1 = QVBoxLayout()
        col1.setSpacing(8)
        col1.addWidget(self._build_geometry_group())
        col1.addWidget(self._build_camera_group())
        col1.addWidget(self._build_lighting_group())
        col1.addStretch(1)

        col2 = QVBoxLayout()
        col2.setSpacing(8)
        col2.addWidget(self._build_display_group())
        col2.addWidget(self._build_parallax_group())
        col2.addWidget(self._build_environment_group())
        col2.addStretch(1)

        layout.addLayout(col1, 1)
        layout.addLayout(col2, 1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_geometry_group(self) -> QGroupBox:
        group = QGroupBox("Geometry")
        layout = QFormLayout(group)

        self.mesh_combo = QComboBox()
        self.mesh_combo.addItems([MESH_LABELS["sphere"], MESH_LABELS["cube"], MESH_LABELS["plane"]])
        self.mesh_combo.currentTextChanged.connect(self._emit_changed)

        layout.addRow("Mesh:", self.mesh_combo)
        return group

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera")
        layout = QFormLayout(group)

        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(20.0, 100.0)
        self.fov_spin.setValue(75.0)
        self.fov_spin.setSuffix("°")
        self.fov_spin.valueChanged.connect(self._emit_changed)

        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.01, 20.0)
        self.distance_spin.setSingleStep(0.1)
        self.distance_spin.setDecimals(2)
        self.distance_spin.setValue(3.5)
        self.distance_spin.valueChanged.connect(self._emit_changed)

        self.rotation_speed_spin = QDoubleSpinBox()
        self.rotation_speed_spin.setRange(0.0, 5.0)
        self.rotation_speed_spin.setSingleStep(0.1)
        self.rotation_speed_spin.setValue(0.5)
        self.rotation_speed_spin.valueChanged.connect(self._emit_changed)

        self.pivot_x_spin = QDoubleSpinBox()
        self.pivot_x_spin.setRange(-50.0, 50.0)
        self.pivot_x_spin.setSingleStep(0.1)
        self.pivot_x_spin.setDecimals(2)
        self.pivot_x_spin.setValue(0.0)
        self.pivot_x_spin.valueChanged.connect(self._emit_changed)

        self.pivot_y_spin = QDoubleSpinBox()
        self.pivot_y_spin.setRange(-50.0, 50.0)
        self.pivot_y_spin.setSingleStep(0.1)
        self.pivot_y_spin.setDecimals(2)
        self.pivot_y_spin.setValue(0.0)
        self.pivot_y_spin.valueChanged.connect(self._emit_changed)

        self.pivot_z_spin = QDoubleSpinBox()
        self.pivot_z_spin.setRange(-50.0, 50.0)
        self.pivot_z_spin.setSingleStep(0.1)
        self.pivot_z_spin.setDecimals(2)
        self.pivot_z_spin.setValue(0.0)
        self.pivot_z_spin.valueChanged.connect(self._emit_changed)

        layout.addRow("FOV:", self.fov_spin)
        layout.addRow("Distance:", self.distance_spin)
        layout.addRow("Rotation Speed:", self.rotation_speed_spin)
        layout.addRow("Pivot X:", self.pivot_x_spin)
        layout.addRow("Pivot Y:", self.pivot_y_spin)
        layout.addRow("Pivot Z:", self.pivot_z_spin)

        reset_pivot_btn = QPushButton("Reset Pivot")
        reset_pivot_btn.setToolTip("Reset camera target pivot point back to (0, 0, 0)")
        reset_pivot_btn.clicked.connect(self._reset_pivot)
        layout.addRow("", reset_pivot_btn)
        return group

    def _build_lighting_group(self) -> QGroupBox:
        group = QGroupBox("Lighting")
        layout = QFormLayout(group)

        self.light_intensity_spin = QDoubleSpinBox()
        self.light_intensity_spin.setRange(0.0, 5.0)
        self.light_intensity_spin.setSingleStep(0.1)
        self.light_intensity_spin.setValue(1.2)
        self.light_intensity_spin.valueChanged.connect(self._emit_changed)

        self.env_intensity_spin = QDoubleSpinBox()
        self.env_intensity_spin.setRange(0.0, 5.0)
        self.env_intensity_spin.setSingleStep(0.1)
        self.env_intensity_spin.setValue(0.6)
        self.env_intensity_spin.valueChanged.connect(self._emit_changed)

        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 4.0)
        self.exposure_spin.setSingleStep(0.1)
        self.exposure_spin.setValue(1.0)
        self.exposure_spin.valueChanged.connect(self._emit_changed)

        self.light_azimuth_spin = QDoubleSpinBox()
        self.light_azimuth_spin.setRange(0.0, 360.0)
        self.light_azimuth_spin.setSingleStep(5.0)
        self.light_azimuth_spin.setValue(45.0)
        self.light_azimuth_spin.setSuffix("°")
        self.light_azimuth_spin.setToolTip("Compass direction the light comes from, around the vertical axis.")
        self.light_azimuth_spin.valueChanged.connect(self._emit_changed)

        self.light_elevation_spin = QDoubleSpinBox()
        self.light_elevation_spin.setRange(-90.0, 90.0)
        self.light_elevation_spin.setSingleStep(5.0)
        self.light_elevation_spin.setValue(50.0)
        self.light_elevation_spin.setSuffix("°")
        self.light_elevation_spin.setToolTip("Height of the light above (positive) or below (negative) the horizon.")
        self.light_elevation_spin.valueChanged.connect(self._emit_changed)

        layout.addRow("Light Intensity:", self.light_intensity_spin)
        layout.addRow("Environment Intensity:", self.env_intensity_spin)
        layout.addRow("Exposure:", self.exposure_spin)
        layout.addRow("Light Azimuth:", self.light_azimuth_spin)
        layout.addRow("Light Elevation:", self.light_elevation_spin)
        return group

    def _build_display_group(self) -> QGroupBox:
        group = QGroupBox("Display")
        layout = QFormLayout(group)

        self.show_grid_checkbox = QCheckBox()
        self.show_grid_checkbox.stateChanged.connect(self._emit_changed)

        self.auto_rotate_checkbox = QCheckBox()
        self.auto_rotate_checkbox.setChecked(True)
        self.auto_rotate_checkbox.stateChanged.connect(self._emit_changed)

        self.wireframe_checkbox = QCheckBox()
        self.wireframe_checkbox.stateChanged.connect(self._emit_changed)

        self.uv_tiling_spin = QDoubleSpinBox()
        self.uv_tiling_spin.setRange(1.0, 5.0)
        self.uv_tiling_spin.setSingleStep(0.5)
        self.uv_tiling_spin.setValue(1.0)
        self.uv_tiling_spin.setSuffix("x")
        self.uv_tiling_spin.setToolTip("Tile the texture across the surface to test seamless wrapping.")
        self.uv_tiling_spin.valueChanged.connect(self._emit_changed)

        layout.addRow("Show Grid:", self.show_grid_checkbox)
        layout.addRow("Auto Rotate:", self.auto_rotate_checkbox)
        layout.addRow("Wireframe Mode:", self.wireframe_checkbox)
        layout.addRow("UV Tiling:", self.uv_tiling_spin)
        return group

    def _build_parallax_group(self) -> QGroupBox:
        group = QGroupBox("Parallax Occlusion")
        layout = QFormLayout(group)

        self.pom_enabled_checkbox = QCheckBox()
        self.pom_enabled_checkbox.setChecked(True)
        self.pom_enabled_checkbox.setToolTip("Uses the Height map to add view-dependent surface depth in the viewport.")
        self.pom_enabled_checkbox.stateChanged.connect(self._emit_changed)

        self.pom_depth_spin = QDoubleSpinBox()
        self.pom_depth_spin.setRange(0.0, 0.20)
        self.pom_depth_spin.setSingleStep(0.005)
        self.pom_depth_spin.setDecimals(3)
        self.pom_depth_spin.setValue(0.04)
        self.pom_depth_spin.setToolTip("UV-space parallax depth. Keep low to avoid stretched detail at grazing angles.")
        self.pom_depth_spin.valueChanged.connect(self._emit_changed)

        self.pom_quality_spin = QSpinBox()
        self.pom_quality_spin.setRange(8, 64)
        self.pom_quality_spin.setSingleStep(4)
        self.pom_quality_spin.setValue(32)
        self.pom_quality_spin.setToolTip("Maximum ray-march layers. Higher improves grazing angles but costs GPU time.")
        self.pom_quality_spin.valueChanged.connect(self._emit_changed)

        layout.addRow("Enable POM:", self.pom_enabled_checkbox)
        layout.addRow("Depth:", self.pom_depth_spin)
        layout.addRow("Quality:", self.pom_quality_spin)
        return group

    def _build_environment_group(self) -> QGroupBox:
        group = QGroupBox("Environment")
        layout = QFormLayout(group)

        hdri_row = QHBoxLayout()
        self.hdri_edit = QLineEdit()
        self.hdri_edit.setReadOnly(True)
        self.hdri_edit.setPlaceholderText("None")
        hdri_browse_btn = QPushButton("Browse…")
        hdri_browse_btn.clicked.connect(self._browse_hdri)
        hdri_clear_btn = QPushButton("Clear")
        hdri_clear_btn.clicked.connect(self._clear_hdri)
        hdri_row.addWidget(self.hdri_edit, 1)
        hdri_row.addWidget(hdri_browse_btn)
        hdri_row.addWidget(hdri_clear_btn)

        self.background_brightness_spin = QDoubleSpinBox()
        self.background_brightness_spin.setRange(0.0, 3.0)
        self.background_brightness_spin.setSingleStep(0.1)
        self.background_brightness_spin.setValue(1.0)
        self.background_brightness_spin.valueChanged.connect(self._emit_changed)

        layout.addRow("HDRI Selection:", hdri_row)
        layout.addRow("Background Brightness:", self.background_brightness_spin)
        return group

    def _browse_hdri(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select HDRI / Environment Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self._hdri_path = path
            self.hdri_edit.setText(path)
            self._emit_changed()

    def _clear_hdri(self) -> None:
        self._hdri_path = None
        self.hdri_edit.clear()
        self._emit_changed()

    def _emit_changed(self, *_args) -> None:
        self.settingsChanged.emit(self.get_settings())

    def _reset_pivot(self) -> None:
        self.pivot_x_spin.setValue(0.0)
        self.pivot_y_spin.setValue(0.0)
        self.pivot_z_spin.setValue(0.0)

    def get_settings(self) -> dict:
        return {
            "mesh": MESH_VALUES.get(self.mesh_combo.currentText(), "sphere"),
            "fov": self.fov_spin.value(),
            "distance": self.distance_spin.value(),
            "rotation_speed": self.rotation_speed_spin.value(),
            "pivot_x": self.pivot_x_spin.value(),
            "pivot_y": self.pivot_y_spin.value(),
            "pivot_z": self.pivot_z_spin.value(),
            "light_intensity": self.light_intensity_spin.value(),
            "env_intensity": self.env_intensity_spin.value(),
            "exposure": self.exposure_spin.value(),
            "light_azimuth": self.light_azimuth_spin.value(),
            "light_elevation": self.light_elevation_spin.value(),
            "show_grid": self.show_grid_checkbox.isChecked(),
            "auto_rotate": self.auto_rotate_checkbox.isChecked(),
            "wireframe": self.wireframe_checkbox.isChecked(),
            "uv_tiling": self.uv_tiling_spin.value(),
            "pom_enabled": self.pom_enabled_checkbox.isChecked(),
            "pom_height_scale": self.pom_depth_spin.value(),
            "pom_max_layers": self.pom_quality_spin.value(),
            "hdri_path": self._hdri_path,
            "background_brightness": self.background_brightness_spin.value(),
        }

    def apply_settings(self, data: dict) -> None:
        if not data:
            return
        self.mesh_combo.setCurrentText(MESH_LABELS.get(data.get("mesh", "sphere"), MESH_LABELS["sphere"]))
        self.fov_spin.setValue(data.get("fov", self.fov_spin.value()))
        self.distance_spin.setValue(data.get("distance", self.distance_spin.value()))
        self.rotation_speed_spin.setValue(data.get("rotation_speed", self.rotation_speed_spin.value()))
        self.pivot_x_spin.setValue(data.get("pivot_x", self.pivot_x_spin.value()))
        self.pivot_y_spin.setValue(data.get("pivot_y", self.pivot_y_spin.value()))
        self.pivot_z_spin.setValue(data.get("pivot_z", self.pivot_z_spin.value()))
        self.light_intensity_spin.setValue(data.get("light_intensity", self.light_intensity_spin.value()))
        self.env_intensity_spin.setValue(data.get("env_intensity", self.env_intensity_spin.value()))
        self.exposure_spin.setValue(data.get("exposure", self.exposure_spin.value()))
        self.light_azimuth_spin.setValue(data.get("light_azimuth", self.light_azimuth_spin.value()))
        self.light_elevation_spin.setValue(data.get("light_elevation", self.light_elevation_spin.value()))
        self.show_grid_checkbox.setChecked(data.get("show_grid", False))
        self.auto_rotate_checkbox.setChecked(data.get("auto_rotate", True))
        self.wireframe_checkbox.setChecked(data.get("wireframe", False))
        self.uv_tiling_spin.setValue(data.get("uv_tiling", self.uv_tiling_spin.value()))
        self.pom_enabled_checkbox.setChecked(data.get("pom_enabled", True))
        self.pom_depth_spin.setValue(data.get("pom_height_scale", self.pom_depth_spin.value()))
        self.pom_quality_spin.setValue(data.get("pom_max_layers", self.pom_quality_spin.value()))
        self._hdri_path = data.get("hdri_path")
        self.hdri_edit.setText(self._hdri_path or "")
        self.background_brightness_spin.setValue(data.get("background_brightness", self.background_brightness_spin.value()))

    def set_distance_value(self, val: float) -> None:
        self.distance_spin.blockSignals(True)
        self.distance_spin.setValue(val)
        self.distance_spin.blockSignals(False)

    def set_pivot_values(self, x: float, y: float, z: float) -> None:
        self.pivot_x_spin.blockSignals(True)
        self.pivot_y_spin.blockSignals(True)
        self.pivot_z_spin.blockSignals(True)
        self.pivot_x_spin.setValue(x)
        self.pivot_y_spin.setValue(y)
        self.pivot_z_spin.setValue(z)
        self.pivot_x_spin.blockSignals(False)
        self.pivot_y_spin.blockSignals(False)
        self.pivot_z_spin.blockSignals(False)

    def set_light_values(self, az: float, el: float) -> None:
        self.light_azimuth_spin.blockSignals(True)
        self.light_elevation_spin.blockSignals(True)
        self.light_azimuth_spin.setValue(az)
        self.light_elevation_spin.setValue(el)
        self.light_azimuth_spin.blockSignals(False)
        self.light_elevation_spin.blockSignals(False)

    def set_auto_rotate(self, val: bool) -> None:
        self.auto_rotate_checkbox.blockSignals(True)
        self.auto_rotate_checkbox.setChecked(val)
        self.auto_rotate_checkbox.blockSignals(False)
