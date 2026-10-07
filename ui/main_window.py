"""Main application window for PBRCELAIN.

Project Section (left, gray): material slots bar + active map inspector +
dynamic property panels + Generate All / Generate Map / Kill Process controls.
Preview Section (right, purple): real-time 3D viewport preview + display-only
preview settings with bi-directional synchronization.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import QSettings, QSize, QTimer, Qt
from PyQt6.QtGui import QAction, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QMenu,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.depth_models import DepthEstimator, DepthResult
from core.height_map import (
    apply_force_seamless_blur,
    apply_seamless_linear_mask,
    build_height_map,
    build_seamless_extended_tile,
    downsample_for_preview,
)
from core.height_map import to_image as height_map_to_image
from core.maps import MAP_TYPE_LABELS, MAP_TYPE_ORDER, MapType
from core.normal_map import generate_normal_map
from core.normal_map import to_image as normal_map_to_image
from core.project import PROJECT_EXTENSION, MapSlotData, ProjectData
from core.roughness_map import generate_roughness_map
from core.roughness_map import to_image as roughness_map_to_image
from core.ao_map import generate_ao_map
from core.curvature_map import generate_curvature_map
from core.curvature_map import to_image as curvature_map_to_image
from core.ao_map import to_image as ao_map_to_image
from core.albedo_adjust import apply_albedo_adjustments
from ui.panels.albedo_panel import AlbedoPanel
from ui.panels.height_panel import HeightPanel
from ui.panels.normal_panel import NormalPanel
from ui.panels.roughness_panel import RoughnessPanel
from ui.panels.ao_panel import AOPanel
from ui.preview_3d import Preview3DWidget
from ui.preview_properties_panel import PreviewPropertiesPanel
from ui.progress_dialog import OperationProgressDialog
from ui import mac_window, window_layout
from ui.theme import IS_MAC, IS_WINDOWS, PLATFORM_THEME, legacy_style, pick, theme_changed
from ui.tiled_preview_dialog import TiledPreviewDialog
from ui.widgets import IMAGE_EXTENSIONS, ImageLabel, SlotButton
from ui.workers import (
    ExportWorker,
    InferenceWorker,
    ModelLoadWorker,
    ProjectLoadWorker,
    ProjectSaveWorker,
    PropertiesPreviewWorker,
    SingleImageSaveWorker,
)


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL Image to a QPixmap, properly handling RGB, 8-bit L, 16-bit I;16/I, and 32-bit float F."""
    if img.mode == "RGB":
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    elif img.mode in ("I;16", "I"):
        arr = (np.asarray(img, dtype=np.float32) / 256.0).clip(0, 255).astype(np.uint8)
        data = arr.tobytes()
        qimg = QImage(data, img.width, img.height, img.width, QImage.Format.Format_Grayscale8)
    elif img.mode == "F":
        arr = (np.asarray(img, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        data = arr.tobytes()
        qimg = QImage(data, img.width, img.height, img.width, QImage.Format.Format_Grayscale8)
    else:
        img = img.convert("L")
        data = img.tobytes("raw", "L")
        qimg = QImage(data, img.width, img.height, img.width, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def _image_to_height_array(image: Image.Image) -> np.ndarray:
    """Grayscale-normalize any uploaded/generated Height image to float32 [0,1]."""
    if image.mode in ("I;16", "I"):
        return np.asarray(image, dtype=np.float32) / 65535.0
    if image.mode == "F":
        return np.asarray(image, dtype=np.float32).clip(0.0, 1.0)
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _image_to_normal_array(image: Image.Image) -> np.ndarray:
    """Normalize any uploaded/generated Normal image to a [0,1] float32 HxWx3 array."""
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


RECENT_PROJECTS_KEY = "recentProjects"
MAX_RECENT_PROJECTS = 8


def _heading(text: str) -> str:
    """Section heading text: title case in the macOS and Windows designs,
    ALL CAPS in the legacy theme."""
    return text if PLATFORM_THEME == "legacy" else text.title()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1440, 860)
        if IS_MAC:
            mac_window.hide_title_bar(self)
        self._initial_layout_done = False
        self._taskbar_progress = None
        self.setAcceptDrops(True)

        self.project = ProjectData.new()
        self.project_path: Optional[str] = None
        self.dirty = False

        self.source_images: dict[MapType, Image.Image] = {}
        self.generated_images: dict[MapType, Image.Image] = {}

        self.estimator: Optional[DepthEstimator] = None
        self.model_thread: Optional[ModelLoadWorker] = None
        self.infer_thread: Optional[InferenceWorker] = None
        self._height_depth_result: Optional[DepthResult] = None
        self._generating_all = False
        self._infer_is_seamless: bool = False
        self._infer_orig_albedo_size: Optional[tuple[int, int]] = None
        self._height_before_infer: Optional[Image.Image] = None

        self.current_map_type: MapType = MapType.ALBEDO
        self._bottom_bar_mode = "generate"
        self._properties_snapshot: dict[MapType, dict] = {}
        self._draft_images: dict[MapType, Image.Image] = {}
        self.properties_preview_thread: Optional[PropertiesPreviewWorker] = None
        self._properties_preview_revision = 0
        self._properties_preview_pending = False

        self._properties_debounce = QTimer(self)
        self._properties_debounce.setSingleShot(True)
        self._properties_debounce.setInterval(180)
        self._properties_debounce.timeout.connect(self._refresh_properties_draft)

        self._slot_buttons: dict[MapType, QPushButton] = {}

        self._build_toolbar()
        self._build_ui()
        self._update_window_title()
        self._reset_properties_snapshots()
        self._select_map_type(MapType.ALBEDO)

    def _build_toolbar(self) -> None:
        if PLATFORM_THEME == "mac":
            self._build_mac_menu()
            return
        if PLATFORM_THEME == "windows":
            self._build_windows_commands()
            return

        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        def add(text: str, handler, shortcut: Optional[str] = None) -> QAction:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(handler)
            toolbar.addAction(action)
            return action

        add("New Project", self._new_project, "Ctrl+N")
        add("Open Project", self._open_project, "Ctrl+O")
        add("Save", self._save_project, "Ctrl+S")
        add("Save As", self._save_project_as, "Ctrl+Shift+S")
        toolbar.addSeparator()
        add("Export Maps", self._export_maps, "Ctrl+E")
        toolbar.addSeparator()
        add("Exit", self.close)

    def _build_mac_menu(self) -> None:
        """macOS: no in-window toolbar — the commands live in the system menu
        bar at the top of the screen (File menu), with the usual ⌘ shortcuts
        and "…" on commands that open a dialog. Quit is in the app menu."""
        file_menu = self.menuBar().addMenu("File")

        def add(text: str, handler, shortcut: str) -> None:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            file_menu.addAction(action)

        add("New Project", self._new_project, "Ctrl+N")
        add("Open Project…", self._open_project, "Ctrl+O")
        file_menu.addSeparator()
        add("Save", self._save_project, "Ctrl+S")
        add("Save As…", self._save_project_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        add("Export Maps…", self._export_maps, "Ctrl+E")
        file_menu.insertMenu(file_menu.actions()[2], self._build_recent_menu())

        quit_action = QAction("Quit PBRCELAIN", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _build_windows_commands(self) -> None:
        """Windows: a menu bar (File, with Open Recent and Exit) plus a Fluent
        command bar with icon + label buttons for the common commands."""
        from ui import windows_theme

        file_menu = self.menuBar().addMenu("&File")
        toolbar = QToolBar("Commands")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        iconed: list[tuple[QAction, str]] = []

        def add(text: str, handler, shortcut: str, icon: Optional[str] = None, short_label: Optional[str] = None) -> QAction:
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            file_menu.addAction(action)
            if icon:
                action.setIconText(short_label or text.replace("&", "").rstrip("…"))
                toolbar.addAction(action)
                iconed.append((action, icon))
            return action

        add("&New Project", self._new_project, "Ctrl+N", "new", "New")
        add("&Open Project…", self._open_project, "Ctrl+O", "open", "Open")
        file_menu.addMenu(self._build_recent_menu())
        file_menu.addSeparator()
        toolbar.addSeparator()
        add("&Save", self._save_project, "Ctrl+S", "save")
        add("Save &As…", self._save_project_as, "Ctrl+Shift+S", "save_as", "Save As")
        file_menu.addSeparator()
        toolbar.addSeparator()
        add("&Export Maps…", self._export_maps, "Ctrl+E", "export", "Export Maps")
        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        def refresh_icons() -> None:
            for action, name in iconed:
                action.setIcon(windows_theme.themed_icon(name))

        refresh_icons()
        theme_changed.connect(refresh_icons)

    def _build_recent_menu(self) -> QMenu:
        menu = QMenu("Open &Recent", self)

        def populate() -> None:
            menu.clear()
            paths = [p for p in self._recent_projects() if os.path.isfile(p)]
            for path in paths:
                action = menu.addAction(os.path.basename(path))
                action.setToolTip(path)
                action.triggered.connect(lambda _checked=False, p=path: self._handle_dropped_project(p))
            if not paths:
                menu.addAction("No Recent Projects").setEnabled(False)
            else:
                menu.addSeparator()
                menu.addAction("Clear Menu").triggered.connect(lambda: QSettings().remove(RECENT_PROJECTS_KEY))

        menu.aboutToShow.connect(populate)
        return menu

    @staticmethod
    def _recent_projects() -> list[str]:
        value = QSettings().value(RECENT_PROJECTS_KEY, [])
        if isinstance(value, str):
            value = [value]
        return [str(p) for p in (value or [])]

    def _add_recent_project(self, path: str) -> None:
        path = os.path.abspath(path)
        paths = [p for p in self._recent_projects() if os.path.normcase(p) != os.path.normcase(path)]
        QSettings().setValue(RECENT_PROJECTS_KEY, [path, *paths][:MAX_RECENT_PROJECTS])

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)
        self._main_splitter = splitter

        splitter.addWidget(self._build_project_section())
        splitter.addWidget(self._build_preview_section())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        if PLATFORM_THEME == "legacy":
            splitter.setSizes([400, 1040])
        else:
            splitter.setChildrenCollapsible(False)
            splitter.widget(0).setMinimumWidth(pick(mac=440, windows=420))
            splitter.setSizes([520, 920])

    def _build_project_section(self) -> QWidget:
        section = QWidget()
        section.setObjectName("ProjectSection")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if IS_MAC:
            title_space = QWidget()
            title_space.setFixedHeight(mac_window.TITLE_BAR_HEIGHT)
            mac_window.install_drag_area(title_space)
            layout.addWidget(title_space)

        heading = QLabel(_heading("MATERIAL SLOTS"))
        heading.setProperty("role", "panel-heading")
        layout.addWidget(heading)

        inspector = QWidget()
        inspector.setObjectName("MapInspector")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(*pick(mac=(14, 6, 14, 10), windows=(16, 6, 16, 12), legacy=(10, 10, 10, 10)))
        inspector_layout.addLayout(self._build_map_selector())
        layout.addWidget(inspector)

        properties_heading = QLabel(_heading("MAP SETTINGS"))
        properties_heading.setProperty("role", "panel-heading")
        layout.addWidget(properties_heading)
        layout.addWidget(self._build_dynamic_panel_host(), 1)

        controls = QWidget()
        controls.setObjectName("MapInspector")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(*pick(mac=(14, 10, 14, 12), windows=(16, 10, 16, 14), legacy=(10, 8, 10, 10)))
        controls_layout.addLayout(self._build_generate_controls())
        layout.addWidget(controls)
        return section

    def _build_map_selector(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(8)

        slots_bar = QHBoxLayout()
        slots_bar.setSpacing(4)
        for map_type in MAP_TYPE_ORDER:
            btn = SlotButton()
            btn.setProperty("role", "slot-btn")
            btn.setCheckable(True)
            btn.setMinimumHeight(44)
            btn.setToolTip(f"Click to select {MAP_TYPE_LABELS[map_type]}, double-click to open 2x2 Tiled Preview")
            btn.clicked.connect(lambda _checked=False, mt=map_type: self._select_map_type(mt))
            btn.doubleClicked.connect(lambda mt=map_type: (self._select_map_type(mt), self._open_tiled_preview(mt)))
            self._slot_buttons[map_type] = btn
            slots_bar.addWidget(btn)
        col.addLayout(slots_bar)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.selected_map_preview = ImageLabel("No map selected")
        self.selected_map_preview.setFixedSize(140, 140)
        self.selected_map_preview.imageDropped.connect(self._on_preview_image_dropped)
        self.selected_map_preview.fileDropped.connect(self._on_other_file_dropped)
        self.selected_map_preview.doubleClicked.connect(self._open_tiled_preview)
        row.addWidget(self.selected_map_preview, 0)

        controls = QVBoxLayout()
        controls.setSpacing(6)

        self.active_slot_label = QLabel(MAP_TYPE_LABELS[MapType.ALBEDO])
        legacy_style(self.active_slot_label, "background-color: transparent; font-weight: bold; font-size: 13px; color: #ffffff;", "slot-title")

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self._upload_image)

        self.save_as_btn = QPushButton("Save Map As…")
        self.save_as_btn.clicked.connect(self._save_current_map_as)

        self.remove_btn = QPushButton("Remove Image")
        self.remove_btn.clicked.connect(self._remove_image)

        controls.addWidget(self.active_slot_label)
        controls.addWidget(self.upload_btn)
        controls.addWidget(self.save_as_btn)
        controls.addWidget(self.remove_btn)
        controls.addStretch(1)

        row.addLayout(controls)
        col.addLayout(row)
        return col

    def _refresh_material_slots_bar(self) -> None:
        for map_type, btn in self._slot_buttons.items():
            btn.setChecked(map_type == self.current_map_type)
            name = MAP_TYPE_LABELS[map_type]
            if map_type in self._draft_images:
                status = "⚡ Draft"
            elif map_type in self.generated_images:
                status = "⚙ Gen"
            elif map_type in self.source_images:
                status = "● Loaded"
            else:
                status = "○ Empty"
            btn.setText(f"{name}\n{status}")

    def _build_dynamic_panel_host(self) -> QWidget:
        host = QWidget()
        host.setObjectName("DynamicPanelHost")
        layout = QVBoxLayout(host)

        self.stacked_panels = QStackedWidget()
        self.height_panel = HeightPanel()
        self.albedo_panel = AlbedoPanel()
        self.normal_panel = NormalPanel()
        self.roughness_panel = RoughnessPanel()
        self.ao_panel = AOPanel()

        self._panels_by_type = {
            MapType.ALBEDO: self.albedo_panel,
            MapType.HEIGHT: self.height_panel,
            MapType.NORMAL: self.normal_panel,
            MapType.ROUGHNESS: self.roughness_panel,
            MapType.AO: self.ao_panel,
        }
        for map_type in MAP_TYPE_ORDER:
            self.stacked_panels.addWidget(self._panels_by_type[map_type])
        layout.addWidget(self.stacked_panels)

        self.height_panel.set_raw_source_provider(self._height_raw_source)
        self.roughness_panel.set_underlay_provider(self._roughness_underlays)
        self.height_panel.modelSettingsChanged.connect(self._on_model_settings_changed)
        self.height_panel.loadModelRequested.connect(self._load_model)
        for map_type in MAP_TYPE_ORDER:
            self._panels_by_type[map_type].pageChanged.connect(self._on_panel_page_changed)
            self._panels_by_type[map_type].propertiesChanged.connect(self._schedule_properties_preview)
            self._panels_by_type[map_type].propertiesChanged.connect(self._update_save_revert_enabled)
        return host

    def _build_generate_controls(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.generate_btn = QPushButton("Generate Map")
        self.generate_btn.setProperty("role", "generate")
        self.generate_btn.clicked.connect(self._on_primary_btn_clicked)

        self.kill_btn = QPushButton("Kill Process")
        self.kill_btn.setProperty("role", "generate")
        self.kill_btn.setEnabled(False)
        self.kill_btn.clicked.connect(self._on_secondary_btn_clicked)

        btn_row.addWidget(self.generate_btn)
        btn_row.addWidget(self.kill_btn)
        col.addLayout(btn_row)

        status_row = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_bar)
        col.addLayout(status_row)
        return col

    def _build_preview_section(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("PreviewSection")
        splitter.setChildrenCollapsible(False)

        top_pane = QWidget()
        top_layout = QVBoxLayout(top_pane)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        viewport_heading = QLabel(_heading("3D VIEWPORT"))
        viewport_heading.setProperty("role", "viewport-label")
        if IS_MAC:
            viewport_heading.setFixedHeight(mac_window.TITLE_BAR_HEIGHT)
            mac_window.install_drag_area(viewport_heading)
        top_layout.addWidget(viewport_heading)

        self.preview_3d = Preview3DWidget()
        self.preview_3d.setMinimumHeight(160)
        viewport = QWidget()
        viewport.setObjectName("ViewportFrame")
        viewport_layout = QVBoxLayout(viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.addWidget(self.preview_3d)
        top_layout.addWidget(viewport, 1)

        bottom_pane = QWidget()
        bottom_layout = QVBoxLayout(bottom_pane)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        preview_heading = QLabel(_heading("VIEW OPTIONS"))
        preview_heading.setProperty("role", "panel-heading")
        bottom_layout.addWidget(preview_heading)

        self.preview_properties_panel = PreviewPropertiesPanel()
        self.preview_properties_panel.setMinimumHeight(120)
        self.preview_properties_panel.settingsChanged.connect(self._on_preview_settings_changed)

        self.preview_3d.distanceChanged.connect(self.preview_properties_panel.set_distance_value)
        self.preview_3d.lightChanged.connect(self.preview_properties_panel.set_light_values)
        self.preview_3d.autoRotateToggled.connect(self.preview_properties_panel.set_auto_rotate)
        self.preview_3d.pivotChanged.connect(self.preview_properties_panel.set_pivot_values)

        self.preview_3d.set_preview_settings(self.preview_properties_panel.get_settings())
        bottom_layout.addWidget(self.preview_properties_panel, 1)

        splitter.addWidget(top_pane)
        splitter.addWidget(bottom_pane)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([580, 240])
        self._preview_splitter = splitter
        return splitter

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._initial_layout_done or PLATFORM_THEME == "legacy":
            return
        self._initial_layout_done = True
        if not window_layout.restore(self, self._main_splitter, self._preview_splitter):
            window_layout.fit_to_screen(
                self, self._main_splitter, self._preview_splitter,
                sidebar_ratio=pick(mac=0.34, windows=0.32), sidebar_min=pick(mac=440, windows=420), sidebar_max=640,
            )
        if IS_WINDOWS:
            from ui.win_window import TaskbarProgress

            self._taskbar_progress = TaskbarProgress(self)

    def _select_map_type(self, map_type: MapType) -> None:
        self._invalidate_properties_preview()
        self.current_map_type = map_type
        self.active_slot_label.setText(MAP_TYPE_LABELS[map_type])
        panel = self._panels_by_type[map_type]
        self.stacked_panels.setCurrentWidget(panel)
        self._refresh_selected_map_preview()
        self._refresh_material_slots_bar()
        self._update_generate_bar_mode(panel.current_page())

    def _update_map_controls_enabled(self) -> None:
        map_type = self.current_map_type
        self.upload_btn.setEnabled(True)
        has_content = (
            map_type in self.source_images
            or map_type in self.generated_images
            or map_type in self._draft_images
        )
        self.remove_btn.setEnabled(has_content)
        self.save_as_btn.setEnabled(has_content)

    def _active_display_image(self, map_type: MapType) -> Optional[Image.Image]:
        return (
            self._draft_images.get(map_type)
            or self.generated_images.get(map_type)
            or self.source_images.get(map_type)
        )

    def _refresh_selected_map_preview(self) -> None:
        image = self._active_display_image(self.current_map_type)
        self.selected_map_preview.set_pixmap_source(pil_to_qpixmap(image) if image is not None else None)
        if image is not None:
            self.selected_map_preview.setToolTip(
                f"Double-click to open 2x2 Tiled Preview ({MAP_TYPE_LABELS.get(self.current_map_type, 'Map')})"
            )
        self.height_panel.set_has_albedo(MapType.ALBEDO in self.source_images)
        self.normal_panel.set_has_height(self._active_display_image(MapType.HEIGHT) is not None)
        self.roughness_panel.set_has_normal(self._active_display_image(MapType.NORMAL) is not None)
        self.ao_panel.set_has_height(self._active_display_image(MapType.HEIGHT) is not None)
        self._update_map_controls_enabled()
        self._refresh_material_slots_bar()

    def _open_tiled_preview(self, target_map_type: Optional[MapType] = None) -> None:
        map_type = target_map_type or self.current_map_type
        image = (
            self.generated_images.get(map_type)
            or self._draft_images.get(map_type)
            or self.source_images.get(map_type)
        )
        if image is None:
            return
        map_name = MAP_TYPE_LABELS.get(map_type, "Active Map")
        dlg = TiledPreviewDialog(image, map_name=map_name, parent=self)
        dlg.exec()

    def _on_preview_image_dropped(self, path: str) -> None:
        self._load_source_image(self.current_map_type, path)

    def _on_other_file_dropped(self, path: str) -> None:
        if path.lower().endswith(PROJECT_EXTENSION):
            self._handle_dropped_project(path)

    def _upload_image(self) -> None:
        filt = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", filt)
        if path:
            self._load_source_image(self.current_map_type, path)

    def _get_dependent_downstream_maps(self, map_type: MapType) -> list[MapType]:
        """Return which existing maps depend on this map type."""
        if map_type == MapType.ALBEDO:
            candidates = (MapType.HEIGHT, MapType.NORMAL, MapType.ROUGHNESS, MapType.AO)
        elif map_type == MapType.HEIGHT:
            candidates = (MapType.NORMAL, MapType.ROUGHNESS, MapType.AO)
        elif map_type == MapType.NORMAL:
            candidates = (MapType.ROUGHNESS,)
        else:
            candidates = ()
        return [m for m in candidates if (m in self.generated_images or m in self.source_images)]

    def _load_source_image(self, map_type: MapType, path: str) -> None:
        if not path.lower().endswith(IMAGE_EXTENSIONS):
            return
        try:
            image = Image.open(path)
            image.load()
            if map_type in (MapType.HEIGHT, MapType.ROUGHNESS) and image.mode in ("I;16", "I", "L"):
                pass
            else:
                image = image.convert("RGB")
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load image", str(exc))
            return

        affected = self._get_dependent_downstream_maps(map_type)

        self.source_images[map_type] = image
        self.generated_images.pop(map_type, None)
        self._draft_images.pop(map_type, None)
        if map_type == MapType.HEIGHT:
            self._height_depth_result = None

        self._mark_dirty()
        if self.current_map_type == map_type:
            self._refresh_selected_map_preview()
        else:
            self._refresh_material_slots_bar()
        self._push_textures_to_preview()

        if affected:
            affected_names = ", ".join(MAP_TYPE_LABELS[m] for m in affected)
            self.status_label.setText(f"{affected_names} maps may be out of sync with your changes.")
        else:
            self.status_label.setText(f"{MAP_TYPE_LABELS[map_type]} loaded.")

    def _remove_image(self) -> None:
        map_type = self.current_map_type
        has_content = (
            map_type in self.source_images
            or map_type in self.generated_images
            or map_type in self._draft_images
        )
        if not has_content:
            return

        reply = QMessageBox.question(
            self,
            f"Remove {MAP_TYPE_LABELS[map_type]} Map?",
            f"Are you sure you want to remove the {MAP_TYPE_LABELS[map_type]} map?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        affected = self._get_dependent_downstream_maps(map_type)

        self.source_images.pop(map_type, None)
        self.generated_images.pop(map_type, None)
        self._draft_images.pop(map_type, None)
        if map_type == MapType.HEIGHT:
            self._height_depth_result = None

        self._mark_dirty()
        self._refresh_selected_map_preview()
        self._refresh_material_slots_bar()
        self._push_textures_to_preview()

        if affected:
            affected_names = ", ".join(MAP_TYPE_LABELS[m] for m in affected)
            msg = f"{affected_names} maps may be out of sync with your changes."
            QMessageBox.information(self, "Warning", msg)
            self.status_label.setText(f"{affected_names} maps may be out of sync with your changes.")
        else:
            self.status_label.setText(f"{MAP_TYPE_LABELS[map_type]} removed.")

    def _save_current_map_as(self) -> None:
        map_type = self.current_map_type
        image = self._active_display_image(map_type)
        if image is None:
            QMessageBox.information(self, "Nothing to save", "This map has no image yet.")
            return

        is_32bit = (image.mode == "F")
        if is_32bit:
            default_name = f"{map_type.value}.exr"
            filter_str = "OpenEXR (*.exr);;TIFF Image (*.tif *.tiff);;All Files (*.*)"
        elif image.mode in ("I;16", "I"):
            default_name = f"{map_type.value}.png"
            filter_str = "PNG Image (*.png);;TIFF Image (*.tif *.tiff);;OpenEXR (*.exr);;All Files (*.*)"
        else:
            default_name = f"{map_type.value}.png"
            filter_str = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;TIFF Image (*.tif *.tiff);;OpenEXR (*.exr);;All Files (*.*)"

        path, _ = QFileDialog.getSaveFileName(self, "Save Map As", default_name, filter_str)
        if not path:
            return
        if is_32bit and not (path.lower().endswith((".exr", ".tif", ".tiff"))):
            path += ".exr"
        elif not is_32bit and not path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")):
            path += ".png"
        dialog = OperationProgressDialog("Saving Map", f"Writing {os.path.basename(path)}…", parent=self)
        worker = SingleImageSaveWorker(image, path, parent=self)

        saved = [False]
        def on_done(_p):
            saved[0] = True
            dialog.accept()

        def on_err(err):
            dialog.reject()
            QMessageBox.critical(self, "Failed to save image", str(err))

        worker.finished_ok.connect(on_done)
        worker.failed.connect(on_err)
        worker.start()
        dialog.exec()

        if saved[0]:
            self.status_label.setText(f"Saved {MAP_TYPE_LABELS[map_type]} map to {path}")

    def _is_model_ready_and_matched(self) -> bool:
        if self.estimator is None or not self.estimator.is_loaded():
            return False
        family, variant = self.height_panel.selected_family_variant()
        device = self.height_panel.selected_device()
        device_matches = (
            device == getattr(self.estimator, "requested_device", None)
            or device == self.estimator.device
            or (device == "auto" and getattr(self.estimator, "requested_device", None) == "auto")
        )
        return (
            self.estimator.family == family
            and self.estimator.variant == variant
            and device_matches
        )

    def _replace_estimator(self, family: str, variant: str, device: str) -> None:
        """Free the previously loaded model before creating a new one, otherwise
        switching models keeps both resident in (V)RAM until garbage collection."""
        if self.estimator is not None:
            self.estimator.unload()
        self.estimator = DepthEstimator(family, variant, device)

    def _on_model_settings_changed(self) -> None:
        """Called when family, variant, or device combo changes in the UI."""
        family, variant = self.height_panel.selected_family_variant()
        device = self.height_panel.selected_device()

        if self.estimator is not None and self.estimator.is_loaded():
            if self._is_model_ready_and_matched():
                self.height_panel.set_model_status(
                    f"Loaded: {self.estimator.family} / {self.estimator.variant} on {self.estimator.device.upper()}"
                )
            else:
                self.height_panel.set_model_status(
                    f"Loaded: {self.estimator.family} / {self.estimator.variant} on {self.estimator.device.upper()} "
                    f"— selected: {family} / {variant} on {device.upper()} (click 'Load Model' to switch)"
                )
        else:
            self.height_panel.set_model_status("No model loaded — click 'Load Model'.")

    def _load_model(self) -> None:
        if self.model_thread is not None and self.model_thread.isRunning():
            return
        family, variant = self.height_panel.selected_family_variant()
        device = self.height_panel.selected_device()

        if self._is_model_ready_and_matched():
            self.height_panel.set_model_status(
                f"Loaded: {self.estimator.family} / {self.estimator.variant} on {self.estimator.device.upper()}"
            )
            return

        self._replace_estimator(family, variant, device)
        self.height_panel.set_model_controls_enabled(False)
        self.generate_btn.setEnabled(False)
        self.height_panel.set_model_status("Loading…")

        dialog = OperationProgressDialog(
            "Loading Depth Model",
            f"Initializing {family} ({variant}) on {device.upper()}…",
            parent=self,
        )
        self.model_thread = ModelLoadWorker(self.estimator)
        self.model_thread.progress.connect(lambda msg: (dialog.set_status(msg), self.height_panel.set_model_status(msg)))

        def on_loaded():
            dialog.accept()
            self._on_model_loaded()

        def on_failed(err):
            dialog.reject()
            self._on_model_load_failed(err)

        self.model_thread.finished_ok.connect(on_loaded)
        self.model_thread.failed.connect(on_failed)
        self.model_thread.start()
        dialog.exec()

    def _on_model_loaded(self) -> None:
        self.height_panel.set_model_controls_enabled(True)
        self.generate_btn.setEnabled(True)
        if self.estimator is None or not self.estimator.is_loaded():
            self.height_panel.set_model_status("No model loaded — click 'Load Model'.")
            return
        self.height_panel.set_model_status(
            f"Loaded: {self.estimator.family} / {self.estimator.variant} on {self.estimator.device.upper()}"
        )

    def _on_model_loaded_for_generate_all(self) -> None:
        self._on_model_loaded()
        if self._generating_all:
            self._start_height_inference()

    def _on_model_loaded_for_generate_height(self) -> None:
        self._on_model_loaded()
        self._start_height_inference()

    def _on_model_load_failed(self, msg: str) -> None:
        self._generating_all = False
        self.height_panel.set_model_controls_enabled(True)
        self.height_panel.set_model_status("Failed to load model.")
        self._set_busy(False)
        if self.estimator is not None:
            self.estimator.unload()
        self.estimator = None
        QMessageBox.critical(self, "Model load failed", msg)

    def _height_raw_source(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Raw input for Height's Properties post-processing."""
        if self._height_depth_result is not None:
            return self._height_depth_result.depth, self._height_depth_result.guide
        src = self.source_images.get(MapType.HEIGHT)
        if src is not None:
            return _image_to_height_array(src), None
        return None, None

    def _roughness_underlays(self) -> tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image]]:
        """Provides reference textures for the Roughness Imperfection Painter."""
        return (
            self._active_display_image(MapType.ALBEDO),
            self._active_display_image(MapType.NORMAL),
            self._active_display_image(MapType.ROUGHNESS),
        )

    def _height_seamless_orig_size(self) -> Optional[tuple[int, int]]:
        """Original tile size if the current raw depth came from a seamless (padded) run."""
        if self._height_depth_result is None:
            return None
        return self._height_depth_result.seamless_orig_size

    def _apply_height_result(self, depth: np.ndarray, guide: Optional[np.ndarray]) -> np.ndarray:
        """Recompute the committed Height map from a raw depth field."""
        options = self.height_panel.current_options()
        height_normed = build_height_map(depth, options, guide=guide)

        seamless_size = self._height_seamless_orig_size()
        if seamless_size is not None:
            ehm, raw1, mask_map = apply_seamless_linear_mask(
                height_normed,
                orig_size=seamless_size,
                padding_pct=0.10,
            )
            height_normed = raw1

        height_img = height_map_to_image(height_normed, bit_depth=options.bit_depth)
        self.generated_images[MapType.HEIGHT] = height_img
        self._draft_images.pop(MapType.HEIGHT, None)

        if MapType.NORMAL in self.generated_images:
            self._generate_normal_from_height(height_normed, seamless=True)
        if MapType.AO in self.generated_images:
            self._generate_ao_from_height(height_normed, commit=True)

        if self.current_map_type in (MapType.HEIGHT, MapType.NORMAL, MapType.AO):
            self._refresh_selected_map_preview()
        self._refresh_material_slots_bar()
        self._push_textures_to_preview()
        return height_normed

    def _generate_normal_from_height(self, height_normed: np.ndarray, seamless: bool = True, commit: bool = True) -> None:
        normal_options = replace(self.normal_panel.current_options(), seamless=seamless)
        albedo_src = self._active_display_image(MapType.ALBEDO)
        if albedo_src is not None:
            normal_options.albedo_source = albedo_src
        normal_arr = generate_normal_map(height_normed, normal_options)
        img = normal_map_to_image(normal_arr)
        if commit:
            self.generated_images[MapType.NORMAL] = img
            self._draft_images.pop(MapType.NORMAL, None)
            if MapType.ROUGHNESS in self.generated_images:
                self._generate_roughness_from_normal(normal_arr, commit=True)
        else:
            self._draft_images[MapType.NORMAL] = img
            if MapType.ROUGHNESS in self.generated_images or MapType.ROUGHNESS in self._draft_images:
                self._generate_roughness_from_normal(normal_arr, commit=False)

    def _generate_roughness_from_normal(self, normal_arr: np.ndarray, commit: bool = True) -> None:
        roughness_options = self.roughness_panel.current_options()
        height_img = self._active_display_image(MapType.HEIGHT)
        if height_img is not None:
            roughness_options.height_source = _image_to_height_array(height_img)
        albedo_img = self._active_display_image(MapType.ALBEDO)
        if albedo_img is not None:
            roughness_options.albedo_source = albedo_img
        roughness_arr = generate_roughness_map(normal_arr, roughness_options)
        img = roughness_map_to_image(roughness_arr)
        if commit:
            self.generated_images[MapType.ROUGHNESS] = img
            self._draft_images.pop(MapType.ROUGHNESS, None)
        else:
            self._draft_images[MapType.ROUGHNESS] = img

    def _generate_ao_from_height(self, height_normed: np.ndarray, commit: bool = True) -> None:
        ao_options = self.ao_panel.current_options()
        ao_arr = generate_ao_map(height_normed, ao_options)
        img = ao_map_to_image(ao_arr, bit_depth=ao_options.bit_depth)
        if commit:
            self.generated_images[MapType.AO] = img
            self._draft_images.pop(MapType.AO, None)
        else:
            self._draft_images[MapType.AO] = img

    def _schedule_properties_preview(self) -> None:
        panel = self._panels_by_type[self.current_map_type]
        if panel.current_page() != "properties":
            return
        self._properties_preview_revision += 1
        self._properties_preview_pending = True
        self._properties_debounce.start()

    def _refresh_properties_draft(self) -> None:
        if not self._properties_preview_pending:
            return
        if self.properties_preview_thread is not None and self.properties_preview_thread.isRunning():
            return

        self._properties_preview_pending = False
        revision = self._properties_preview_revision
        map_type = self.current_map_type
        has_normal = MapType.NORMAL in self.generated_images or MapType.NORMAL in self._draft_images
        has_roughness = MapType.ROUGHNESS in self.generated_images or MapType.ROUGHNESS in self._draft_images
        has_ao = MapType.AO in self.generated_images or MapType.AO in self._draft_images

        if map_type == MapType.HEIGHT:
            depth, guide = self._height_raw_source()
            if depth is None:
                return
            worker = PropertiesPreviewWorker(
                revision, map_type.value,
                height_depth=depth,
                height_guide=guide,
                height_seamless_orig_size=self._height_seamless_orig_size(),
                height_options=self.height_panel.current_options(),
                normal_options=replace(self.normal_panel.current_options(), seamless=True),
                roughness_options=self.roughness_panel.current_options(),
                ao_options=self.ao_panel.current_options(),
                make_normal=has_normal,
                make_roughness=has_normal and has_roughness,
                make_ao=has_ao,
            )
        elif map_type == MapType.NORMAL:
            height_image = self._active_display_image(MapType.HEIGHT)
            if height_image is None:
                return
            normal_opts = replace(self.normal_panel.current_options(), seamless=True)
            normal_opts.albedo_source = self._active_display_image(MapType.ALBEDO)
            worker = PropertiesPreviewWorker(
                revision, map_type.value,
                height_image=height_image,
                normal_options=normal_opts,
                roughness_options=self.roughness_panel.current_options(),
                make_roughness=has_roughness,
            )
        elif map_type == MapType.ROUGHNESS:
            normal_image = self._active_display_image(MapType.NORMAL)
            if normal_image is None:
                return
            rough_opts = self.roughness_panel.current_options()
            height_img = self._active_display_image(MapType.HEIGHT)
            if height_img is not None:
                rough_opts.height_source = _image_to_height_array(height_img)
            rough_opts.albedo_source = self._active_display_image(MapType.ALBEDO)
            worker = PropertiesPreviewWorker(
                revision, map_type.value,
                normal_image=normal_image,
                roughness_options=rough_opts,
                make_roughness=True,
            )
        elif map_type == MapType.AO:
            height_image = self._active_display_image(MapType.HEIGHT)
            if height_image is None:
                return
            worker = PropertiesPreviewWorker(
                revision, map_type.value,
                height_image=height_image,
                ao_options=self.ao_panel.current_options(),
            )
        elif map_type == MapType.ALBEDO:
            albedo_src = self.source_images.get(MapType.ALBEDO)
            if albedo_src is None:
                return
            worker = PropertiesPreviewWorker(
                revision, map_type.value,
                albedo_image=albedo_src,
                albedo_shift_rules=self.albedo_panel.current_shift_rules(),
                albedo_options=self.albedo_panel.current_adjust_options(),
            )
        else:
            return

        self.properties_preview_thread = worker
        worker.result_ready.connect(self._on_properties_preview_ready)
        worker.failed.connect(self._on_properties_preview_failed)
        worker.finished.connect(lambda: self._on_properties_preview_finished(worker))
        worker.start()

    def _on_properties_preview_ready(self, revision: int, map_type: str, images: dict[str, Image.Image]) -> None:
        if revision != self._properties_preview_revision or map_type != self.current_map_type.value:
            return
        for name, image in images.items():
            self._draft_images[MapType(name)] = image
        self._refresh_selected_map_preview()
        self._push_textures_to_preview()

    def _on_properties_preview_failed(self, revision: int, message: str) -> None:
        if revision == self._properties_preview_revision:
            self.status_label.setText(f"Preview update failed: {message}")

    def _on_properties_preview_finished(self, worker: PropertiesPreviewWorker) -> None:
        if self.properties_preview_thread is worker:
            self.properties_preview_thread = None
        worker.deleteLater()
        if self._properties_preview_pending:
            self._properties_debounce.start(0)

    def _invalidate_properties_preview(self) -> None:
        self._properties_preview_revision += 1
        self._properties_preview_pending = False
        self._properties_debounce.stop()

    def _generate_all_maps(self) -> None:
        albedo_source = self.source_images.get(MapType.ALBEDO)
        if albedo_source is None:
            QMessageBox.warning(
                self,
                "Albedo required",
                "Please upload an Albedo image first before generating maps.",
            )
            return
        if self.infer_thread is not None and self.infer_thread.isRunning():
            QMessageBox.information(self, "Busy", "Generation already running, please wait.")
            return

        self._generating_all = True

        if not self._is_model_ready_and_matched():
            self.status_label.setText("Loading depth model for generation…")
            self._set_busy(True)
            family, variant = self.height_panel.selected_family_variant()
            device = self.height_panel.selected_device()
            self._replace_estimator(family, variant, device)

            self.height_panel.set_model_controls_enabled(False)
            dialog = OperationProgressDialog(
                "Loading Depth Model",
                f"Initializing {family} ({variant}) on {device.upper()}…",
                parent=self,
            )
            self.model_thread = ModelLoadWorker(self.estimator, parent=self)
            self.model_thread.progress.connect(lambda msg: (dialog.set_status(msg), self.height_panel.set_model_status(msg)))

            def on_loaded():
                dialog.accept()
                self._on_model_loaded_for_generate_all()

            def on_failed(err):
                dialog.reject()
                self._on_model_load_failed(err)

            self.model_thread.finished_ok.connect(on_loaded)
            self.model_thread.failed.connect(on_failed)
            self.model_thread.start()
            dialog.exec()
            return

        self._start_height_inference()

    def _generate_current_map(self) -> None:
        map_type = self.current_map_type
        if map_type == MapType.ALBEDO:
            self._generate_albedo_map_only()
            return
        if map_type == MapType.ROUGHNESS:
            self._generate_roughness_map_only()
            return
        if map_type == MapType.NORMAL:
            self._generate_normal_map_only()
            return
        if map_type == MapType.AO:
            self._generate_ao_map_only()
            return
        self._generate_height_map()

    def _generate_albedo_map_only(self) -> None:
        albedo_src = self.source_images.get(MapType.ALBEDO)
        if albedo_src is None:
            QMessageBox.information(
                self,
                "Albedo required",
                "Upload an Albedo base texture first.",
            )
            return
        opts = self.albedo_panel.current_adjust_options()
        rules = self.albedo_panel.current_shift_rules()
        result_img = apply_albedo_adjustments(albedo_src, options=opts, rules=rules)
        self.generated_images[MapType.ALBEDO] = result_img
        self._draft_images.pop(MapType.ALBEDO, None)
        self._refresh_selected_map_preview()
        self._push_textures_to_preview()
        self._mark_dirty()
        self.status_label.setText("Albedo map updated with seamless & color adjustments.")

    def _generate_ao_map_only(self) -> None:
        height_image = self._active_display_image(MapType.HEIGHT)
        if height_image is None:
            QMessageBox.information(
                self,
                "Height required",
                "Upload or generate a Height map first — AO is derived from it.",
            )
            return
        height_arr = _image_to_height_array(height_image)
        self._generate_ao_from_height(height_arr, commit=True)
        self._refresh_selected_map_preview()
        self._push_textures_to_preview()
        self._properties_snapshot[MapType.AO] = self.ao_panel.properties_to_dict()
        self._mark_dirty()
        self.status_label.setText("Ambient Occlusion map generated from Height.")

    def _generate_roughness_map_only(self) -> None:
        normal_image = self._active_display_image(MapType.NORMAL)
        if normal_image is None:
            QMessageBox.information(
                self,
                "Normal required",
                "Upload or generate a Normal map first — Roughness is derived from it.",
            )
            return
        self._generate_roughness_from_normal(_image_to_normal_array(normal_image))
        self._refresh_selected_map_preview()
        self._push_textures_to_preview()
        self._properties_snapshot[MapType.ROUGHNESS] = self.roughness_panel.properties_to_dict()
        self._mark_dirty()
        self.status_label.setText("Roughness map generated from Normal.")

    def _generate_normal_map_only(self) -> None:
        height_image = self._active_display_image(MapType.HEIGHT)
        if height_image is None:
            QMessageBox.information(
                self,
                "Height required",
                "Upload or generate a Height map first — Normal is derived from it.",
            )
            return
        height_arr = _image_to_height_array(height_image)
        self._generate_normal_from_height(height_arr, seamless=True)
        self._refresh_selected_map_preview()
        self._push_textures_to_preview()
        self._properties_snapshot[MapType.NORMAL] = self.normal_panel.properties_to_dict()
        self._mark_dirty()
        self.status_label.setText("Normal map generated from Height.")

    def _generate_height_map(self) -> None:
        albedo_source = self.source_images.get(MapType.ALBEDO)
        if albedo_source is None:
            QMessageBox.information(
                self,
                "Nothing to generate",
                "Height is generated from the Albedo source image — switch to "
                "the Albedo tab and upload an image first.",
            )
            return
        if not self._is_model_ready_and_matched():
            if self.model_thread is not None and self.model_thread.isRunning():
                QMessageBox.information(self, "Busy", "Model is loading, please wait.")
                return
            self.status_label.setText("Loading depth model for generation…")
            self._set_busy(True)
            family, variant = self.height_panel.selected_family_variant()
            device = self.height_panel.selected_device()
            self._replace_estimator(family, variant, device)

            self.height_panel.set_model_controls_enabled(False)
            dialog = OperationProgressDialog(
                "Loading Depth Model",
                f"Initializing {family} ({variant}) on {device.upper()}…",
                parent=self,
            )
            self.model_thread = ModelLoadWorker(self.estimator, parent=self)
            self.model_thread.progress.connect(lambda msg: (dialog.set_status(msg), self.height_panel.set_model_status(msg)))

            def on_loaded():
                dialog.accept()
                self._on_model_loaded_for_generate_height()

            def on_failed(err):
                dialog.reject()
                self._on_model_load_failed(err)

            self.model_thread.finished_ok.connect(on_loaded)
            self.model_thread.failed.connect(on_failed)
            self.model_thread.start()
            dialog.exec()
            return
        if self.infer_thread is not None and self.infer_thread.isRunning():
            QMessageBox.information(self, "Busy", "Already generating, please wait.")
            return

        self._start_height_inference()

    def _start_height_inference(self) -> None:
        albedo_source = self.source_images.get(MapType.ALBEDO)
        if albedo_source is None:
            return

        is_seamless = self.height_panel.is_seamless_generation()
        self._infer_is_seamless = is_seamless
        self._infer_orig_albedo_size = albedo_source.size

        infer_source = albedo_source
        if is_seamless:
            try:
                extended, tile_box = build_seamless_extended_tile(albedo_source, padding_pct=0.10)
                if self.height_panel.is_force_seamless_blur():
                    extended = apply_force_seamless_blur(extended, tile_box, blur_pct=0.03)
                infer_source = extended
            except Exception as exc:
                print(f"[Seamless] Failed to build extended tile: {exc}")
                infer_source = albedo_source
                self._infer_is_seamless = False

        tiled = self.height_panel.is_chunked()
        if tiled:
            reply = QMessageBox.question(
                self,
                "Chunk estimation enabled",
                "Chunk estimation runs the model many times (1 full-image pass + "
                "one pass per chunk) and can take significantly longer, especially "
                "on CPU. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._generating_all = False
                self._set_busy(False)
                return

        self._set_busy(True)
        self.status_label.setText("Running depth estimation…")
        self._height_before_infer = self.generated_images.get(MapType.HEIGHT)

        tile_size, overlap_pct, batch_size = self.height_panel.chunk_params()
        self.infer_thread = InferenceWorker(
            self.estimator,
            infer_source,
            tiled=tiled,
            tile_size=tile_size,
            overlap_pct=overlap_pct,
            batch_size=batch_size,
        )
        self.infer_thread.tile_progress.connect(self.status_label.setText)
        self.infer_thread.tile_preview.connect(self._on_tile_preview)
        self.infer_thread.result_ready.connect(self._on_depth_ready)
        self.infer_thread.failed.connect(self._on_infer_failed)
        self.infer_thread.finished_ok.connect(self._on_infer_finished)
        self.infer_thread.cancelled.connect(self._on_infer_cancelled)
        self.infer_thread.start()

    def _on_tile_preview(self, depth: object, guide: object) -> None:
        options = self.height_panel.current_options()
        preview_depth = downsample_for_preview(depth)
        normed = build_height_map(preview_depth, options, guide=guide)
        img = height_map_to_image(normed, bit_depth=8)
        self.generated_images[MapType.HEIGHT] = img
        if self.current_map_type == MapType.HEIGHT:
            self.selected_map_preview.set_pixmap_source(pil_to_qpixmap(img))
        self._refresh_material_slots_bar()

    def _on_depth_ready(self, result: DepthResult) -> None:
        self._height_before_infer = None
        if self._infer_is_seamless and self._infer_orig_albedo_size is not None:
            result.seamless_orig_size = self._infer_orig_albedo_size
        self._height_depth_result = result
        height_normed = self._apply_height_result(result.depth, result.guide)
        self._properties_snapshot[MapType.HEIGHT] = self.height_panel.properties_to_dict()

        if self._generating_all:
            if MapType.NORMAL not in self.source_images:
                self._generate_normal_from_height(height_normed, seamless=True, commit=True)
                self._properties_snapshot[MapType.NORMAL] = self.normal_panel.properties_to_dict()
            if MapType.ROUGHNESS not in self.source_images and MapType.NORMAL in self.generated_images:
                normal_img = self.generated_images[MapType.NORMAL]
                self._generate_roughness_from_normal(_image_to_normal_array(normal_img), commit=True)
                self._properties_snapshot[MapType.ROUGHNESS] = self.roughness_panel.properties_to_dict()
            if MapType.AO not in self.source_images:
                self._generate_ao_from_height(height_normed, commit=True)
                self._properties_snapshot[MapType.AO] = self.ao_panel.properties_to_dict()

        self._mark_dirty()
        self._refresh_selected_map_preview()
        self._refresh_material_slots_bar()
        self._push_textures_to_preview()

    def _restore_height_before_infer(self) -> None:
        previous = self._height_before_infer
        self._height_before_infer = None
        if previous is not None:
            self.generated_images[MapType.HEIGHT] = previous
        else:
            self.generated_images.pop(MapType.HEIGHT, None)
        self._refresh_selected_map_preview()
        self._push_textures_to_preview()

    def _on_infer_failed(self, msg: str) -> None:
        self._generating_all = False
        self._restore_height_before_infer()
        self._set_busy(False)
        self.status_label.setText(f"Failed: {msg}")
        QMessageBox.critical(self, "Generation failed", msg)

    def _on_infer_finished(self) -> None:
        self._set_busy(False)
        if self._generating_all:
            self._generating_all = False
            self.status_label.setText("All maps generated successfully.")
        else:
            self.status_label.setText("Height map generated.")

    def _on_infer_cancelled(self) -> None:
        self._generating_all = False
        self._restore_height_before_infer()
        self._set_busy(False)
        self.status_label.setText("Cancelled.")

    def _kill_process(self) -> None:
        self._generating_all = False
        if self.infer_thread is not None and self.infer_thread.isRunning():
            self.infer_thread.requestInterruption()
            self.status_label.setText("Cancelling…")
        if self.model_thread is not None and self.model_thread.isRunning():
            self.model_thread.requestInterruption()
            self.status_label.setText("Cancelling…")

    def _on_panel_page_changed(self, page: str) -> None:
        self._update_generate_bar_mode(page)

    def _reset_properties_snapshots(self) -> None:
        self._properties_snapshot = {
            map_type: self._panels_by_type[map_type].properties_to_dict() for map_type in MAP_TYPE_ORDER
        }

    def _update_generate_bar_mode(self, mode: str) -> None:
        self._bottom_bar_mode = mode
        if mode == "properties":
            self.generate_btn.setText("Save")
            self.kill_btn.setText("Revert")
            self._update_save_revert_enabled()
        else:
            self.generate_btn.setText("Generate Map")
            self.kill_btn.setText("Kill Process")
            busy = (
                (self.infer_thread is not None and self.infer_thread.isRunning())
                or (self.model_thread is not None and self.model_thread.isRunning())
            )
            self.generate_btn.setEnabled(not busy)
            self.kill_btn.setEnabled(busy)

    def _update_save_revert_enabled(self) -> None:
        if self._bottom_bar_mode != "properties":
            return
        map_type = self.current_map_type
        panel = self._panels_by_type[map_type]
        changed = panel.properties_to_dict() != self._properties_snapshot.get(map_type, {})
        self.generate_btn.setEnabled(changed)
        self.kill_btn.setEnabled(changed)

    def _on_primary_btn_clicked(self) -> None:
        if self._bottom_bar_mode == "properties":
            self._save_current_properties()
        else:
            self._generate_current_map()

    def _on_secondary_btn_clicked(self) -> None:
        if self._bottom_bar_mode == "properties":
            self._revert_current_properties()
        else:
            self._kill_process()

    def _bake_all_pending_properties(self) -> None:
        """Bake uncommitted property adjustments into full-resolution
        generated images before saving or exporting, so user tweaks are never lost."""
        self._invalidate_properties_preview()

        height_panel_dict = self.height_panel.properties_to_dict()
        if height_panel_dict != self._properties_snapshot.get(MapType.HEIGHT, {}) or MapType.HEIGHT in self._draft_images:
            depth, guide = self._height_raw_source()
            if depth is not None:
                self._apply_height_result(depth, guide)
                self._draft_images.pop(MapType.HEIGHT, None)
                self._properties_snapshot[MapType.HEIGHT] = height_panel_dict

        normal_panel_dict = self.normal_panel.properties_to_dict()
        if normal_panel_dict != self._properties_snapshot.get(MapType.NORMAL, {}) or MapType.NORMAL in self._draft_images:
            height_image = self._active_display_image(MapType.HEIGHT)
            if height_image is not None:
                self._generate_normal_from_height(_image_to_height_array(height_image), seamless=True, commit=True)
                self._draft_images.pop(MapType.NORMAL, None)
                self._properties_snapshot[MapType.NORMAL] = normal_panel_dict

        roughness_panel_dict = self.roughness_panel.properties_to_dict()
        if roughness_panel_dict != self._properties_snapshot.get(MapType.ROUGHNESS, {}) or MapType.ROUGHNESS in self._draft_images:
            normal_image = self._active_display_image(MapType.NORMAL)
            if normal_image is not None:
                self._generate_roughness_from_normal(_image_to_normal_array(normal_image), commit=True)
                self._draft_images.pop(MapType.ROUGHNESS, None)
                self._properties_snapshot[MapType.ROUGHNESS] = roughness_panel_dict

        albedo_panel_dict = self.albedo_panel.properties_to_dict()
        if albedo_panel_dict != self._properties_snapshot.get(MapType.ALBEDO, {}) or MapType.ALBEDO in self._draft_images:
            albedo_src = self.source_images.get(MapType.ALBEDO)
            if albedo_src is not None:
                shifted = apply_albedo_adjustments(
                    albedo_src,
                    options=self.albedo_panel.current_adjust_options(),
                    rules=self.albedo_panel.current_shift_rules(),
                )
                self.generated_images[MapType.ALBEDO] = shifted
                self._draft_images.pop(MapType.ALBEDO, None)
                self._properties_snapshot[MapType.ALBEDO] = albedo_panel_dict

        ao_panel_dict = self.ao_panel.properties_to_dict()
        if ao_panel_dict != self._properties_snapshot.get(MapType.AO, {}) or MapType.AO in self._draft_images:
            height_image = self._active_display_image(MapType.HEIGHT)
            if height_image is not None:
                self._generate_ao_from_height(_image_to_height_array(height_image), commit=True)
                self._draft_images.pop(MapType.AO, None)
                self._properties_snapshot[MapType.AO] = ao_panel_dict

        self._refresh_material_slots_bar()

    def _save_current_properties(self) -> None:
        self._invalidate_properties_preview()
        map_type = self.current_map_type
        if map_type == MapType.HEIGHT:
            depth, guide = self._height_raw_source()
            if depth is not None:
                self._apply_height_result(depth, guide)
                self._draft_images.pop(MapType.HEIGHT, None)
                self._refresh_selected_map_preview()
        elif map_type == MapType.NORMAL:
            height_image = self._active_display_image(MapType.HEIGHT)
            if height_image is not None:
                self._generate_normal_from_height(_image_to_height_array(height_image), seamless=True, commit=True)
                self._refresh_selected_map_preview()
                self._push_textures_to_preview()
        elif map_type == MapType.ROUGHNESS:
            normal_image = self._active_display_image(MapType.NORMAL)
            if normal_image is not None:
                self._generate_roughness_from_normal(_image_to_normal_array(normal_image), commit=True)
                self._refresh_selected_map_preview()
                self._push_textures_to_preview()
        elif map_type == MapType.AO:
            height_image = self._active_display_image(MapType.HEIGHT)
            if height_image is not None:
                self._generate_ao_from_height(_image_to_height_array(height_image), commit=True)
                self._refresh_selected_map_preview()
                self._push_textures_to_preview()
        elif map_type == MapType.ALBEDO:
            albedo_src = self.source_images.get(MapType.ALBEDO)
            if albedo_src is not None:
                shifted = apply_albedo_adjustments(
                    albedo_src,
                    options=self.albedo_panel.current_adjust_options(),
                    rules=self.albedo_panel.current_shift_rules(),
                )
                self.generated_images[MapType.ALBEDO] = shifted
                self._draft_images.pop(MapType.ALBEDO, None)
                self._refresh_selected_map_preview()
                self._push_textures_to_preview()

        panel = self._panels_by_type[map_type]
        self._properties_snapshot[map_type] = panel.properties_to_dict()
        self._update_save_revert_enabled()
        self._refresh_material_slots_bar()
        self._mark_dirty()
        self.status_label.setText(f"Saved {MAP_TYPE_LABELS[map_type]} properties.")

    def _revert_current_properties(self) -> None:
        self._invalidate_properties_preview()
        map_type = self.current_map_type
        panel = self._panels_by_type[map_type]
        snapshot = self._properties_snapshot.get(map_type, {})
        panel.properties_apply_dict(snapshot)
        self._draft_images.pop(map_type, None)
        if map_type == MapType.HEIGHT:
            self._draft_images.pop(MapType.NORMAL, None)
            self._draft_images.pop(MapType.ROUGHNESS, None)
            self._draft_images.pop(MapType.AO, None)
        elif map_type == MapType.NORMAL:
            self._draft_images.pop(MapType.ROUGHNESS, None)
        self._refresh_selected_map_preview()
        self._refresh_material_slots_bar()
        self._push_textures_to_preview()
        self._update_save_revert_enabled()
        self.status_label.setText(f"Reverted {MAP_TYPE_LABELS[map_type]} properties.")

    def _set_busy(self, busy: bool) -> None:
        if self._bottom_bar_mode == "generate":
            self.generate_btn.setEnabled(not busy)
            self.kill_btn.setEnabled(busy)
        self.height_panel.set_model_controls_enabled(not busy)
        self.height_panel.set_chunk_controls_enabled(not busy)
        self.progress_bar.setVisible(busy)
        if self._taskbar_progress is not None:
            self._taskbar_progress.set_busy(busy)
        if not busy and not self.isActiveWindow():
            QApplication.alert(self)

    def _on_preview_settings_changed(self, settings: dict) -> None:
        self.preview_3d.set_preview_settings(settings)
        self._mark_dirty()

    def _push_textures_to_preview(self) -> None:
        albedo = self._active_display_image(MapType.ALBEDO)
        height = self._active_display_image(MapType.HEIGHT)
        normal = self._active_display_image(MapType.NORMAL)
        roughness = self._active_display_image(MapType.ROUGHNESS)
        ao = self._active_display_image(MapType.AO)
        self.preview_3d.set_textures(albedo, normal, roughness, height=height, ao=ao)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_window_title()

    def _update_window_title(self) -> None:
        if IS_MAC:
            self.setWindowTitle("[*]")
            self.setWindowModified(self.dirty)
            return
        name = os.path.basename(self.project_path) if self.project_path else "Untitled"
        star = "*" if self.dirty else ""
        self.setWindowTitle(f"PBRCELAIN — {name}{star}")

    def _confirm_discard_if_dirty(self) -> bool:
        if not self.dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            "You have unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self._save_project()
        return reply == QMessageBox.StandardButton.Discard

    def _new_project(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self.project = ProjectData.new()
        self.project_path = None
        self.source_images.clear()
        self.generated_images.clear()
        self._draft_images.clear()
        self._height_depth_result = None
        self.dirty = False
        for panel in (self.height_panel, self.normal_panel, self.albedo_panel, self.roughness_panel, self.ao_panel):
            panel.reset_to_defaults()
        self.preview_properties_panel.apply_settings({})
        self.preview_3d.set_preview_settings(self.preview_properties_panel.get_settings())
        self._reset_properties_snapshots()
        self._select_map_type(MapType.ALBEDO)
        self._push_textures_to_preview()
        self._update_window_title()

    def _open_project(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", f"PBRCELAIN Project (*{PROJECT_EXTENSION})")
        if not path:
            return
        self._load_project_from_path(path)

    def _handle_dropped_project(self, path: str) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: str) -> None:
        fname = os.path.basename(path)
        dialog = OperationProgressDialog("Loading Project", f"Opening {fname}…", parent=self)
        worker = ProjectLoadWorker(path, parent=self)
        worker.progress.connect(dialog.set_status)

        loaded_result = []

        def on_loaded(project, source_images, cache_images, height_raw):
            loaded_result.append((project, source_images, cache_images, height_raw))
            dialog.accept()

        def on_failed(err):
            dialog.reject()
            QMessageBox.critical(self, "Failed to open project", str(err))

        worker.finished_ok.connect(on_loaded)
        worker.failed.connect(on_failed)
        worker.start()
        dialog.exec()

        if not loaded_result:
            return

        project, source_images, cache_images, height_raw = loaded_result[0]
        self.project = project
        self.project_path = path
        self._add_recent_project(path)
        self.source_images = source_images
        self.generated_images = cache_images
        self._draft_images.clear()
        self._height_depth_result = height_raw
        self.dirty = False

        self.height_panel.apply_options_dict(project.map_slots.get(MapType.HEIGHT.value, MapSlotData()).options)
        self.normal_panel.apply_options_dict(project.map_slots.get(MapType.NORMAL.value, MapSlotData()).options)
        self.albedo_panel.apply_options_dict(project.map_slots.get(MapType.ALBEDO.value, MapSlotData()).options)
        self.roughness_panel.apply_options_dict(project.map_slots.get(MapType.ROUGHNESS.value, MapSlotData()).options)
        self.ao_panel.apply_options_dict(project.map_slots.get(MapType.AO.value, MapSlotData()).options)
        self.preview_properties_panel.apply_settings(project.preview_settings)
        self.preview_3d.set_preview_settings(self.preview_properties_panel.get_settings())
        self._reset_properties_snapshots()

        self._select_map_type(MapType.ALBEDO)
        self._push_textures_to_preview()
        self._update_window_title()
        self.status_label.setText(f"Project loaded from {fname}")

    def _collect_project_data(self) -> ProjectData:
        return ProjectData(
            map_slots={
                MapType.ALBEDO.value: MapSlotData(
                    has_source=MapType.ALBEDO in self.source_images,
                    has_cache=MapType.ALBEDO in self.generated_images,
                    options=self.albedo_panel.to_options_dict(),
                ),
                MapType.HEIGHT.value: MapSlotData(
                    has_source=MapType.HEIGHT in self.source_images,
                    has_cache=MapType.HEIGHT in self.generated_images,
                    options=self.height_panel.to_options_dict(),
                ),
                MapType.NORMAL.value: MapSlotData(
                    has_source=MapType.NORMAL in self.source_images,
                    has_cache=MapType.NORMAL in self.generated_images,
                    options=self.normal_panel.to_options_dict(),
                ),
                MapType.ROUGHNESS.value: MapSlotData(
                    has_source=MapType.ROUGHNESS in self.source_images,
                    has_cache=MapType.ROUGHNESS in self.generated_images,
                    options=self.roughness_panel.to_options_dict(),
                ),
                MapType.AO.value: MapSlotData(
                    has_source=MapType.AO in self.source_images,
                    has_cache=MapType.AO in self.generated_images,
                    options=self.ao_panel.to_options_dict(),
                ),
            },
            preview_settings=self.preview_properties_panel.get_settings(),
        )

    def _save_project(self) -> bool:
        if self.project_path is None:
            return self._save_project_as()
        return self._write_project(self.project_path)

    def _save_project_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", "", f"PBRCELAIN Project (*{PROJECT_EXTENSION})")
        if not path:
            return False
        if not path.lower().endswith(PROJECT_EXTENSION):
            path += PROJECT_EXTENSION
        return self._write_project(path)

    def _write_project(self, path: str) -> bool:
        self._bake_all_pending_properties()
        project = self._collect_project_data()

        fname = os.path.basename(path)
        dialog = OperationProgressDialog("Saving Project", f"Packaging {fname}…", parent=self)
        worker = ProjectSaveWorker(
            path,
            project,
            self.source_images,
            self.generated_images,
            height_raw=self._height_depth_result,
            parent=self,
        )
        worker.progress.connect(dialog.set_status)

        saved = [False]

        def on_saved():
            saved[0] = True
            dialog.accept()

        def on_failed(err):
            dialog.reject()
            QMessageBox.critical(self, "Failed to save project", str(err))

        worker.finished_ok.connect(on_saved)
        worker.failed.connect(on_failed)
        worker.start()
        dialog.exec()

        if saved[0]:
            self.project = project
            self.project_path = path
            self._add_recent_project(path)
            self.dirty = False
            self._update_window_title()
            self.status_label.setText(f"Project saved to {fname}")
        return saved[0]

    def _export_maps(self) -> None:
        self._bake_all_pending_properties()

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not out_dir:
            return

        items = []
        for mt in MAP_TYPE_ORDER:
            img = self.generated_images.get(mt) or self.source_images.get(mt)
            if img is not None:
                ext = ".exr" if getattr(img, "mode", None) == "F" else ".png"
                fname = f"{mt.value}{ext}"
                items.append((MAP_TYPE_LABELS[mt], img, fname))

        height_img = self.generated_images.get(MapType.HEIGHT) or self.source_images.get(MapType.HEIGHT)
        if height_img is not None and self.ao_panel.export_curvature():
            curvature = generate_curvature_map(_image_to_height_array(height_img))
            items.append(("Curvature", curvature_map_to_image(curvature), "curvature.png"))

        if not items:
            QMessageBox.information(self, "Nothing to export", "No maps have been generated or uploaded yet.")
            return

        dialog = OperationProgressDialog("Exporting Maps", "Preparing export…", parent=self)
        dialog.set_progress(0, len(items))

        worker = ExportWorker(out_dir, items, parent=self)
        worker.progress.connect(lambda cur, total, msg: (dialog.set_progress(cur, total), dialog.set_status(msg)))

        export_res = []

        def on_exported(exported):
            export_res.extend(exported)
            dialog.accept()

        def on_failed(err):
            dialog.reject()
            QMessageBox.critical(self, "Export Failed", str(err))

        worker.finished_ok.connect(on_exported)
        worker.failed.connect(on_failed)
        worker.start()
        dialog.exec()

        if export_res:
            exported_list = "\n".join(f"• {item}" for item in export_res)
            self.status_label.setText(f"Exported {len(export_res)} map(s) to {out_dir}")
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {len(export_res)} map(s) to:\n{out_dir}\n\n{exported_list}",
            )

    def dragEnterEvent(self, event) -> None:
        for url in event.mimeData().urls():
            if url.toLocalFile().lower().endswith(PROJECT_EXTENSION):
                event.acceptProposedAction()
                return
        super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(PROJECT_EXTENSION):
                event.acceptProposedAction()
                self._handle_dropped_project(path)
                return
        super().dropEvent(event)

    def closeEvent(self, event) -> None:
        if not self._confirm_discard_if_dirty():
            event.ignore()
            return
        if self.infer_thread is not None and self.infer_thread.isRunning():
            self.infer_thread.requestInterruption()
        for thread in (self.model_thread, self.infer_thread, self.properties_preview_thread):
            if thread is not None and thread.isRunning():
                thread.requestInterruption()
                thread.quit()
                thread.wait(5000)
        if PLATFORM_THEME != "legacy":
            window_layout.save(self, self._main_splitter, self._preview_splitter)
        super().closeEvent(event)
