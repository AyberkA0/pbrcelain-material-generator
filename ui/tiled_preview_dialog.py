"""Interactive 2x2 Tiled Seamless Preview Dialog for PBRCELAIN.

Allows inspecting edge and corner seamlessness of any map in 2x2 (or 3x3) repeating
grids with smooth panning, cursor-anchored zooming, and optional seam guide lines.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui import trackpad
from ui.theme import IS_MAC, legacy_style

class InteractiveTiledCanvas(QWidget):
    zoomChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._source_image: Image.Image | None = None
        self._tile_count: int = 2
        self._tiled_pixmap: QPixmap | None = None
        self._tile_w: int = 0
        self._tile_h: int = 0

        self._zoom: float = 1.0
        self._user_zoomed = False
        self._pan: QPointF = QPointF(0, 0)
        self._show_guides: bool = True

        self._is_dragging: bool = False
        self._last_mouse_pos: QPoint = QPoint()

    def set_source_image(self, image: Image.Image | QPixmap | np.ndarray, tile_count: int = 2) -> None:
        if isinstance(image, Image.Image):
            if image.mode in ("I;16", "I"):
                arr = (np.asarray(image, dtype=np.float32) / 256.0).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr, mode="L").convert("RGB")
            elif image.mode == "F":
                arr = (np.asarray(image, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(arr, mode="L").convert("RGB")
            elif "A" in image.mode:
                pil_img = image.convert("RGBA")
            elif image.mode == "L":
                pil_img = image.convert("RGB")
            else:
                pil_img = image.convert("RGB")
        elif isinstance(image, QPixmap):
            qimg = image.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            ptr = qimg.bits()
            ptr.setsize(qimg.sizeInBytes())
            arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width(), 4))
            pil_img = Image.fromarray(arr, mode="RGBA")
        elif isinstance(image, np.ndarray):
            arr = image
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            elif arr.dtype == np.uint16:
                arr = (arr.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                pil_img = Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pil_img = Image.fromarray(arr, mode="RGB")
            else:
                pil_img = Image.fromarray(arr, mode="RGBA")
        else:
            return

        self._source_image = pil_img
        self._tile_count = max(1, tile_count)
        self._tile_w, self._tile_h = pil_img.size
        self._rebuild_tiled_pixmap()
        self.fit_to_window()

    def set_tile_count(self, count: int) -> None:
        if count != self._tile_count:
            self._tile_count = max(1, count)
            self._rebuild_tiled_pixmap()
            self.update()

    def set_show_guides(self, show: bool) -> None:
        self._show_guides = show
        self.update()

    def _rebuild_tiled_pixmap(self) -> None:
        if self._source_image is None:
            self._tiled_pixmap = None
            return

        w, h = self._tile_w, self._tile_h
        n = self._tile_count
        mode = self._source_image.mode
        composite = Image.new(mode, (w * n, h * n))

        for r in range(n):
            for c in range(n):
                composite.paste(self._source_image, (c * w, r * h))

        if mode == "RGBA":
            qfmt = QImage.Format.Format_RGBA8888
            raw_data = composite.tobytes("raw", "RGBA")
        else:
            rgb_comp = composite.convert("RGB")
            qfmt = QImage.Format.Format_RGB888
            raw_data = rgb_comp.tobytes("raw", "RGB")

        qimg = QImage(raw_data, w * n, h * n, (4 if mode == "RGBA" else 3) * w * n, qfmt)
        self._tiled_pixmap = QPixmap.fromImage(qimg)

    def fit_to_window(self) -> None:
        if self._tiled_pixmap is None or self._tiled_pixmap.isNull():
            return
        vw = max(10, self.width() - 30)
        vh = max(10, self.height() - 30)
        pw = self._tiled_pixmap.width()
        ph = self._tiled_pixmap.height()
        scale = min(vw / float(pw), vh / float(ph), 1.0)
        self._zoom = max(0.05, scale)
        self._user_zoomed = False
        self._center_image()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def set_zoom(self, zoom: float, anchor_pos: QPoint | None = None) -> None:
        new_zoom = float(np.clip(zoom, 0.05, 16.0))
        self._user_zoomed = True
        if abs(new_zoom - self._zoom) < 1e-4:
            return

        if anchor_pos is not None:
            anchor_x = anchor_pos.x()
            anchor_y = anchor_pos.y()
            img_x = (anchor_x - self._pan.x()) / self._zoom
            img_y = (anchor_y - self._pan.y()) / self._zoom
            self._zoom = new_zoom
            new_pan_x = anchor_x - img_x * self._zoom
            new_pan_y = anchor_y - img_y * self._zoom
            self._pan = QPointF(new_pan_x, new_pan_y)
        else:
            cx = self.width() / 2.0
            cy = self.height() / 2.0
            img_x = (cx - self._pan.x()) / self._zoom
            img_y = (cy - self._pan.y()) / self._zoom
            self._zoom = new_zoom
            self._pan = QPointF(cx - img_x * self._zoom, cy - img_y * self._zoom)

        self.zoomChanged.emit(self._zoom)
        self.update()

    def _center_image(self) -> None:
        if self._tiled_pixmap is None:
            return
        disp_w = self._tiled_pixmap.width() * self._zoom
        disp_h = self._tiled_pixmap.height() * self._zoom
        px = (self.width() - disp_w) / 2.0
        py = (self.height() - disp_h) / 2.0
        self._pan = QPointF(px, py)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#181818"))

        if self._tiled_pixmap is None or self._tiled_pixmap.isNull():
            painter.setPen(QColor("#8a8a8a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No map loaded to preview")
            return

        pw = self._tiled_pixmap.width()
        ph = self._tiled_pixmap.height()
        disp_w = pw * self._zoom
        disp_h = ph * self._zoom

        target_rect = QRectF(self._pan.x(), self._pan.y(), disp_w, disp_h)

        border_rect = QRectF(target_rect.x() - 1, target_rect.y() - 1, target_rect.width() + 2, target_rect.height() + 2)
        painter.fillRect(border_rect, QColor("#101010"))

        if self._zoom >= 2.0:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        else:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.drawPixmap(target_rect.toRect(), self._tiled_pixmap)

        if self._show_guides and self._tile_count > 1:
            pen = QPen(QColor(56, 189, 248, 190), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            n = self._tile_count
            step_w = self._tile_w * self._zoom
            step_h = self._tile_h * self._zoom

            for i in range(1, n):
                x = self._pan.x() + i * step_w
                painter.drawLine(int(round(x)), int(round(self._pan.y())), int(round(x)), int(round(self._pan.y() + disp_h)))

            for j in range(1, n):
                y = self._pan.y() + j * step_h
                painter.drawLine(int(round(self._pan.x())), int(round(y)), int(round(self._pan.x() + disp_w)), int(round(y)))

            if n == 2:
                cx = int(round(self._pan.x() + step_w))
                cy = int(round(self._pan.y() + step_h))
                cross_pen = QPen(QColor(244, 63, 94, 230), 2, Qt.PenStyle.SolidLine)
                painter.setPen(cross_pen)
                painter.drawLine(cx - 8, cy, cx + 8, cy)
                painter.drawLine(cx, cy - 8, cx, cy + 8)

    def wheelEvent(self, event) -> None:
        if trackpad.is_trackpad_scroll(event):
            self._pan += trackpad.scroll_delta(event)
            self.update()
            return
        delta = event.angleDelta().y()
        if delta != 0:
            factor = 1.15 if delta > 0 else 1.0 / 1.15
            self.set_zoom(self._zoom * factor, anchor_pos=event.position().toPoint())

    def event(self, event) -> bool:
        factor = trackpad.pinch_factor(event)
        if factor is not None:
            self.set_zoom(self._zoom * factor, anchor_pos=event.position().toPoint())
            return True
        if trackpad.is_smart_zoom(event):
            if abs(self._zoom - 1.0) < 0.05:
                self.fit_to_window()
            else:
                self.set_zoom(1.0, anchor_pos=event.position().toPoint())
            return True
        return super().event(event)

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._is_dragging = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._is_dragging:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self._pan += QPointF(delta.x(), delta.y())
            self.update()
            event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._is_dragging and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._is_dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if abs(self._zoom - 1.0) < 0.05:
                self.fit_to_window()
            else:
                self.set_zoom(1.0, anchor_pos=event.pos())
            event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # The image is set (and first fitted) before the dialog is laid out,
        # so keep re-fitting on resize until the user picks a zoom themselves.
        if not self._user_zoomed:
            self.fit_to_window()
        elif self._zoom <= 1.0:
            self._center_image()


class TiledPreviewDialog(QDialog):
    """Photoshop-style dialog presenting 2x2 / 3x3 tiled seamless preview with zoom and pan."""

    def __init__(self, image: Image.Image | QPixmap | np.ndarray, map_name: str = "Active Map", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{map_name} - 2x2 Tiled Seamless Preview")
        self.resize(900, 850)
        self.setMinimumSize(600, 500)

        legacy_style(self, """
            QDialog {
                background-color: #282828;
                color: #dedede;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
                font-size: 11px;
            }
            QWidget#ToolbarPanel {
                background-color: #323232;
                border-bottom: 1px solid #202020;
                padding: 4px 8px;
            }
            QWidget#StatusBarPanel {
                background-color: #242424;
                border-top: 1px solid #202020;
                padding: 4px 10px;
            }
            QPushButton {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
                border-color: #555555;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #282828;
            }
            QComboBox {
                background-color: #1a1a1a;
                border: 1px solid #3c3c3c;
                border-radius: 2px;
                padding: 3px 6px;
                color: #ffffff;
            }
            QCheckBox {
                color: #dedede;
                spacing: 6px;
            }
            QLabel {
                color: #b0b0b0;
                font-size: 11px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QWidget()
        toolbar.setObjectName("ToolbarPanel")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 6, 8, 6)
        tb_layout.setSpacing(10)

        tb_layout.addWidget(QLabel("Tiling Mode:"))
        self.tile_combo = QComboBox()
        self.tile_combo.addItem("2x2 Tiled (Seamless Inspection)", 2)
        self.tile_combo.addItem("3x3 Tiled (Full Field)", 3)
        self.tile_combo.addItem("1x1 Single Tile", 1)
        self.tile_combo.currentIndexChanged.connect(self._on_tile_mode_changed)
        tb_layout.addWidget(self.tile_combo)

        tb_layout.addSpacing(8)

        self.guides_check = QCheckBox("Show Seam Guides")
        self.guides_check.setChecked(True)
        self.guides_check.setToolTip("Overlay dashed guide lines showing the exact boundaries between tiles.")
        self.guides_check.toggled.connect(self._on_guides_toggled)
        tb_layout.addWidget(self.guides_check)

        tb_layout.addStretch(1)

        self.btn_zoom_out = QPushButton("− Zoom")
        self.btn_zoom_out.clicked.connect(lambda: self.canvas.set_zoom(self.canvas._zoom / 1.25))
        tb_layout.addWidget(self.btn_zoom_out)

        self.btn_100 = QPushButton("100% (1:1)")
        self.btn_100.clicked.connect(lambda: self.canvas.set_zoom(1.0))
        tb_layout.addWidget(self.btn_100)

        self.btn_fit = QPushButton("Fit Window")
        self.btn_fit.clicked.connect(self._on_fit_clicked)
        tb_layout.addWidget(self.btn_fit)

        self.btn_zoom_in = QPushButton("+ Zoom")
        self.btn_zoom_in.clicked.connect(lambda: self.canvas.set_zoom(self.canvas._zoom * 1.25))
        tb_layout.addWidget(self.btn_zoom_in)

        layout.addWidget(toolbar)

        self.canvas = InteractiveTiledCanvas(self)
        self.canvas.zoomChanged.connect(self._update_status)
        layout.addWidget(self.canvas, 1)

        status_bar = QWidget()
        status_bar.setObjectName("StatusBarPanel")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(10, 4, 10, 4)
        sb_layout.setSpacing(12)

        self.info_label = QLabel("")
        self.hint_label = QLabel(
            "💡 Drag or two-finger swipe to pan | Pinch or mouse wheel to zoom | Double-click to reset zoom"
            if IS_MAC else
            "💡 Drag to pan | Mouse wheel to zoom | Double-click to reset zoom"
        )
        legacy_style(self.hint_label, "color: #777777; font-size: 10px;", "hint")
        sb_layout.addWidget(self.info_label)
        sb_layout.addStretch(1)
        sb_layout.addWidget(self.hint_label)
        layout.addWidget(status_bar)

        self.canvas.set_source_image(image, tile_count=2)
        self._update_status(self.canvas._zoom)

    def _on_tile_mode_changed(self) -> None:
        count = self.tile_combo.currentData()
        self.canvas.set_tile_count(count)
        self._update_status(self.canvas._zoom)

    def _on_guides_toggled(self, checked: bool) -> None:
        self.canvas.set_show_guides(checked)

    def _on_fit_clicked(self) -> None:
        self.canvas.fit_to_window()

    def _update_status(self, zoom: float) -> None:
        w = self.canvas._tile_w
        h = self.canvas._tile_h
        n = self.canvas._tile_count
        comp_w = w * n
        comp_h = h * n
        zoom_pct = int(round(zoom * 100))
        self.info_label.setText(
            f"<b>Tile:</b> {w} × {h} px | <b>Composite:</b> {comp_w} × {comp_h} px | <b>Zoom:</b> {zoom_pct}%"
        )
