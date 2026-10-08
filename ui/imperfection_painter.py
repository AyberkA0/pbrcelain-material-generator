from __future__ import annotations

from typing import Literal

from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QImage,
    QKeySequence,
    QPainter,
    QPointingDevice,
    QPen,
    QPixmap,
    QRadialGradient,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PIL import Image
import numpy as np

from ui import trackpad
from ui.theme import legacy_style, pick

_CARD_STYLE = "QFrame#PainterCard { background-color: #323232; border: 1px solid #202020; border-radius: 3px; padding: 6px; }"

BrushMode = Literal["rough", "gloss", "eraser"]


def _shortcut_text(key: QKeySequence.StandardKey) -> str:
    bindings = QKeySequence.keyBindings(key)
    return bindings[0].toString(QKeySequence.SequenceFormat.NativeText) if bindings else ""


class ImperfectionCanvasWidget(QWidget):
    maskChanged = pyqtSignal()

    def __init__(self, width: int = 512, height: int = 512, parent=None):
        super().__init__(parent)
        self.canvas_w = max(128, width)
        self.canvas_h = max(128, height)

        self._paint_image = QImage(self.canvas_w, self.canvas_h, QImage.Format.Format_ARGB32_Premultiplied)
        self._paint_image.fill(Qt.GlobalColor.transparent)

        self._underlays: dict[str, QPixmap] = {}
        self._active_underlay_key = "albedo"
        self._underlay_opacity = 0.65

        self.brush_mode: BrushMode = "rough"
        self.brush_size = 42
        self.brush_hardness = 0.2
        self.brush_flow = 0.5
        self._pressure = 1.0
        self._pen_eraser = False
        self.target_roughness = 1.0

        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._last_mouse_pos: QPointF | None = None
        self._last_canvas_pos: QPointF | None = None
        self._is_panning = False
        self._is_painting = False
        self._cursor_canvas_pos: QPointF | None = None

        self._undo_stack: list[QImage] = []
        self._redo_stack: list[QImage] = []
        self._max_undo = 25

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("background-color: #181818;")

    def set_underlay_images(
        self,
        albedo: Image.Image | None = None,
        normal: Image.Image | None = None,
        roughness: Image.Image | None = None,
    ) -> None:
        self._underlays.clear()
        if albedo is not None:
            alb_thumb = albedo.convert("RGB").resize((self.canvas_w, self.canvas_h), Image.Resampling.BILINEAR)
            arr = np.ascontiguousarray(np.asarray(alb_thumb, dtype=np.uint8))
            qimg = QImage(arr.data, self.canvas_w, self.canvas_h, self.canvas_w * 3, QImage.Format.Format_RGB888)
            self._underlays["albedo"] = QPixmap.fromImage(qimg.copy())

        if normal is not None:
            norm_thumb = normal.convert("RGB").resize((self.canvas_w, self.canvas_h), Image.Resampling.BILINEAR)
            arr = np.ascontiguousarray(np.asarray(norm_thumb, dtype=np.uint8))
            qimg = QImage(arr.data, self.canvas_w, self.canvas_h, self.canvas_w * 3, QImage.Format.Format_RGB888)
            self._underlays["normal"] = QPixmap.fromImage(qimg.copy())

        if roughness is not None:
            rough_thumb = roughness.convert("L").resize((self.canvas_w, self.canvas_h), Image.Resampling.BILINEAR)
            arr = np.ascontiguousarray(np.asarray(rough_thumb, dtype=np.uint8))
            qimg = QImage(arr.data, self.canvas_w, self.canvas_h, self.canvas_w, QImage.Format.Format_Grayscale8)
            self._underlays["roughness"] = QPixmap.fromImage(qimg.copy())

        self.update()

    def set_active_underlay(self, key: str) -> None:
        self._active_underlay_key = key
        self.update()

    def set_underlay_opacity(self, val: float) -> None:
        self._underlay_opacity = min(max(val, 0.0), 1.0)
        self.update()

    def load_mask_array(self, mask: np.ndarray | None) -> None:
        if mask is None or mask.size == 0:
            self.clear_canvas()
            return
        h, w = mask.shape[:2]
        if h != self.canvas_h or w != self.canvas_w:
            mask = cv2_resize_mask(mask, self.canvas_w, self.canvas_h)
        c_arr = np.ascontiguousarray(mask, dtype=np.uint8)
        qimg = QImage(c_arr.data, self.canvas_w, self.canvas_h, self.canvas_w * 4, QImage.Format.Format_RGBA8888)
        self._paint_image = qimg.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied).copy()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._save_undo_state()
        self.update()

    def get_mask_array(self) -> np.ndarray | None:
        rgba_img = self._paint_image.convertToFormat(QImage.Format.Format_RGBA8888)
        ptr = rgba_img.bits()
        ptr.setsize(rgba_img.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape((self.canvas_h, self.canvas_w, 4)).copy()
        if not np.any(arr[..., 3] > 0):
            return None
        return arr

    def has_painted_content(self) -> bool:
        return self.get_mask_array() is not None

    def clear_canvas(self) -> None:
        self._save_undo_state()
        self._paint_image.fill(Qt.GlobalColor.transparent)
        self.update()
        self.maskChanged.emit()

    def invert_roughness(self) -> None:
        self._save_undo_state()
        arr = self.get_mask_array()
        if arr is not None:
            arr[..., 0] = 255 - arr[..., 0]
            self.load_mask_array(arr)
            self.maskChanged.emit()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def _save_undo_state(self) -> None:
        self._undo_stack.append(self._paint_image.copy())
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> None:
        if self._undo_stack:
            self._redo_stack.append(self._paint_image.copy())
            self._paint_image = self._undo_stack.pop()
            self.update()
            self.maskChanged.emit()

    def redo(self) -> None:
        if self._redo_stack:
            self._undo_stack.append(self._paint_image.copy())
            self._paint_image = self._redo_stack.pop()
            self.update()
            self.maskChanged.emit()

    def reset_view(self) -> None:
        w_avail = self.width() - 40
        h_avail = self.height() - 40
        if w_avail <= 0 or h_avail <= 0:
            return
        scale_w = w_avail / self.canvas_w
        scale_h = h_avail / self.canvas_h
        self._zoom = min(scale_w, scale_h, 1.5)
        cx = (self.width() - self.canvas_w * self._zoom) / 2.0
        cy = (self.height() - self.canvas_h * self._zoom) / 2.0
        self._pan = QPointF(cx, cy)
        self.update()

    def _widget_to_canvas(self, pt: QPointF) -> QPointF:
        return QPointF(
            (pt.x() - self._pan.x()) / self._zoom,
            (pt.y() - self._pan.y()) / self._zoom,
        )

    def _canvas_to_widget(self, pt: QPointF) -> QPointF:
        return QPointF(
            pt.x() * self._zoom + self._pan.x(),
            pt.y() * self._zoom + self._pan.y(),
        )

    def _paint_stamp(self, center: QPointF, painter: QPainter) -> None:
        radius = max(2.0, self.brush_size / 2.0 * (0.3 + 0.7 * self._pressure))
        flow = self.brush_flow * self._pressure
        grad = QRadialGradient(center, radius)

        if self.brush_mode == "eraser" or self._pen_eraser:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            alpha_int = int(round(flow * 255))
            grad.setColorAt(0.0, QColor(0, 0, 0, alpha_int))
            hard_stop = min(max(self.brush_hardness, 0.01), 0.99)
            grad.setColorAt(hard_stop, QColor(0, 0, 0, alpha_int))
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            target_val = int(round(self.target_roughness * 255))
            alpha_int = int(round(flow * 255))
            color = QColor(target_val, 0, 0, alpha_int)
            grad.setColorAt(0.0, color)
            hard_stop = min(max(self.brush_hardness, 0.01), 0.99)
            grad.setColorAt(hard_stop, color)
            grad.setColorAt(1.0, QColor(target_val, 0, 0, 0))

        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius, radius)

    def _apply_brush_stroke(self, p1: QPointF, p2: QPointF) -> None:
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        dist = np.hypot(dx, dy)
        step = max(1.5, self.brush_size * 0.15)
        num_steps = max(1, int(np.ceil(dist / step)))

        painter = QPainter(self._paint_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for s in range(num_steps + 1):
            t = s / float(num_steps)
            pt = QPointF(p1.x() + dx * t, p1.y() + dy * t)
            self._paint_stamp(pt, painter)

        painter.end()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor("#121214"))

        c_top_left = self._canvas_to_widget(QPointF(0, 0))
        c_bottom_right = self._canvas_to_widget(QPointF(self.canvas_w, self.canvas_h))
        canvas_rect = QRectF(c_top_left, c_bottom_right)

        self._draw_checkerboard(painter, canvas_rect)

        if self._active_underlay_key in self._underlays:
            pixmap = self._underlays[self._active_underlay_key]
            painter.setOpacity(self._underlay_opacity)
            painter.drawPixmap(canvas_rect.toRect(), pixmap)
            painter.setOpacity(1.0)

        painter.drawImage(canvas_rect, self._paint_image)

        painter.setPen(QPen(QColor("#38bdf8" if self.brush_mode != "eraser" else "#fbbf24"), 1.0, Qt.PenStyle.DashLine))
        painter.drawRect(canvas_rect)

        if self._cursor_canvas_pos is not None:
            mouse_widget_pt = self._canvas_to_widget(self._cursor_canvas_pos)
            screen_r = (self.brush_size / 2.0) * self._zoom
            painter.setPen(QPen(QColor(255, 255, 255, 210), 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(mouse_widget_pt, screen_r, screen_r)
            if self.brush_hardness > 0.05:
                inner_r = screen_r * self.brush_hardness
                painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0, Qt.PenStyle.DotLine))
                painter.drawEllipse(mouse_widget_pt, inner_r, inner_r)

    def _draw_checkerboard(self, painter: QPainter, rect: QRectF) -> None:
        painter.fillRect(rect, QColor("#1c1c1f"))
        check_size = max(8.0, 16.0 * self._zoom)
        painter.save()
        painter.setClipRect(rect)
        c1 = QColor("#242429")
        cols = int(np.ceil(rect.width() / check_size))
        rows = int(np.ceil(rect.height() / check_size))
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    painter.fillRect(
                        QRectF(rect.left() + c * check_size, rect.top() + r * check_size, check_size, check_size),
                        c1,
                    )
        painter.restore()

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        self._last_mouse_pos = pos

        if event.button() == Qt.MouseButton.MiddleButton or (event.modifiers() & Qt.KeyboardModifier.AltModifier):
            self._is_panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self._begin_stroke(pos)

    def _begin_stroke(self, pos: QPointF) -> None:
        c_pos = self._widget_to_canvas(pos)
        self._last_canvas_pos = c_pos
        self._is_painting = True
        self._save_undo_state()
        painter = QPainter(self._paint_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._paint_stamp(c_pos, painter)
        painter.end()
        self.update()

    def _continue_stroke(self, pos: QPointF) -> None:
        curr_canvas_pos = self._widget_to_canvas(pos)
        self._apply_brush_stroke(self._last_canvas_pos, curr_canvas_pos)
        self._last_canvas_pos = curr_canvas_pos
        self.update()

    def _end_stroke(self) -> None:
        self._is_painting = False
        self._last_canvas_pos = None
        self.maskChanged.emit()
        self.update()

    def tabletEvent(self, event) -> None:
        etype = event.type()
        pos = event.position()
        self._cursor_canvas_pos = self._widget_to_canvas(pos)
        self._pressure = max(0.05, min(1.0, event.pressure()))
        if etype == QEvent.Type.TabletPress and event.button() == Qt.MouseButton.LeftButton:
            self._pen_eraser = event.pointerType() == QPointingDevice.PointerType.Eraser
            self._begin_stroke(pos)
        elif etype == QEvent.Type.TabletMove and self._is_painting and self._last_canvas_pos is not None:
            self._continue_stroke(pos)
        elif etype == QEvent.Type.TabletRelease and self._is_painting:
            self._end_stroke()
            self._pressure = 1.0
            self._pen_eraser = False
        else:
            self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._cursor_canvas_pos = self._widget_to_canvas(pos)

        if self._is_panning and self._last_mouse_pos is not None:
            delta = pos - self._last_mouse_pos
            self._pan += delta
            self._last_mouse_pos = pos
            self.update()
            return

        if self._is_painting and self._last_canvas_pos is not None:
            self._continue_stroke(pos)
            return

        self._last_mouse_pos = pos
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if event.button() == Qt.MouseButton.LeftButton and self._is_painting:
            self._end_stroke()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if trackpad.is_trackpad_scroll(event):
            self._pan += trackpad.scroll_delta(event)
            self.update()
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        self._zoom_at(event.position(), 1.15 if angle > 0 else (1.0 / 1.15))

    def event(self, event) -> bool:
        factor = trackpad.pinch_factor(event)
        if factor is not None:
            self._zoom_at(event.position(), factor)
            return True
        if trackpad.is_smart_zoom(event):
            self.reset_view()
            return True
        return super().event(event)

    def _zoom_at(self, mouse_pt, factor: float) -> None:
        canvas_before = self._widget_to_canvas(mouse_pt)
        new_zoom = min(max(self._zoom * factor, 0.15), 12.0)
        self._zoom = new_zoom

        canvas_after = self._widget_to_canvas(mouse_pt)
        self._pan += (canvas_after - canvas_before) * self._zoom
        self.update()

    def leaveEvent(self, event) -> None:
        self._cursor_canvas_pos = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._zoom == 1.0 and self._pan == QPointF(0, 0):
            self.reset_view()


class ImperfectionPainterDialog(QDialog):
    def __init__(
        self,
        mask: np.ndarray | None = None,
        albedo: Image.Image | None = None,
        normal: Image.Image | None = None,
        roughness: Image.Image | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Roughness Imperfection & Wear Painter")
        self.resize(1120, 680)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        header = QLabel(
            "<b>Surface Imperfection Painter:</b> Paint custom surface wear, rust, or shiny polished areas. "
            "Use the Underlay selector to align with cracks or features from your original photo."
        )
        legacy_style(header, "color: #94a3b8; font-size: 11px;", "hint")
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        ctrl_panel = QWidget()
        ctrl_panel.setMaximumWidth(280)
        ctrl_layout = QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 10, 0)
        ctrl_layout.setSpacing(12)

        tool_frame = QFrame()
        tool_frame.setObjectName("PainterCard")
        legacy_style(tool_frame, _CARD_STYLE)
        tool_vbox = QVBoxLayout(tool_frame)
        tool_vbox.setSpacing(6)

        tool_title = QLabel("BRUSH PRESETS")
        legacy_style(tool_title, "font-weight: bold; font-size: 10px; color: #8a8a8a; letter-spacing: 0.5px;", "section-caps")
        tool_vbox.addWidget(tool_title)

        self.btn_rough = QPushButton("🔨 Rusty / Worn")
        self.btn_rough.setCheckable(True)
        self.btn_rough.setChecked(True)
        legacy_style(self.btn_rough, """
            QPushButton:checked {
                background-color: #1473e6;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #0d66d0;
                border-radius: 3px;
                padding: 6px;
                text-align: left;
            }
            QPushButton:!checked {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 6px;
                text-align: left;
            }
        """, "brush-preset")
        self.btn_rough.clicked.connect(lambda: self._set_brush_mode("rough"))
        tool_vbox.addWidget(self.btn_rough)

        self.btn_gloss = QPushButton("✨ Glossy / Wet")
        self.btn_gloss.setCheckable(True)
        legacy_style(self.btn_gloss, """
            QPushButton:checked {
                background-color: #1473e6;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #0d66d0;
                border-radius: 3px;
                padding: 6px;
                text-align: left;
            }
            QPushButton:!checked {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 6px;
                text-align: left;
            }
        """, "brush-preset")
        self.btn_gloss.clicked.connect(lambda: self._set_brush_mode("gloss"))
        tool_vbox.addWidget(self.btn_gloss)

        self.btn_eraser = QPushButton("🧹 Eraser (Revert)")
        self.btn_eraser.setCheckable(True)
        legacy_style(self.btn_eraser, """
            QPushButton:checked {
                background-color: #1473e6;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #0d66d0;
                border-radius: 3px;
                padding: 6px;
                text-align: left;
            }
            QPushButton:!checked {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 6px;
                text-align: left;
            }
        """, "brush-preset")
        self.btn_eraser.clicked.connect(lambda: self._set_brush_mode("eraser"))
        tool_vbox.addWidget(self.btn_eraser)

        ctrl_layout.addWidget(tool_frame)

        brush_frame = QFrame()
        brush_frame.setObjectName("PainterCard")
        legacy_style(brush_frame, _CARD_STYLE)
        brush_vbox = QVBoxLayout(brush_frame)
        brush_vbox.setSpacing(8)

        param_title = QLabel("BRUSH PARAMETERS")
        legacy_style(param_title, "font-weight: bold; font-size: 10px; color: #8a8a8a; letter-spacing: 0.5px;", "section-caps")
        brush_vbox.addWidget(param_title)

        size_box = QHBoxLayout()
        size_lbl = QLabel("Size:")
        self.size_spin = QSpinBox()
        self.size_spin.setRange(4, 300)
        self.size_spin.setValue(45)
        self.size_spin.setSuffix(" px")
        self.size_spin.valueChanged.connect(self._on_brush_param_changed)
        size_box.addWidget(size_lbl)
        size_box.addWidget(self.size_spin)
        brush_vbox.addLayout(size_box)

        hard_box = QHBoxLayout()
        hard_lbl = QLabel("Hardness:")
        self.hard_spin = QDoubleSpinBox()
        self.hard_spin.setRange(0.0, 1.0)
        self.hard_spin.setSingleStep(0.05)
        self.hard_spin.setValue(0.20)
        self.hard_spin.valueChanged.connect(self._on_brush_param_changed)
        hard_box.addWidget(hard_lbl)
        hard_box.addWidget(self.hard_spin)
        brush_vbox.addLayout(hard_box)

        flow_box = QHBoxLayout()
        flow_lbl = QLabel("Flow / Opacity:")
        self.flow_spin = QDoubleSpinBox()
        self.flow_spin.setRange(0.05, 1.0)
        self.flow_spin.setSingleStep(0.05)
        self.flow_spin.setValue(0.40)
        self.flow_spin.valueChanged.connect(self._on_brush_param_changed)
        flow_box.addWidget(flow_lbl)
        flow_box.addWidget(self.flow_spin)
        brush_vbox.addLayout(flow_box)

        rough_box = QHBoxLayout()
        rough_lbl = QLabel("Roughness Val:")
        self.rough_spin = QDoubleSpinBox()
        self.rough_spin.setRange(0.0, 1.0)
        self.rough_spin.setSingleStep(0.05)
        self.rough_spin.setValue(1.0)
        self.rough_spin.valueChanged.connect(self._on_brush_param_changed)
        rough_box.addWidget(rough_lbl)
        rough_box.addWidget(self.rough_spin)
        brush_vbox.addLayout(rough_box)

        ctrl_layout.addWidget(brush_frame)

        under_frame = QFrame()
        under_frame.setObjectName("PainterCard")
        legacy_style(under_frame, _CARD_STYLE)
        under_vbox = QVBoxLayout(under_frame)
        under_vbox.setSpacing(6)

        under_title = QLabel("UNDERLAY REFERENCE")
        legacy_style(under_title, "font-weight: bold; font-size: 10px; color: #8a8a8a; letter-spacing: 0.5px;", "section-caps")
        under_vbox.addWidget(under_title)

        self.underlay_combo = QComboBox()
        self.underlay_combo.addItems(["Albedo (Color Photo)", "Normal Map", "Base Roughness", "Checkerboard"])
        self.underlay_combo.currentTextChanged.connect(self._on_underlay_changed)
        under_vbox.addWidget(self.underlay_combo)

        opac_box = QHBoxLayout()
        opac_lbl = QLabel("Opacity:")
        self.underlay_opac_spin = QDoubleSpinBox()
        self.underlay_opac_spin.setRange(0.0, 1.0)
        self.underlay_opac_spin.setSingleStep(0.05)
        self.underlay_opac_spin.setValue(0.65)
        self.underlay_opac_spin.valueChanged.connect(lambda v: self.canvas.set_underlay_opacity(v))
        opac_box.addWidget(opac_lbl)
        opac_box.addWidget(self.underlay_opac_spin)
        under_vbox.addLayout(opac_box)

        ctrl_layout.addWidget(under_frame)

        act_box = QVBoxLayout()
        act_box.setSpacing(6)

        hist_row = QHBoxLayout()
        undo_btn = QPushButton(f"↶ Undo ({_shortcut_text(QKeySequence.StandardKey.Undo)})")
        undo_btn.clicked.connect(lambda: self.canvas.undo())
        redo_btn = QPushButton(f"↷ Redo ({_shortcut_text(QKeySequence.StandardKey.Redo)})")
        redo_btn.clicked.connect(lambda: self.canvas.redo())
        hist_row.addWidget(undo_btn)
        hist_row.addWidget(redo_btn)
        act_box.addLayout(hist_row)

        fit_btn = QPushButton("Fit to View")
        fit_btn.clicked.connect(lambda: self.canvas.reset_view())
        act_box.addWidget(fit_btn)

        clear_btn = QPushButton("Clear All Paint")
        clear_btn.clicked.connect(lambda: self.canvas.clear_canvas())
        act_box.addWidget(clear_btn)

        ctrl_layout.addLayout(act_box)
        ctrl_layout.addStretch(1)

        splitter.addWidget(ctrl_panel)

        canvas_container = QWidget()
        canvas_vbox = QVBoxLayout(canvas_container)
        canvas_vbox.setContentsMargins(0, 0, 0, 0)
        canvas_vbox.setSpacing(4)

        init_w, init_h = 512, 512
        if albedo is not None:
            init_w, init_h = albedo.size
        elif normal is not None:
            init_w, init_h = normal.size

        max_dim = max(init_w, init_h)
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            init_w = int(round(init_w * scale))
            init_h = int(round(init_h * scale))

        self.canvas = ImperfectionCanvasWidget(init_w, init_h, self)
        self.canvas.set_underlay_images(albedo=albedo, normal=normal, roughness=roughness)
        if mask is not None:
            self.canvas.load_mask_array(mask)

        canvas_vbox.addWidget(self.canvas, 1)

        status_bar = QLabel(pick(
            mac="💡 Pinch or Wheel: Zoom | Two-Finger Swipe, Middle-Click or ⌥-Drag: Pan | "
                "Click or Pen: Paint (pen pressure sets size and flow)",
            windows="💡 Pinch or Wheel: Zoom | Two-Finger Swipe, Middle-Click or Alt+Drag: Pan | "
                    "Click or Pen: Paint (pen pressure sets size and flow)",
            legacy="💡 Wheel: Zoom | Middle-Click or Alt+Drag: Pan | Left-Click: Paint smooth strokes",
        ))
        legacy_style(status_bar, "color: #64748b; font-size: 10px;", "hint")
        canvas_vbox.addWidget(status_bar)

        splitter.addWidget(canvas_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        undo_act = QAction(self)
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(self.canvas.undo)
        self.addAction(undo_act)

        redo_act = QAction(self)
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(self.canvas.redo)
        self.addAction(redo_act)

        self._sync_brush_to_canvas()

    def _set_brush_mode(self, mode: BrushMode) -> None:
        self.btn_rough.setChecked(mode == "rough")
        self.btn_gloss.setChecked(mode == "gloss")
        self.btn_eraser.setChecked(mode == "eraser")
        self.canvas.brush_mode = mode

        if mode == "rough":
            self.rough_spin.setValue(1.0)
            self.rough_spin.setEnabled(True)
        elif mode == "gloss":
            self.rough_spin.setValue(0.0)
            self.rough_spin.setEnabled(True)
        else:
            self.rough_spin.setEnabled(False)

        self._sync_brush_to_canvas()

    def _on_brush_param_changed(self) -> None:
        self._sync_brush_to_canvas()

    def _sync_brush_to_canvas(self) -> None:
        self.canvas.brush_size = self.size_spin.value()
        self.canvas.brush_hardness = self.hard_spin.value()
        self.canvas.brush_flow = self.flow_spin.value()
        self.canvas.target_roughness = self.rough_spin.value()

    def _on_underlay_changed(self, text: str) -> None:
        key_map = {
            "Albedo (Color Photo)": "albedo",
            "Normal Map": "normal",
            "Base Roughness": "roughness",
            "Checkerboard": "none",
        }
        self.canvas.set_active_underlay(key_map.get(text, "albedo"))

    def result_mask(self) -> np.ndarray | None:
        return self.canvas.get_mask_array()


def cv2_resize_mask(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    import cv2
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
