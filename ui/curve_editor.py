"""Interactive dual-axis curve editor for the bowl/dome correction profile.

Supports smooth Monotone Cubic Bezier curves for independent Horizontal (X)
and Vertical (Y) lens curvature profiles.

The curve maps normalized distance from image center (0 = center, 1 = edge)
to a correction multiplier applied to the auto-fitted quadratic bowl/dome
surface. A flat curve at 1.0 reproduces the auto-fitted behavior; dragging
points scales the correction up or down per-axis.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap

from ui.theme import IS_MAC, legacy_style, pick
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import numpy as np

from core.height_map import (
    DEFAULT_BOWL_CURVE,
    HeightMapOptions,
    apply_gamma,
    compute_bezier_segments,
    downsample_for_preview,
    evaluate_bezier_curve,
    fit_polynomial_surface,
    normalize,
)
from ui.widgets import ImageLabel

Y_MIN, Y_MAX = 0.0, 2.0
POINT_HIT_RADIUS = 9
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 46, 20, 24, 36

COLOR_X = QColor("#38bdf8")
COLOR_X_GHOST = QColor(56, 189, 248, 85)
COLOR_Y = QColor("#fbbf24")
COLOR_Y_GHOST = QColor(251, 191, 36, 85)


class CurveEditorWidget(QWidget):
    curveChanged = pyqtSignal()

    def __init__(
        self,
        curve_x: list[tuple[float, float]] | None = None,
        curve_y: list[tuple[float, float]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        cx = curve_x or list(DEFAULT_BOWL_CURVE)
        cy = curve_y or list(DEFAULT_BOWL_CURVE)
        self.points_x: list[list[float]] = [[float(x), float(y)] for x, y in cx]
        self.points_y: list[list[float]] = [[float(x), float(y)] for x, y in cy]

        self._active_axis = "x"
        self._link_axes = False
        self._drag_index: int | None = None
        self._hover_pos: QPointF | None = None
        self._selected_index: int | None = None
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._ensure_endpoints(self.points_x)
        self._ensure_endpoints(self.points_y)

        self.setMinimumSize(420, 280)
        self.setMouseTracking(True)

    @staticmethod
    def _ensure_endpoints(pts: list[list[float]]) -> None:
        pts.sort(key=lambda p: p[0])
        if not pts or pts[0][0] > 0.0:
            pts.insert(0, [0.0, 1.0])
        if pts[-1][0] < 1.0:
            pts.append([1.0, 1.0])
        pts[0][0] = 0.0
        pts[-1][0] = 1.0

    @property
    def active_axis(self) -> str:
        return self._active_axis

    def set_active_axis(self, axis: str) -> None:
        if axis in ("x", "y") and axis != self._active_axis:
            self._active_axis = axis
            self._drag_index = None
            self._selected_index = None
            self.update()

    @property
    def link_axes(self) -> bool:
        return self._link_axes

    def set_link_axes(self, linked: bool) -> None:
        self._link_axes = linked
        if linked:
            if self._active_axis == "x":
                self.points_y = [[p[0], p[1]] for p in self.points_x]
            else:
                self.points_x = [[p[0], p[1]] for p in self.points_y]
            self.update()
            self.curveChanged.emit()

    def _active_points(self) -> list[list[float]]:
        return self.points_x if self._active_axis == "x" else self.points_y

    def _inactive_points(self) -> list[list[float]]:
        return self.points_y if self._active_axis == "x" else self.points_x

    def get_points_x(self) -> list[tuple[float, float]]:
        return [(p[0], p[1]) for p in self.points_x]

    def get_points_y(self) -> list[tuple[float, float]]:
        return [(p[0], p[1]) for p in self.points_y]

    def get_points(self) -> list[tuple[float, float]]:
        """Legacy helper returning active points."""
        return [(p[0], p[1]) for p in self._active_points()]

    def reset_flat_active(self) -> None:
        pts = [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
        if self._active_axis == "x":
            self.points_x = pts
        else:
            self.points_y = pts
        if self._link_axes:
            if self._active_axis == "x":
                self.points_y = [[p[0], p[1]] for p in self.points_x]
            else:
                self.points_x = [[p[0], p[1]] for p in self.points_y]
        self._drag_index = None
        self.update()
        self.curveChanged.emit()

    def reset_flat_both(self) -> None:
        self.points_x = [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
        self.points_y = [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
        self._drag_index = None
        self.update()
        self.curveChanged.emit()

    def copy_active_to_other(self) -> None:
        if self._active_axis == "x":
            self.points_y = [[p[0], p[1]] for p in self.points_x]
        else:
            self.points_x = [[p[0], p[1]] for p in self.points_y]
        self.update()
        self.curveChanged.emit()

    def reset_flat(self) -> None:
        self.reset_flat_both()

    def _plot_rect(self) -> QRectF:
        return QRectF(
            MARGIN_LEFT,
            MARGIN_TOP,
            max(1, self.width() - MARGIN_LEFT - MARGIN_RIGHT),
            max(1, self.height() - MARGIN_TOP - MARGIN_BOTTOM),
        )

    def _data_to_pixel(self, x: float, y: float) -> QPointF:
        rect = self._plot_rect()
        px = rect.left() + x * rect.width()
        py = rect.top() + (1.0 - (y - Y_MIN) / (Y_MAX - Y_MIN)) * rect.height()
        return QPointF(px, py)

    def _pixel_to_data(self, px: float, py: float) -> tuple[float, float]:
        rect = self._plot_rect()
        x = (px - rect.left()) / rect.width()
        y = Y_MIN + (1.0 - (py - rect.top()) / rect.height()) * (Y_MAX - Y_MIN)
        return x, y

    def _point_at(self, pos: QPointF, pts: list[list[float]]) -> int | None:
        for i, (x, y) in enumerate(pts):
            p = self._data_to_pixel(x, y)
            dist_sq = (p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2
            if dist_sq <= (POINT_HIT_RADIUS * 1.5) ** 2:
                return i
        return None

    def _draw_bezier_path(
        self,
        painter: QPainter,
        points: list[list[float]],
        color: QColor,
        width: float,
        pen_style: Qt.PenStyle = Qt.PenStyle.SolidLine,
    ) -> None:
        pts_tuple = [(p[0], p[1]) for p in points]
        segments = compute_bezier_segments(pts_tuple)
        if not segments:
            if points:
                p = self._data_to_pixel(points[0][0], points[0][1])
                painter.drawPoint(p)
            return

        pen = QPen(color, width, pen_style, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        p0 = self._data_to_pixel(segments[0][0][0], segments[0][0][1])
        path.moveTo(p0)
        for _, c1, c2, p1 in segments:
            c1_px = self._data_to_pixel(c1[0], c1[1])
            c2_px = self._data_to_pixel(c2[0], c2[1])
            p1_px = self._data_to_pixel(p1[0], p1[1])
            path.cubicTo(c1_px, c2_px, p1_px)

        painter.drawPath(path)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._plot_rect()

        painter.fillRect(self.rect(), QColor("#18181b"))
        painter.fillRect(rect, QColor("#0e0e11"))

        grid_pen = QPen(QColor("#27272a"), 1.0)
        painter.setPen(grid_pen)
        for i in range(5):
            x = i / 4.0
            painter.drawLine(self._data_to_pixel(x, Y_MIN), self._data_to_pixel(x, Y_MAX))
        for i in range(5):
            y = Y_MIN + (Y_MAX - Y_MIN) * i / 4.0
            painter.drawLine(self._data_to_pixel(0.0, y), self._data_to_pixel(1.0, y))

        ref_pen = QPen(QColor("#52525b"), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(ref_pen)
        painter.drawLine(self._data_to_pixel(0.0, 1.0), self._data_to_pixel(1.0, 1.0))

        label_font = QFont(self.font())
        label_font.setPointSize(pick(mac=10, windows=8, legacy=8))
        painter.setFont(label_font)
        painter.setPen(QColor("#71717a"))

        painter.drawText(
            QRectF(0, rect.bottom() + 4, self.width(), 20),
            Qt.AlignmentFlag.AlignHCenter,
            "Center (0.0)  ────────  Normalized Radius  ────────  Edge (1.0)",
        )

        painter.drawText(QRectF(rect.left() - 42, rect.bottom() - 8, 36, 16), Qt.AlignmentFlag.AlignRight, "0.0")
        painter.drawText(QRectF(rect.left() - 42, self._data_to_pixel(0, 0.5).y() - 8, 36, 16), Qt.AlignmentFlag.AlignRight, "0.5")
        painter.drawText(QRectF(rect.left() - 42, self._data_to_pixel(0, 1.0).y() - 8, 36, 16), Qt.AlignmentFlag.AlignRight, "1.0")
        painter.drawText(QRectF(rect.left() - 42, self._data_to_pixel(0, 1.5).y() - 8, 36, 16), Qt.AlignmentFlag.AlignRight, "1.5")
        painter.drawText(QRectF(rect.left() - 42, rect.top() - 8, 36, 16), Qt.AlignmentFlag.AlignRight, "2.0")

        inactive_is_x = (self._active_axis == "y")
        ghost_color = COLOR_X_GHOST if inactive_is_x else COLOR_Y_GHOST
        ghost_pts = self._inactive_points()
        self._draw_bezier_path(painter, ghost_pts, ghost_color, width=1.8, pen_style=Qt.PenStyle.DashLine)

        for x, y in ghost_pts:
            p = self._data_to_pixel(x, y)
            painter.setPen(QPen(QColor(0, 0, 0, 100), 1.0))
            painter.setBrush(ghost_color)
            painter.drawEllipse(p, 3.5, 3.5)

        active_color = COLOR_X if self._active_axis == "x" else COLOR_Y
        active_pts = self._active_points()
        self._draw_bezier_path(painter, active_pts, active_color, width=2.4, pen_style=Qt.PenStyle.SolidLine)

        for i, (x, y) in enumerate(active_pts):
            p = self._data_to_pixel(x, y)
            is_endpoint = (i == 0 or i == len(active_pts) - 1)
            is_dragged = (i == self._drag_index)

            painter.setPen(QPen(QColor("#09090b"), 1.8))
            if is_dragged:
                painter.setBrush(QColor("#f43f5e"))
                painter.drawEllipse(p, 7.0, 7.0)
            elif is_endpoint:
                painter.setBrush(active_color)
                painter.drawEllipse(p, 5.5, 5.5)
            else:
                painter.setBrush(QColor("#ffffff"))
                painter.drawEllipse(p, 5.5, 5.5)

        legend_rect = QRectF(rect.right() - 200, rect.top() + 6, 194, 22)
        painter.fillRect(legend_rect, QColor(24, 24, 27, 180))
        painter.setPen(QPen(QColor("#3f3f46"), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(legend_rect, 4, 4)

        painter.setPen(COLOR_X if self._active_axis == "x" else COLOR_X_GHOST)
        painter.drawText(QRectF(legend_rect.left() + 8, legend_rect.top() + 3, 90, 16), Qt.AlignmentFlag.AlignLeft, "─ X (Horiz)")
        painter.setPen(COLOR_Y if self._active_axis == "y" else COLOR_Y_GHOST)
        painter.drawText(QRectF(legend_rect.left() + 102, legend_rect.top() + 3, 90, 16), Qt.AlignmentFlag.AlignLeft, "─ Y (Vert)")

        if self._drag_index is not None and 0 <= self._drag_index < len(active_pts):
            dx, dy = active_pts[self._drag_index]
            drag_p = self._data_to_pixel(dx, dy)
            info_text = f"Dist: {dx:.2f} | Mult: {dy:.2f}×"
            info_box = QRectF(drag_p.x() + 10, drag_p.y() - 24, 115, 20)
            if info_box.right() > rect.right():
                info_box.moveLeft(drag_p.x() - 125)
            painter.fillRect(info_box, QColor(0, 0, 0, 210))
            painter.setPen(QColor("#e4e4e7"))
            painter.drawText(info_box, Qt.AlignmentFlag.AlignCenter, info_text)

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        pts = self._active_points()
        idx = self._point_at(pos, pts)

        is_delete_click = event.button() == Qt.MouseButton.RightButton or (
            IS_MAC
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.MetaModifier
        )

        if is_delete_click:
            self._delete_point(idx)
        elif event.button() == Qt.MouseButton.LeftButton:
            if idx is None:
                x, y = self._pixel_to_data(pos.x(), pos.y())
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, Y_MIN), Y_MAX)
                insert_at = len(pts)
                for i, p in enumerate(pts):
                    if p[0] > x:
                        insert_at = i
                        break
                pts.insert(insert_at, [x, y])
                idx = insert_at
                if self._link_axes:
                    other = self._inactive_points()
                    other.insert(insert_at, [x, y])

            self._drag_index = idx
            self._selected_index = idx
            self.update()
            self.curveChanged.emit()

    def _delete_point(self, idx: int | None) -> None:
        """Remove an interior control point (the two endpoints are fixed)."""
        pts = self._active_points()
        if idx is not None and idx != 0 and idx != len(pts) - 1:
            pts.pop(idx)
            if self._link_axes:
                other = self._inactive_points()
                if idx < len(other):
                    other.pop(idx)
            self._selected_index = None
            self.update()
            self.curveChanged.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_point(self._selected_index)
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        self._hover_pos = pos

        pts = self._active_points()
        if self._drag_index is None:
            hover_idx = self._point_at(pos, pts)
            if hover_idx is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        x, y = self._pixel_to_data(pos.x(), pos.y())
        y = min(max(y, Y_MIN), Y_MAX)
        i = self._drag_index

        if i == 0:
            x = 0.0
        elif i == len(pts) - 1:
            x = 1.0
        else:
            lo = pts[i - 1][0] + 0.01
            hi = pts[i + 1][0] - 0.01
            x = min(max(x, lo), hi) if hi > lo else pts[i][0]

        pts[i] = [x, y]
        if self._link_axes:
            other = self._inactive_points()
            if i < len(other):
                other[i] = [x, y]

        self.update()
        self.curveChanged.emit()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_index is not None:
            self._drag_index = None
            self.update()
            self.curveChanged.emit()


class CurveEditorDialog(QDialog):
    """Dialog housing dual-axis Monotone Bezier curve controls with side-by-side live image preview."""

    liveCurvesChanged = pyqtSignal(list, list)

    def __init__(
        self,
        curve_x: list[tuple[float, float]] | None = None,
        curve_y: list[tuple[float, float]] | None = None,
        depth: np.ndarray | None = None,
        options: HeightMapOptions | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Dual-Axis Lens Bowl/Dome Correction & Live Preview")
        self.resize(980, 560)

        cx = curve_x or list(DEFAULT_BOWL_CURVE)
        cy = curve_y or cx or list(DEFAULT_BOWL_CURVE)
        self._options = options or HeightMapOptions()

        self._preview_depth: np.ndarray | None = None
        self._surface: np.ndarray | None = None
        self._nx: np.ndarray | None = None
        self._ny: np.ndarray | None = None
        self._raw_pixmap: QPixmap | None = None
        self._live_pixmap: QPixmap | None = None

        if depth is not None and depth.size > 0:
            self._preview_depth = downsample_for_preview(depth, max_dim=512)
            self._surface = fit_polynomial_surface(self._preview_depth, order=2)
            h, w = self._preview_depth.shape
            cy_coord, cx_coord = (h - 1) / 2.0, (w - 1) / 2.0
            self._nx = np.clip(np.abs(np.arange(w, dtype=np.float32) - cx_coord) / max(cx_coord, 1.0), 0.0, 1.0)
            self._ny = np.clip(np.abs(np.arange(h, dtype=np.float32) - cy_coord) / max(cy_coord, 1.0), 0.0, 1.0)
            self._raw_pixmap = self._array_to_pixmap(self._preview_depth)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        info = QLabel(
            "<b>Dual-Axis Lens Profile:</b> Compensates for anisotropic horizontal & vertical lens curvature.<br>"
            "• <b>Left-Click & Drag</b>: Smooth Bezier curve control points with instant live preview.<br>"
            + (
                "• <b>Right-Click</b>, <b>Control-Click</b> or <b>Delete</b> key: Delete point. "
                if IS_MAC else
                "• <b>Right-Click</b> or <b>Delete</b> key: Delete point. "
            )
            + "(1.0 = auto-fitted shape, 0.0 = flat baseline)."
        )
        info.setWordWrap(True)
        legacy_style(info, "color: #a1a1aa; font-size: 11px;", "hint")
        main_layout.addWidget(info)

        split_layout = QHBoxLayout()
        split_layout.setSpacing(16)

        left_col = QVBoxLayout()
        left_col.setSpacing(8)

        axis_row = QHBoxLayout()
        axis_row.setSpacing(8)

        self.btn_x = QPushButton("↔ Horizontal (X)")
        self.btn_x.setCheckable(True)
        self.btn_x.setChecked(True)
        legacy_style(self.btn_x, """
            QPushButton:checked {
                background-color: #1473e6;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #0d66d0;
                border-radius: 3px;
                padding: 5px 12px;
            }
            QPushButton:!checked {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 5px 12px;
            }
        """)
        self.btn_x.clicked.connect(lambda: self._set_axis("x"))
        axis_row.addWidget(self.btn_x)

        self.btn_y = QPushButton("↕ Vertical (Y)")
        self.btn_y.setCheckable(True)
        self.btn_y.setChecked(False)
        legacy_style(self.btn_y, """
            QPushButton:checked {
                background-color: #1473e6;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #0d66d0;
                border-radius: 3px;
                padding: 5px 12px;
            }
            QPushButton:!checked {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 5px 12px;
            }
        """)
        self.btn_y.clicked.connect(lambda: self._set_axis("y"))
        axis_row.addWidget(self.btn_y)

        axis_row.addSpacing(10)

        self.link_checkbox = QCheckBox("🔗 Link X && Y")
        self.link_checkbox.setToolTip("Synchronize both axes symmetrically for spherical/isotropic lenses.")
        legacy_style(self.link_checkbox, "color: #e4e4e7; font-size: 11px;")
        self.link_checkbox.toggled.connect(self._on_link_toggled)
        axis_row.addWidget(self.link_checkbox)

        axis_row.addStretch(1)
        left_col.addLayout(axis_row)

        self.editor = CurveEditorWidget(cx, cy, self)
        left_col.addWidget(self.editor, 1)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        reset_active_btn = QPushButton("Reset Active (1.0)")
        reset_active_btn.setToolTip("Reset the currently selected axis to a neutral flat line.")
        reset_active_btn.clicked.connect(self.editor.reset_flat_active)
        action_row.addWidget(reset_active_btn)

        reset_both_btn = QPushButton("Reset Both")
        reset_both_btn.setToolTip("Reset both Horizontal and Vertical curves to 1.0.")
        reset_both_btn.clicked.connect(self.editor.reset_flat_both)
        action_row.addWidget(reset_both_btn)

        copy_btn = QPushButton("Copy Active → Other")
        copy_btn.setToolTip("Copy the active curve profile to the other axis.")
        copy_btn.clicked.connect(self.editor.copy_active_to_other)
        action_row.addWidget(copy_btn)

        action_row.addStretch(1)
        left_col.addLayout(action_row)

        split_layout.addLayout(left_col, 5)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        preview_header = QHBoxLayout()
        preview_title = QLabel("LIVE HEIGHT PREVIEW")
        legacy_style(preview_title, "font-weight: bold; font-size: 11px; color: #a1a1aa; letter-spacing: 0.5px;", "section-caps")
        preview_header.addWidget(preview_title)

        preview_header.addStretch(1)

        self.compare_btn = QPushButton("Show Original (Raw)")
        self.compare_btn.setCheckable(True)
        self.compare_btn.setToolTip("Toggle to compare between raw uncorrected depth and current curve correction.")
        legacy_style(self.compare_btn, """
            QPushButton:checked {
                background-color: #1473e6;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #0d66d0;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:!checked {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 4px 10px;
                font-size: 11px;
            }
        """)
        self.compare_btn.toggled.connect(self._on_compare_toggled)
        if self._preview_depth is None:
            self.compare_btn.setEnabled(False)
        preview_header.addWidget(self.compare_btn)

        right_col.addLayout(preview_header)

        self.preview_label = ImageLabel("No Height map source loaded.\n(Upload or generate a map to see live preview)", self)
        self.preview_label.setMinimumSize(320, 260)
        legacy_style(self.preview_label, """
            QLabel {
                background-color: #181818;
                border: 1px solid #202020;
                border-radius: 2px;
                color: #8a8a8a;
                font-size: 11px;
            }
        """)
        right_col.addWidget(self.preview_label, 1)

        preview_status = QLabel("⚡ Live 60 FPS update • Real-time dual-axis correction")
        legacy_style(preview_status, "color: #71717a; font-size: 10px;", "hint")
        right_col.addWidget(preview_status)

        split_layout.addLayout(right_col, 5)

        main_layout.addLayout(split_layout, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.editor.curveChanged.connect(self._on_curve_changed)
        self._update_preview()

    def _array_to_pixmap(self, arr: np.ndarray) -> QPixmap:
        normed = normalize(arr, self._options.low_percentile, self._options.high_percentile)
        if self._options.invert:
            normed = 1.0 - normed
        if self._options.gamma != 1.0:
            normed = apply_gamma(normed, self._options.gamma)
        h, w = normed.shape
        arr8 = np.ascontiguousarray((normed * 255.0).clip(0, 255).astype(np.uint8))
        qimg = QImage(arr8.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())

    def _on_curve_changed(self) -> None:
        self._update_preview()
        self.liveCurvesChanged.emit(self.editor.get_points_x(), self.editor.get_points_y())

    def _update_preview(self) -> None:
        if self._preview_depth is None or self._surface is None or self._nx is None or self._ny is None:
            return

        cx_pts = self.editor.get_points_x()
        cy_pts = self.editor.get_points_y()

        mx = evaluate_bezier_curve(cx_pts, self._nx)
        my = evaluate_bezier_curve(cy_pts, self._ny)
        mult = my[:, np.newaxis] * mx[np.newaxis, :]
        working = self._preview_depth - self._surface * mult

        self._live_pixmap = self._array_to_pixmap(working)
        if not self.compare_btn.isChecked():
            self.preview_label.set_pixmap_source(self._live_pixmap)

    def _on_compare_toggled(self, checked: bool) -> None:
        if checked:
            self.preview_label.set_pixmap_source(self._raw_pixmap)
        else:
            self.preview_label.set_pixmap_source(self._live_pixmap)

    def _set_axis(self, axis: str) -> None:
        self.btn_x.setChecked(axis == "x")
        self.btn_y.setChecked(axis == "y")
        self.editor.set_active_axis(axis)

    def _on_link_toggled(self, checked: bool) -> None:
        self.editor.set_link_axes(checked)

    def result_curves(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return (curve_x, curve_y)."""
        return self.editor.get_points_x(), self.editor.get_points_y()

    def result_points(self) -> list[tuple[float, float]]:
        """Legacy helper returning active or X curve points."""
        return self.editor.get_points_x()
