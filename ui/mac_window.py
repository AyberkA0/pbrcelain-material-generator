"""macOS-only window chrome for the main window.

The main window has no visible title bar on macOS: content extends to the
top edge, the traffic-light buttons float over the sidebar, and the menu bar
at the top of the screen carries the commands. Widgets placed in the old
title-bar row become move handles via `install_drag_area`.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt
from PyQt6.QtWidgets import QMainWindow, QSplitter, QWidget

TITLE_BAR_HEIGHT = 28


def hide_title_bar(window: QMainWindow) -> None:
    window.setWindowFlag(Qt.WindowType.ExpandedClientAreaHint, True)
    window.setWindowFlag(Qt.WindowType.NoTitleBarBackgroundHint, True)
    window.setAttribute(Qt.WidgetAttribute.WA_ContentsMarginsRespectsSafeArea, False)


class _DragArea(QObject):
    """Moves the window when `widget` is dragged; double-click zooms it."""

    def __init__(self, widget: QWidget):
        super().__init__(widget)
        self._offset: QPoint | None = None

    def eventFilter(self, obj, event) -> bool:
        etype = event.type()
        if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            window = obj.window()
            self._offset = event.globalPosition().toPoint() - window.frameGeometry().topLeft()
            # Prefer the native move (keeps window snapping/tiling); if the
            # system doesn't take over, the MouseMove branch moves it instead.
            handle = window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
            return True
        if etype == QEvent.Type.MouseMove and self._offset is not None:
            if event.buttons() & Qt.MouseButton.LeftButton:
                obj.window().move(event.globalPosition().toPoint() - self._offset)
                return True
        if etype == QEvent.Type.MouseButtonRelease:
            self._offset = None
        if etype == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
            window = obj.window()
            window.showNormal() if window.isMaximized() else window.showMaximized()
            return True
        return False


def install_drag_area(widget: QWidget) -> None:
    widget.installEventFilter(_DragArea(widget))


def fit_to_screen(window: QMainWindow, sidebar_splitter: QSplitter, preview_splitter: QSplitter) -> None:
    """Size the window and its sections relative to the screen rather than
    fixed pixels, so small and large displays both get a sensible split."""
    screen = window.screen()
    if screen is None:
        return
    avail = screen.availableGeometry()
    w = min(int(avail.width() * 0.94), 2200)
    h = min(int(avail.height() * 0.94), 1400)
    window.setGeometry(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2, w, h)

    # Sidebar ~1/3 of the width (room for the 3-column settings grid).
    side = max(440, min(640, int(w * 0.34)))
    sidebar_splitter.setSizes([side, w - side])

    # Viewport ~60% of the height, View Options the remaining ~40%.
    content_h = h - TITLE_BAR_HEIGHT
    view_h = int(content_h * 0.6)
    preview_splitter.setSizes([view_h, content_h - view_h])
