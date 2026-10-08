from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSettings
from PyQt6.QtWidgets import QMainWindow, QSplitter

_KEYS = ("window/geometry", "window/sidebarSplit", "window/previewSplit")


def fit_to_screen(window: QMainWindow, sidebar_splitter: QSplitter, preview_splitter: QSplitter,
                  sidebar_ratio: float, sidebar_min: int, sidebar_max: int) -> None:
    screen = window.screen()
    if screen is None:
        return
    avail = screen.availableGeometry()
    w = min(int(avail.width() * 0.94), 2200)
    h = min(int(avail.height() * 0.94), 1400)
    window.setGeometry(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2, w, h)

    side = max(sidebar_min, min(sidebar_max, int(w * sidebar_ratio)))
    sidebar_splitter.setSizes([side, w - side])
    preview_splitter.setSizes([600, 400])


def restore(window: QMainWindow, sidebar_splitter: QSplitter, preview_splitter: QSplitter) -> bool:
    settings = QSettings()
    geometry, sidebar, preview = (settings.value(k) for k in _KEYS)
    if not isinstance(geometry, QByteArray) or not window.restoreGeometry(geometry):
        return False
    if isinstance(sidebar, QByteArray):
        sidebar_splitter.restoreState(sidebar)
    if isinstance(preview, QByteArray):
        preview_splitter.restoreState(preview)
    return True


def save(window: QMainWindow, sidebar_splitter: QSplitter, preview_splitter: QSplitter) -> None:
    settings = QSettings()
    for key, value in zip(_KEYS, (window.saveGeometry(), sidebar_splitter.saveState(), preview_splitter.saveState())):
        settings.setValue(key, value)
