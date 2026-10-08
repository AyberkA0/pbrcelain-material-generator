from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PyQt6.QtWidgets import QLabel, QPushButton

from ui.theme import PLATFORM_THEME


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class SlotButton(QPushButton):
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class ImageLabel(QLabel):
    imageDropped = pyqtSignal(str)
    fileDropped = pyqtSignal(str)
    doubleClicked = pyqtSignal()

    def __init__(self, placeholder_text: str = "No image", parent=None):
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self._placeholder_text = placeholder_text
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        if PLATFORM_THEME == "legacy":
            self.setStyleSheet("background-color: #1a1a1a; color: #8a8a8a; border: 1px solid #202020; border-radius: 2px;")
        self.setText(placeholder_text)
        self.setAcceptDrops(True)

    def set_pixmap_source(self, pixmap: QPixmap | None) -> None:
        self._source_pixmap = pixmap
        if pixmap is None:
            self.setText(self._placeholder_text)
            self.setToolTip("")
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self._rescale()
            self.setToolTip("Double-click to open 2x2 Tiled Seamless Preview")
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def clear_image(self) -> None:
        self.set_pixmap_source(None)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def _rescale(self) -> None:
        if self._source_pixmap is None:
            return
        dpr = self.devicePixelRatioF()
        scaled = self._source_pixmap.scaled(
            self.size() * dpr,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled.setDevicePixelRatio(dpr)
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.DevicePixelRatioChange:
            self._rescale()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            super().dropEvent(event)
            return
        path = urls[0].toLocalFile()
        event.acceptProposedAction()
        if path.lower().endswith(IMAGE_EXTENSIONS):
            self.imageDropped.emit(path)
        else:
            self.fileDropped.emit(path)
