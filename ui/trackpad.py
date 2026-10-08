from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QInputDevice, QNativeGestureEvent, QWheelEvent

from ui.theme import IS_MAC, IS_WINDOWS


def is_trackpad_scroll(event: QWheelEvent) -> bool:
    if IS_MAC:
        return not event.pixelDelta().isNull() and event.phase() != Qt.ScrollPhase.NoScrollPhase
    if IS_WINDOWS:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        device = event.pointingDevice()
        return device is not None and device.type() == QInputDevice.DeviceType.TouchPad
    return False


def scroll_delta(event: QWheelEvent) -> QPointF:
    d = event.pixelDelta()
    if d.isNull():
        a = event.angleDelta()
        return QPointF(a.x() * 0.5, a.y() * 0.5)
    return QPointF(d.x(), d.y())


def pinch_factor(event: QEvent) -> float | None:
    if event.type() != QEvent.Type.NativeGesture:
        return None
    if not isinstance(event, QNativeGestureEvent):
        return None
    if event.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
        return None
    return max(0.2, 1.0 + event.value())


def is_smart_zoom(event: QEvent) -> bool:
    return (
        event.type() == QEvent.Type.NativeGesture
        and isinstance(event, QNativeGestureEvent)
        and event.gestureType() == Qt.NativeGestureType.SmartZoomNativeGesture
    )
