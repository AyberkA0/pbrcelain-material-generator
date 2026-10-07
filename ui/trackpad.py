"""Trackpad helpers for the canvases (3D viewport, tiled preview, painter).

On a Mac trackpad a two-finger swipe arrives as a stream of wheel events
with pixel deltas and pinching as native zoom gestures; Windows precision
touchpads send pinches as Ctrl+wheel. Treating the swipe like a mouse wheel
(zoom) makes the view jump around, so canvases use these helpers to
pan/orbit on two-finger swipes and zoom on pinch, while a real mouse wheel
keeps zooming as before.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QInputDevice, QNativeGestureEvent, QWheelEvent

from ui.theme import IS_MAC, IS_WINDOWS


def is_trackpad_scroll(event: QWheelEvent) -> bool:
    """True for a two-finger trackpad/touchpad swipe, False for a mouse wheel.

    macOS: swipes carry a scroll phase (incl. the momentum tail), mouse
    wheels don't. Windows: only when Qt reports the device as a touchpad;
    a precision-touchpad pinch arrives as Ctrl+wheel and stays a zoom."""
    if IS_MAC:
        return not event.pixelDelta().isNull() and event.phase() != Qt.ScrollPhase.NoScrollPhase
    if IS_WINDOWS:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        device = event.pointingDevice()
        return device is not None and device.type() == QInputDevice.DeviceType.TouchPad
    return False


def scroll_delta(event: QWheelEvent) -> QPointF:
    """Swipe distance in widget pixels, following the system's natural
    scrolling setting (content moves with the fingers)."""
    d = event.pixelDelta()
    if d.isNull():
        a = event.angleDelta()
        return QPointF(a.x() * 0.5, a.y() * 0.5)
    return QPointF(d.x(), d.y())


def pinch_factor(event: QEvent) -> float | None:
    """Zoom multiplier for a pinch gesture event, or None if `event` isn't one."""
    if event.type() != QEvent.Type.NativeGesture:
        return None
    if not isinstance(event, QNativeGestureEvent):
        return None
    if event.gestureType() != Qt.NativeGestureType.ZoomNativeGesture:
        return None
    return max(0.2, 1.0 + event.value())


def is_smart_zoom(event: QEvent) -> bool:
    """Two-finger double tap (macOS "smart zoom")."""
    return (
        event.type() == QEvent.Type.NativeGesture
        and isinstance(event, QNativeGestureEvent)
        and event.gestureType() == Qt.NativeGestureType.SmartZoomNativeGesture
    )
