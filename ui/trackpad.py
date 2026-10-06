"""Trackpad helpers for the canvases (3D viewport, tiled preview, painter).

On a Mac trackpad a two-finger swipe arrives as a stream of wheel events
with pixel deltas, and pinching arrives as native zoom gestures. Treating
the swipe like a mouse wheel (zoom) makes the view jump around, so canvases
use these helpers to pan/orbit on two-finger swipes and zoom on pinch,
while a real mouse wheel keeps zooming as before.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QNativeGestureEvent, QWheelEvent

from ui.theme import IS_MAC


def is_trackpad_scroll(event: QWheelEvent) -> bool:
    """True for a two-finger trackpad swipe (incl. its momentum tail), False
    for a physical mouse wheel. Mouse wheels report no scroll phase."""
    if not IS_MAC:
        return False
    if event.pixelDelta().isNull():
        return False
    return event.phase() != Qt.ScrollPhase.NoScrollPhase


def scroll_delta(event: QWheelEvent) -> QPointF:
    """Swipe distance in widget pixels, following the system's natural
    scrolling setting (content moves with the fingers)."""
    d = event.pixelDelta()
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
