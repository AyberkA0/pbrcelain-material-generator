"""Windows-only window integration: a title bar that follows the app's
Light/Dark theme (with the Windows 11 Mica backdrop) and progress shown on
the taskbar button.

Everything here talks to Win32/DWM/COM through ctypes and silently does
nothing where an API isn't available (older Windows versions, other
platforms), so callers don't need their own guards.
"""
from __future__ import annotations

import ctypes
import sys
import uuid

_IS_WINDOWS = sys.platform == "win32"

# DWM window attributes
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20          # Windows 10 20H1+ / 11
_DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19      # Windows 10 1809–1909
_DWMWA_SYSTEMBACKDROP_TYPE = 38              # Windows 11 22H2+
_DWMSBT_MAINWINDOW = 2                       # Mica


def _hwnd(widget) -> int:
    return int(widget.winId())


def _set_dwm_int(hwnd: int, attribute: int, value: int) -> bool:
    data = ctypes.c_int(value)
    result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
        ctypes.c_void_p(hwnd), ctypes.c_uint(attribute), ctypes.byref(data), ctypes.sizeof(data)
    )
    return result == 0


def apply_title_bar_theme(widget, dark: bool) -> None:
    """Dark or light title bar matching the app theme, Mica backdrop on
    Windows 11 (visible in the title bar, the content stays opaque)."""
    if not _IS_WINDOWS:
        return
    try:
        hwnd = _hwnd(widget)
        if not _set_dwm_int(hwnd, _DWMWA_USE_IMMERSIVE_DARK_MODE, int(dark)):
            _set_dwm_int(hwnd, _DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, int(dark))
        _set_dwm_int(hwnd, _DWMWA_SYSTEMBACKDROP_TYPE, _DWMSBT_MAINWINDOW)
    except (AttributeError, OSError):
        pass


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def from_string(cls, text: str) -> "_GUID":
        u = uuid.UUID(text)
        guid = cls()
        guid.Data1, guid.Data2, guid.Data3 = u.fields[0], u.fields[1], u.fields[2]
        guid.Data4[:] = u.bytes[8:]
        return guid


_CLSID_TASKBAR_LIST = "{56FDF344-FD6D-11d0-958A-006097C9A090}"
_IID_ITASKBAR_LIST3 = "{ea1afb91-9e28-4b86-90e9-9e9f8a5eefaf}"
_CLSCTX_INPROC_SERVER = 1

# ITaskbarList3 vtable slots (IUnknown 0-2, ITaskbarList 3-7, ITaskbarList2 8,
# SetProgressValue 9)
_SLOT_HR_INIT = 3
_SLOT_SET_PROGRESS_STATE = 10

TBPF_NOPROGRESS = 0x0
TBPF_INDETERMINATE = 0x1


class TaskbarProgress:
    """Progress on the window's taskbar button (ITaskbarList3)."""

    def __init__(self, widget):
        self._widget = widget
        self._taskbar = None
        if not _IS_WINDOWS:
            return
        try:
            ctypes.windll.ole32.CoInitialize(None)
            ptr = ctypes.c_void_p()
            clsid = _GUID.from_string(_CLSID_TASKBAR_LIST)
            iid = _GUID.from_string(_IID_ITASKBAR_LIST3)
            hr = ctypes.windll.ole32.CoCreateInstance(
                ctypes.byref(clsid), None, _CLSCTX_INPROC_SERVER, ctypes.byref(iid), ctypes.byref(ptr)
            )
            if hr == 0 and ptr.value:
                self._taskbar = ptr
                self._call(_SLOT_HR_INIT, [])
        except (AttributeError, OSError):
            self._taskbar = None

    def _call(self, slot: int, args: list, argtypes: list | None = None) -> None:
        vtable = ctypes.cast(self._taskbar, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *(argtypes or []))
        prototype(vtable[slot])(self._taskbar, *args)

    def _set_state(self, state: int) -> None:
        if self._taskbar is None:
            return
        try:
            self._call(_SLOT_SET_PROGRESS_STATE, [ctypes.c_void_p(_hwnd(self._widget)), state],
                       [ctypes.c_void_p, ctypes.c_int])
        except OSError:
            pass

    def set_busy(self, busy: bool) -> None:
        """Indeterminate (pulsing) progress while busy, cleared otherwise."""
        self._set_state(TBPF_INDETERMINATE if busy else TBPF_NOPROGRESS)
