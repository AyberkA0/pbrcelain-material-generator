"""Entry point for PBRCELAIN — a PBR material authoring tool.

Generates Albedo/Height/Normal/Roughness material maps from source images.
Height estimation uses monocular depth models (Depth Anything V2 /
Marigold); Normal is derived from Height; Roughness is a placeholder for
now.

Usage: python main.py [project.pcln | project.zip]
"""
import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _redirect_missing_std_streams() -> None:
    """Under pythonw.exe (e.g. launched by double-clicking a .pcln file) there
    is no console, so sys.stdout/stderr are None and any library progress bar
    (Hugging Face downloads, tqdm) would crash on write. Send them to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    log_dir = os.path.join(base, "PBRCELAIN")
    try:
        os.makedirs(log_dir, exist_ok=True)
        log = open(os.path.join(log_dir, "pbrcelain.log"), "w", encoding="utf-8", buffering=1)
    except OSError:
        log = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


_redirect_missing_std_streams()

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtWidgets import QApplication

from core.project import is_project_path
from ui.preview_3d import configure_surface_format
from ui.theme import apply_app_theme


def _project_path_from_args(argv: list[str]) -> str | None:
    for arg in argv[1:]:
        if is_project_path(arg) and os.path.isfile(arg):
            return os.path.abspath(arg)
    return None


class _MacFileOpenHandler(QObject):
    """macOS delivers files opened from Finder (double-click, "Open With",
    dropping on the Dock icon) as QFileOpenEvents instead of argv. They can
    arrive before the main window exists, so the first one is held until
    `attach()` is called."""

    def __init__(self, app: QApplication):
        super().__init__(app)
        self._window = None
        self._pending: str | None = None
        app.installEventFilter(self)

    def attach(self, window) -> None:
        self._window = window
        if self._pending is not None:
            path, self._pending = self._pending, None
            QTimer.singleShot(0, lambda: window._load_project_from_path(path))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FileOpen:
            path = event.file()
            if path and is_project_path(path):
                if self._window is None:
                    self._pending = path
                else:
                    QTimer.singleShot(0, lambda: self._window._handle_dropped_project(path))
                return True
        return False


def _set_app_icon(app: QApplication) -> None:
    """Window/taskbar/Dock icon (regenerate the files with make_icons.py)."""
    from PyQt6.QtGui import QIcon

    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "assets")
    if sys.platform == "darwin":
        app.setWindowIcon(QIcon(os.path.join(assets, "mac", "app_icon.png")))
        return
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PBRCELAIN.App")
        except (AttributeError, OSError):
            pass
        app.setWindowIcon(QIcon(os.path.join(assets, "app_icon.ico")))
        return
    app.setWindowIcon(QIcon(os.path.join(assets, "app_icon.png")))


def main() -> int:
    configure_surface_format()
    app = QApplication(sys.argv)
    app.setApplicationName("PBRCELAIN")
    app.setOrganizationName("PBRCELAIN")
    apply_app_theme(app)

    _set_app_icon(app)

    file_open_handler = _MacFileOpenHandler(app) if sys.platform == "darwin" else None

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    if file_open_handler is not None:
        file_open_handler.attach(window)

    project_path = _project_path_from_args(sys.argv)
    if project_path is not None:
        QTimer.singleShot(0, lambda: window._load_project_from_path(project_path))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
