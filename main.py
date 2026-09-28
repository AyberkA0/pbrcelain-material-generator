"""Entry point for PBRCELAIN — a PBR material authoring tool.

Generates Albedo/Height/Normal/Roughness material maps from source images.
Height estimation uses monocular depth models (Depth Anything V2 /
Marigold); Normal is derived from Height; Roughness is a placeholder for
now.

Usage: python main.py [project.pcln]
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

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from core.project import PROJECT_EXTENSION
from ui.preview_3d import configure_surface_format
from ui.theme import APP_STYLESHEET


def _project_path_from_args(argv: list[str]) -> str | None:
    for arg in argv[1:]:
        if arg.lower().endswith(PROJECT_EXTENSION) and os.path.isfile(arg):
            return os.path.abspath(arg)
    return None


def main() -> int:
    configure_surface_format()
    app = QApplication(sys.argv)
    app.setApplicationName("PBRCELAIN")
    app.setOrganizationName("PBRCELAIN")
    app.setStyleSheet(APP_STYLESHEET)

    from ui.main_window import MainWindow

    window = MainWindow()
    window.show()

    project_path = _project_path_from_args(sys.argv)
    if project_path is not None:
        QTimer.singleShot(0, lambda: window._load_project_from_path(project_path))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
