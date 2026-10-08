from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.theme import legacy_style

class OperationProgressDialog(QDialog):
    def __init__(
        self,
        title: str = "Please Wait",
        initial_status: str = "Processing…",
        parent: QWidget | None = None,
        cancellable: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        self.setFixedSize(400, 140)
        legacy_style(self, """
            QDialog {
                background-color: #323232;
                border: 1px solid #202020;
                border-radius: 3px;
            }
            QLabel[role="dialog-title"] {
                color: #ffffff;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel[role="dialog-status"] {
                color: #8a8a8a;
                font-size: 11px;
            }
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #3c3c3c;
                border-radius: 2px;
                height: 8px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #1473e6;
                border-radius: 1px;
            }
            QPushButton {
                background-color: #3e3e3e;
                color: #dedede;
                border: 1px solid #282828;
                border-radius: 3px;
                padding: 4px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4c4c4c;
                border-color: #555555;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #282828;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 16)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setProperty("role", "dialog-title")
        layout.addWidget(self.title_label)

        self.status_label = QLabel(initial_status)
        self.status_label.setProperty("role", "dialog-status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        if cancellable:
            btn_row = QHBoxLayout()
            btn_row.addStretch()
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.clicked.connect(self.reject)
            btn_row.addWidget(self.cancel_btn)
            layout.addLayout(btn_row)
        else:
            self.cancel_btn = None

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_progress(self, current: int, total: int = 100) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setRange(0, 0)
