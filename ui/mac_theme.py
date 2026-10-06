"""macOS theme for PBRCELAIN.

Qt's native macOS style ignores most of the QSS box model (it adds its own
layout margins, centers QFormLayouts and shrinks spin-box arrows), which is
what makes widgets drift and squeeze on a Mac. On macOS the app therefore
runs on the Fusion style with this stylesheet, which follows the macOS
design language: system font, rounded controls, segmented tabs and the
user's system accent color.

The theme follows the system appearance automatically: it is rebuilt from
the light or dark token set whenever macOS switches between Light and Dark
mode.

Windows/Linux never import this module; they keep `theme.APP_STYLESHEET`.
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor, QFontDatabase, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication, QComboBox

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mac").replace("\\", "/")

DARK = {
    "window": "#1e1e1e",
    "sidebar": "#262626",
    "toolbar": "#2a2a2a",
    "separator": "#0f0f0f",
    "group": "rgba(255, 255, 255, 0.045)",
    "hairline": "rgba(255, 255, 255, 0.09)",
    "text": "#e5e5e7",
    "strong": "#ffffff",
    "secondary": "#98989d",
    "tertiary": "#6e6e73",
    "control": "#3a3a3c",
    "control_hover": "#444446",
    "control_pressed": "#545456",
    "control_border": "#3a3a3c",
    "control_border_top": "rgba(255, 255, 255, 0.12)",
    "control_disabled": "#2c2c2e",
    "field": "#1c1c1e",
    "field_border": "#3d3d40",
    "field_disabled": "#262628",
    "field_disabled_border": "#2e2e30",
    "spin_button": "#2c2c2e",
    "popup": "#2c2c2e",
    "popup_border": "#48484a",
    "hover_overlay": "rgba(255, 255, 255, 0.08)",
    "segment_track": "#2c2c2e",
    "segment_selected": "#5a5a5e",
    "check": "#3a3a3c",
    "check_border": "#58585c",
    "check_disabled": "#2a2a2c",
    "scroll": "rgba(255, 255, 255, 0.22)",
    "scroll_hover": "rgba(255, 255, 255, 0.38)",
    "progress_track": "#3a3a3c",
    "well": "#161616",
    "well_border": "rgba(255, 255, 255, 0.09)",
    "well_text": "#6e6e73",
    "canvas": "#141414",
    "slot_checked_alpha": 0.22,
    "arrow_suffix": "",
}

LIGHT = {
    "window": "#ececee",
    "sidebar": "#f6f6f8",
    "toolbar": "#f0f0f2",
    "separator": "#d4d4d8",
    "group": "rgba(0, 0, 0, 0.035)",
    "hairline": "rgba(0, 0, 0, 0.10)",
    "text": "#1d1d1f",
    "strong": "#000000",
    "secondary": "#6e6e73",
    "tertiary": "#a1a1a6",
    "control": "#ffffff",
    "control_hover": "#f4f4f6",
    "control_pressed": "#e4e4e7",
    "control_border": "rgba(0, 0, 0, 0.14)",
    "control_border_top": "rgba(0, 0, 0, 0.10)",
    "control_disabled": "#f2f2f4",
    "field": "#ffffff",
    "field_border": "#c7c7cc",
    "field_disabled": "#f2f2f4",
    "field_disabled_border": "#dcdce0",
    "spin_button": "#f5f5f7",
    "popup": "#ffffff",
    "popup_border": "rgba(0, 0, 0, 0.15)",
    "hover_overlay": "rgba(0, 0, 0, 0.06)",
    "segment_track": "#e3e3e6",
    "segment_selected": "#ffffff",
    "check": "#ffffff",
    "check_border": "#b8b8bd",
    "check_disabled": "#f2f2f4",
    "scroll": "rgba(0, 0, 0, 0.25)",
    "scroll_hover": "rgba(0, 0, 0, 0.42)",
    "progress_track": "#d9d9dc",
    "well": "#e3e3e6",
    "well_border": "rgba(0, 0, 0, 0.10)",
    "well_text": "#8e8e93",
    "canvas": "#141414",
    "slot_checked_alpha": 0.14,
    "arrow_suffix": "-light",
}

GREEN = "#30a46c"


def is_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return True
    return app.styleHints().colorScheme() != Qt.ColorScheme.Light


def tokens() -> dict:
    return DARK if is_dark() else LIGHT


def accent_color() -> QColor:
    """The user's system accent color (System Settings → Appearance)."""
    color = QGuiApplication.palette().color(QPalette.ColorRole.Accent)
    if not color.isValid() or color.alpha() == 0:
        color = QColor("#0a84ff")
    return color


def _rgba(color: QColor, alpha: float) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


def build_stylesheet(t: dict, accent: QColor) -> str:
    a = accent.name()
    a_hover = accent.lighter(112).name()
    a_pressed = accent.darker(112).name()
    a_soft = _rgba(accent, t["slot_checked_alpha"])
    arrows = t["arrow_suffix"]
    return f"""
QWidget {{
    background-color: transparent;
    color: {t["text"]};
    font-size: 12px;
}}

QMainWindow, QDialog, QMessageBox {{
    background-color: {t["window"]};
}}

QToolTip {{
    background-color: {t["popup"]};
    color: {t["text"]};
    border: 1px solid {t["popup_border"]};
    border-radius: 6px;
    padding: 5px 7px;
}}

QMenu {{
    background-color: {t["popup"]};
    color: {t["text"]};
    border: 1px solid {t["popup_border"]};
    border-radius: 8px;
    padding: 5px;
}}

QMenu::item {{
    padding: 4px 20px 4px 12px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {a};
    color: #ffffff;
}}

QMenu::separator {{
    height: 1px;
    background-color: {t["hairline"]};
    margin: 4px 8px;
}}

QLabel {{
    background-color: transparent;
    color: {t["text"]};
}}

QLabel[role="dim"] {{
    color: {t["secondary"]};
    font-size: 11px;
}}

QLabel[role="hint"] {{
    color: {t["secondary"]};
    font-size: 11px;
}}

QLabel[role="hint-active"] {{
    color: {a};
    font-size: 11px;
    font-weight: 600;
}}

QLabel[role="panel-title"], QLabel[role="slot-title"], QLabel[role="dialog-title"] {{
    color: {t["strong"]};
    font-weight: 600;
    font-size: 13px;
}}

QLabel[role="dialog-status"] {{
    color: {t["secondary"]};
}}

QLabel[role="card-title"] {{
    color: {a};
    font-weight: 600;
}}

QLabel[role="section-caps"] {{
    color: {t["secondary"]};
    font-size: 10px;
    font-weight: 700;
}}

QLabel[role="info-icon"] {{
    color: {a};
    font-size: 14px;
    font-weight: bold;
    padding: 0 4px;
}}

QLabel[role="panel-heading"], QLabel[role="viewport-label"] {{
    color: {t["secondary"]};
    background-color: transparent;
    border: none;
    padding: 10px 14px 4px 14px;
    font-size: 11px;
    font-weight: 700;
}}

QLabel[role="viewport-label"] {{
    padding: 0px 14px;
}}

QGroupBox {{
    background-color: {t["group"]};
    border: 1px solid {t["hairline"]};
    border-radius: 10px;
    margin-top: 22px;
    padding: 10px 10px 8px 10px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 4px;
    top: 2px;
    padding: 0 2px;
    color: {t["secondary"]};
    font-weight: 600;
    font-size: 12px;
}}

QFrame#PainterCard, QFrame#HeightWorkflowCard {{
    background-color: {t["group"]};
    border: 1px solid {t["hairline"]};
    border-radius: 10px;
    padding: 6px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QPushButton {{
    background-color: {t["control"]};
    color: {t["text"]};
    border: 1px solid {t["control_border"]};
    border-top-color: {t["control_border_top"]};
    border-radius: 6px;
    padding: 4px 12px;
    min-height: 18px;
}}

QPushButton:hover {{
    background-color: {t["control_hover"]};
}}

QPushButton:pressed {{
    background-color: {t["control_pressed"]};
}}

QPushButton:checked {{
    background-color: {a};
    border-color: {a};
    color: #ffffff;
    font-weight: 600;
}}

QPushButton:disabled {{
    color: {t["tertiary"]};
    background-color: {t["control_disabled"]};
    border-color: {t["control_disabled"]};
}}

QPushButton:focus {{
    outline: none;
}}

QDialogButtonBox QPushButton:default {{
    background-color: {a};
    border-color: {a};
    color: #ffffff;
}}

QPushButton[role="brush-preset"] {{
    text-align: left;
    padding: 6px 10px;
}}

QPushButton[role="generate"], QPushButton[role="generate-all"] {{
    background-color: {a};
    color: #ffffff;
    font-weight: 600;
    border: 1px solid {a};
    padding: 6px 12px;
    min-height: 20px;
    border-radius: 7px;
}}

QPushButton[role="generate"]:hover {{
    background-color: {a_hover};
    border-color: {a_hover};
}}

QPushButton[role="generate"]:pressed {{
    background-color: {a_pressed};
    border-color: {a_pressed};
}}

QPushButton[role="generate-all"] {{
    background-color: {GREEN};
    border-color: {GREEN};
}}

QPushButton[role="generate-all"]:hover {{
    background-color: #3bb87c;
    border-color: #3bb87c;
}}

QPushButton[role="generate-all"]:pressed {{
    background-color: #268a5a;
    border-color: #268a5a;
}}

QPushButton[role="generate"]:disabled, QPushButton[role="generate-all"]:disabled {{
    background-color: {t["control_disabled"]};
    border-color: {t["control_disabled"]};
    color: {t["tertiary"]};
}}

QPushButton[role="slot-btn"] {{
    background-color: {t["group"]};
    color: {t["secondary"]};
    border: 1px solid {t["hairline"]};
    border-radius: 8px;
    padding: 6px 2px;
    font-size: 11px;
    font-weight: 600;
    text-align: center;
    min-height: 40px;
}}

QPushButton[role="slot-btn"]:hover {{
    background-color: {t["hover_overlay"]};
    color: {t["strong"]};
}}

QPushButton[role="slot-btn"]:checked {{
    background-color: {a_soft};
    border: 1px solid {a};
    color: {t["strong"]};
}}

QWidget#SubtabBar {{
    background-color: {t["segment_track"]};
    border: 1px solid {t["hairline"]};
    border-radius: 8px;
}}

QPushButton[role="subtab"] {{
    background-color: transparent;
    color: {t["secondary"]};
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 14px;
    min-height: 18px;
    font-weight: 500;
}}

QPushButton[role="subtab"]:hover {{
    color: {t["strong"]};
}}

QPushButton[role="subtab"]:checked {{
    background-color: {t["segment_selected"]};
    color: {t["strong"]};
    border: 1px solid {t["control_border"]};
    border-top-color: {t["control_border_top"]};
    font-weight: 600;
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {t["field"]};
    border: 1px solid {t["field_border"]};
    border-radius: 6px;
    padding: 3px 8px;
    color: {t["strong"]};
    selection-background-color: {a};
    selection-color: #ffffff;
    min-height: 18px;
}}

QComboBox {{
    background-color: {t["control"]};
    border: 1px solid {t["control_border"]};
    border-top-color: {t["control_border_top"]};
    padding: 3px 26px 3px 9px;
    color: {t["text"]};
    combobox-popup: 0;
}}

QComboBox:hover {{
    background-color: {t["control_hover"]};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 18px;
    margin-right: 3px;
    border: none;
    border-radius: 4px;
    background-color: {a};
}}

QComboBox::down-arrow {{
    image: url("{_ASSETS}/chevron-updown.svg");
    width: 8px;
    height: 12px;
}}

QComboBox::drop-down:disabled {{
    background-color: transparent;
}}

QComboBox::down-arrow:disabled {{
    image: url("{_ASSETS}/chevron-updown-disabled.svg");
}}

QComboBox QAbstractItemView {{
    background-color: {t["popup"]};
    border: 1px solid {t["popup_border"]};
    border-radius: 6px;
    padding: 4px;
    outline: none;
    color: {t["text"]};
    selection-background-color: {a};
    selection-color: #ffffff;
}}

QComboBox QAbstractItemView::item {{
    min-height: 22px;
    padding: 0 6px;
    border-radius: 4px;
}}

QSpinBox, QDoubleSpinBox {{
    padding-right: 20px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border: none;
    border-left: 1px solid {t["field_border"]};
    border-top-right-radius: 6px;
    background-color: {t["spin_button"]};
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border: none;
    border-left: 1px solid {t["field_border"]};
    border-bottom-right-radius: 6px;
    background-color: {t["spin_button"]};
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {t["control_hover"]};
}}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
    background-color: {t["control_pressed"]};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{_ASSETS}/chevron-up{arrows}.svg");
    width: 9px;
    height: 6px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{_ASSETS}/chevron-down{arrows}.svg");
    width: 9px;
    height: 6px;
}}

QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: none;
}}

QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border: 1px solid {a};
}}

QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{
    background-color: {t["field_disabled"]};
    color: {t["tertiary"]};
    border-color: {t["field_disabled_border"]};
}}

QCheckBox {{
    background-color: transparent;
    color: {t["text"]};
    spacing: 7px;
}}

QCheckBox:disabled {{
    color: {t["tertiary"]};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    background-color: {t["check"]};
    border: 1px solid {t["check_border"]};
    border-radius: 4px;
}}

QCheckBox::indicator:checked {{
    background-color: {a};
    border-color: {a};
    image: url("{_ASSETS}/check.svg");
}}

QCheckBox::indicator:disabled {{
    background-color: {t["check_disabled"]};
    border-color: {t["field_disabled_border"]};
}}

QCheckBox::indicator:checked:disabled {{
    image: url("{_ASSETS}/check-disabled.svg");
}}

QProgressBar {{
    border: none;
    border-radius: 3px;
    text-align: center;
    background-color: {t["progress_track"]};
    max-height: 6px;
    font-size: 1px;
    color: transparent;
}}

QProgressBar::chunk {{
    background-color: {a};
    border-radius: 3px;
}}

QSplitter::handle {{
    background-color: {t["separator"]};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

QScrollBar:vertical {{
    width: 10px;
    background: transparent;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    min-height: 28px;
    background: {t["scroll"]};
    border-radius: 3px;
}}

QScrollBar::handle:vertical:hover {{
    background: {t["scroll_hover"]};
}}

QScrollBar:horizontal {{
    height: 10px;
    background: transparent;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    min-width: 28px;
    background: {t["scroll"]};
    border-radius: 3px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {t["scroll_hover"]};
}}

QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    width: 0px;
    height: 0px;
    background: transparent;
    border: none;
}}

ImageLabel {{
    background-color: {t["well"]};
    color: {t["well_text"]};
    border: 1px solid {t["well_border"]};
    border-radius: 8px;
}}

QWidget#ToolbarPanel {{
    background-color: {t["toolbar"]};
    border-bottom: 1px solid {t["separator"]};
}}

QWidget#StatusBarPanel {{
    background-color: {t["sidebar"]};
    border-top: 1px solid {t["separator"]};
}}

#ProjectSection {{
    background-color: {t["sidebar"]};
}}

#PreviewSection {{
    background-color: {t["window"]};
}}

#DynamicPanelHost, #PreviewPropertiesPanel, #MapInspector {{
    background-color: transparent;
    border: none;
}}

#ViewportFrame {{
    background-color: {t["canvas"]};
    border: none;
}}
"""


def _fit_popup_to_contents(combo: QComboBox) -> None:
    """Let the drop-down list be wider than its (possibly narrow) combo box so
    long entries are never cut off."""
    view = combo.view()
    fm = view.fontMetrics()
    widest = max((fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
    icon_w = combo.iconSize().width() + 6 if any(not combo.itemIcon(i).isNull() for i in range(combo.count())) else 0
    scrollbar = view.verticalScrollBar().sizeHint().width() if combo.count() > combo.maxVisibleItems() else 0
    # Item padding (QSS) + popup frame/padding + a little breathing room.
    view.setMinimumWidth(max(combo.width(), widest + icon_w + scrollbar + 40))


class _AppEventFilter(QObject):
    """App-wide hooks: widen combo popups as they open."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress or event.type() == QEvent.Type.KeyPress:
            if isinstance(obj, QComboBox):
                _fit_popup_to_contents(obj)
        return False


_installed = False
_event_filter: _AppEventFilter | None = None


def _apply_current(app: QApplication) -> None:
    app.setStyleSheet(build_stylesheet(tokens(), accent_color()))


def apply(app: QApplication) -> None:
    global _installed, _event_filter
    app.setStyle("Fusion")

    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPointSize(12)
    app.setFont(font)

    _apply_current(app)

    if not _installed:
        _installed = True
        app.styleHints().colorSchemeChanged.connect(lambda _scheme: _apply_current(app))
        _event_filter = _AppEventFilter(app)
        app.installEventFilter(_event_filter)
