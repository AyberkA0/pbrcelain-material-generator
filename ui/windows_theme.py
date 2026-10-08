from __future__ import annotations

import os

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QIcon, QPalette, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

from ui import combo_popup, win_window
from ui.theme import theme_changed

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "windows").replace("\\", "/")

DARK = {
    "scheme": "dark",
    "window": "#202020",
    "layer": "#272727",
    "card": "#2b2b2b",
    "card_border": "#1d1d1d",
    "divider": "#1a1a1a",
    "text": "#ffffff",
    "secondary": "#c5c5c5",
    "tertiary": "#9a9a9a",
    "disabled": "#5d5d5d",
    "control": "#2d2d2d",
    "control_hover": "#323232",
    "control_pressed": "#272727",
    "control_border": "#353535",
    "control_border_bottom": "#2a2a2a",
    "control_disabled": "#262626",
    "field": "#2d2d2d",
    "field_focus": "#1f1f1f",
    "field_underline": "#9a9a9a",
    "popup": "#2c2c2c",
    "popup_border": "#3a3a3a",
    "subtle_hover": "rgba(255, 255, 255, 0.06)",
    "subtle_pressed": "rgba(255, 255, 255, 0.04)",
    "check": "#1e1e1e",
    "check_border": "#9a9a9a",
    "scroll": "#8a8a8a",
    "progress_track": "#3a3a3a",
    "well": "#1c1c1c",
    "canvas": "#141414",
    "success": "#6ccb5f",
    "on_success": "#000000",
}

LIGHT = {
    "scheme": "light",
    "window": "#f3f3f3",
    "layer": "#f9f9f9",
    "card": "#fbfbfb",
    "card_border": "#e5e5e5",
    "divider": "#e0e0e0",
    "text": "#1b1b1b",
    "secondary": "#5d5d5d",
    "tertiary": "#8b8b8b",
    "disabled": "#a0a0a0",
    "control": "#fbfbfb",
    "control_hover": "#f6f6f6",
    "control_pressed": "#f0f0f0",
    "control_border": "#e5e5e5",
    "control_border_bottom": "#cccccc",
    "control_disabled": "#f5f5f5",
    "field": "#fbfbfb",
    "field_focus": "#ffffff",
    "field_underline": "#868686",
    "popup": "#f9f9f9",
    "popup_border": "#dcdcdc",
    "subtle_hover": "rgba(0, 0, 0, 0.04)",
    "subtle_pressed": "rgba(0, 0, 0, 0.02)",
    "check": "#f5f5f5",
    "check_border": "#868686",
    "scroll": "#8a8a8a",
    "progress_track": "#d6d6d6",
    "well": "#e9e9e9",
    "canvas": "#141414",
    "success": "#0f7b0f",
    "on_success": "#ffffff",
}


def is_dark() -> bool:
    app = QGuiApplication.instance()
    if app is None:
        return True
    return app.styleHints().colorScheme() != Qt.ColorScheme.Light


def tokens() -> dict:
    return DARK if is_dark() else LIGHT


def _system_accent() -> QColor:
    color = QGuiApplication.palette().color(QPalette.ColorRole.Accent)
    if not color.isValid() or color.alpha() == 0:
        color = QColor("#0078d4")
    return color


def accent_colors(dark: bool) -> dict:
    base = _system_accent()
    h, s, _l, _a = base.getHslF()
    h = max(h, 0.0)
    if dark:
        fill = QColor.fromHslF(h, s, 0.70)
        return {
            "accent": fill.name(),
            "accent_hover": QColor.fromHslF(h, s, 0.65).name(),
            "accent_pressed": QColor.fromHslF(h, s, 0.60).name(),
            "on_accent": "#000000",
            "accent_text": fill.name(),
        }
    fill = QColor.fromHslF(h, s, 0.36)
    return {
        "accent": fill.name(),
        "accent_hover": QColor.fromHslF(h, s, 0.42).name(),
        "accent_pressed": QColor.fromHslF(h, s, 0.48).name(),
        "on_accent": "#ffffff",
        "accent_text": fill.name(),
    }


def build_stylesheet(t: dict, a: dict) -> str:
    scheme = t["scheme"]
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
    border-radius: 4px;
    padding: 6px 8px;
}}


QMenu {{
    background-color: {t["popup"]};
    color: {t["text"]};
    border: 1px solid {t["popup_border"]};
    border-radius: 8px;
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 28px 6px 12px;
    border-radius: 4px;
}}

QMenu::item:selected {{
    background-color: {t["subtle_hover"]};
}}

QMenu::item:disabled {{
    color: {t["disabled"]};
}}

QMenu::separator {{
    height: 1px;
    background-color: {t["card_border"]};
    margin: 4px 0px;
}}

QMenu::right-arrow {{
    image: url("{_ASSETS}/chevron-down-{scheme}.svg");
    width: 10px;
    height: 10px;
}}

QToolBar {{
    background-color: {t["window"]};
    border: none;
    border-bottom: 1px solid {t["divider"]};
    spacing: 2px;
    padding: 4px 8px;
}}

QToolBar::separator {{
    width: 1px;
    background-color: {t["card_border"]};
    margin: 6px 6px;
}}

QToolButton {{
    background-color: transparent;
    color: {t["text"]};
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 8px;
}}

QToolButton:hover {{
    background-color: {t["subtle_hover"]};
}}

QToolButton[popupMode="1"] {{
    padding-right: 22px;
}}

QToolButton::menu-button {{
    border: none;
    border-left: 1px solid transparent;
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
    width: 18px;
}}

QToolButton::menu-button:hover {{
    border-left-color: {t["card_border"]};
    background-color: {t["subtle_hover"]};
}}

QToolButton::menu-arrow {{
    image: url("{_ASSETS}/chevron-down-{scheme}.svg");
    width: 10px;
    height: 10px;
}}

QToolButton:pressed {{
    background-color: {t["subtle_pressed"]};
    color: {t["secondary"]};
}}


QLabel {{
    background-color: transparent;
    color: {t["text"]};
}}

QLabel[role="dim"], QLabel[role="hint"] {{
    color: {t["secondary"]};
    font-size: 11px;
}}

QLabel[role="hint-active"], QLabel[role="card-title"] {{
    color: {a["accent_text"]};
    font-weight: 600;
}}

QLabel[role="info-icon"] {{
    color: {a["accent_text"]};
    font-size: 14px;
    padding: 0 4px;
}}

QLabel[role="panel-title"], QLabel[role="slot-title"], QLabel[role="dialog-title"] {{
    color: {t["text"]};
    font-size: 14px;
    font-weight: 600;
}}

QLabel[role="dialog-status"] {{
    color: {t["secondary"]};
}}

QLabel[role="section-caps"] {{
    color: {t["secondary"]};
    font-size: 11px;
    font-weight: 600;
}}

QLabel[role="panel-heading"], QLabel[role="viewport-label"] {{
    color: {t["text"]};
    background-color: transparent;
    border: none;
    padding: 12px 16px 6px 16px;
    font-size: 13px;
    font-weight: 600;
}}

QLabel[role="viewport-label"] {{
    padding: 8px 16px;
}}


QGroupBox {{
    background-color: {t["card"]};
    border: 1px solid {t["card_border"]};
    border-radius: 8px;
    margin-top: 24px;
    padding: 12px 12px 10px 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 2px;
    top: 2px;
    padding: 0px;
    color: {t["text"]};
    font-weight: 600;
    font-size: 13px;
}}

QFrame#PainterCard, QFrame#HeightWorkflowCard {{
    background-color: {t["card"]};
    border: 1px solid {t["card_border"]};
    border-radius: 8px;
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
    border-bottom-color: {t["control_border_bottom"]};
    border-radius: 4px;
    padding: 5px 11px;
    min-height: 20px;
}}

QPushButton:hover {{
    background-color: {t["control_hover"]};
}}

QPushButton:pressed {{
    background-color: {t["control_pressed"]};
    color: {t["secondary"]};
    border-bottom-color: {t["control_border"]};
}}

QPushButton:disabled {{
    background-color: {t["control_disabled"]};
    color: {t["disabled"]};
    border-color: {t["control_border"]};
}}

QPushButton:focus {{
    outline: none;
}}

QPushButton:checked, QDialogButtonBox QPushButton:default,
QPushButton[role="generate"] {{
    background-color: {a["accent"]};
    color: {a["on_accent"]};
    border: 1px solid {a["accent"]};
    border-bottom-color: {a["accent_pressed"]};
}}

QPushButton:checked:hover, QDialogButtonBox QPushButton:default:hover,
QPushButton[role="generate"]:hover {{
    background-color: {a["accent_hover"]};
    border-color: {a["accent_hover"]};
}}

QPushButton:checked:pressed, QDialogButtonBox QPushButton:default:pressed,
QPushButton[role="generate"]:pressed {{
    background-color: {a["accent_pressed"]};
    border-color: {a["accent_pressed"]};
}}

QPushButton[role="generate"] {{
    font-weight: 600;
    padding: 7px 12px;
}}

QPushButton[role="generate"]:disabled {{
    background-color: {t["control_disabled"]};
    color: {t["disabled"]};
    border-color: {t["control_border"]};
}}

QPushButton[role="generate-all"] {{
    background-color: {t["success"]};
    color: {t["on_success"]};
    border: 1px solid {t["success"]};
    font-weight: 600;
    padding: 7px 12px;
}}

QPushButton[role="generate-all"]:disabled {{
    background-color: {t["control_disabled"]};
    color: {t["disabled"]};
    border-color: {t["control_border"]};
}}

QPushButton[role="brush-preset"] {{
    text-align: left;
    padding: 6px 10px;
}}

QPushButton[role="slot-btn"] {{
    font-size: 11px;
    font-weight: 600;
    min-height: 40px;
    padding: 6px 2px;
}}

QPushButton[role="slot-btn"]:!checked {{
    color: {t["secondary"]};
}}

QPushButton[role="subtab"] {{
    background-color: transparent;
    color: {t["secondary"]};
    border: none;
    border-bottom: 3px solid transparent;
    border-radius: 0px;
    padding: 6px 12px 4px 12px;
    min-height: 20px;
}}

QPushButton[role="subtab"]:hover {{
    color: {t["text"]};
    background-color: {t["subtle_hover"]};
}}

QPushButton[role="subtab"]:checked {{
    color: {t["text"]};
    background-color: transparent;
    border-bottom: 3px solid {a["accent"]};
    font-weight: 600;
}}


QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {t["field"]};
    color: {t["text"]};
    border: 1px solid {t["control_border"]};
    border-bottom: 1px solid {t["field_underline"]};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 20px;
    selection-background-color: {a["accent"]};
    selection-color: {a["on_accent"]};
}}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    background-color: {t["control_hover"]};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    background-color: {t["field_focus"]};
    border-bottom: 2px solid {a["accent"]};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {t["control_disabled"]};
    color: {t["disabled"]};
    border-bottom-color: {t["control_border"]};
}}

QSpinBox, QDoubleSpinBox {{
    padding-right: 24px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    border: none;
    border-radius: 3px;
    background-color: transparent;
    margin: 2px 2px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-position: top right;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-position: bottom right;
}}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {t["subtle_hover"]};
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{_ASSETS}/chevron-up-{scheme}.svg");
    width: 10px;
    height: 10px;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{_ASSETS}/chevron-down-{scheme}.svg");
    width: 10px;
    height: 10px;
}}

QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    image: none;
}}

QComboBox {{
    background-color: {t["control"]};
    color: {t["text"]};
    border: 1px solid {t["control_border"]};
    border-bottom-color: {t["control_border_bottom"]};
    border-radius: 4px;
    padding: 4px 30px 4px 10px;
    min-height: 20px;
    combobox-popup: 0;
}}

QComboBox:hover {{
    background-color: {t["control_hover"]};
}}

QComboBox:disabled {{
    background-color: {t["control_disabled"]};
    color: {t["disabled"]};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border: none;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url("{_ASSETS}/chevron-down-{scheme}.svg");
    width: 12px;
    height: 12px;
}}

QComboBox QAbstractItemView {{
    background-color: {t["popup"]};
    color: {t["text"]};
    border: 1px solid {t["popup_border"]};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}

QComboBox QAbstractItemView::item {{
    min-height: 28px;
    padding: 0px 10px;
    border-radius: 4px;
    border-left: 3px solid transparent;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {t["subtle_hover"]};
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {t["subtle_hover"]};
    color: {t["text"]};
    border-left: 3px solid {a["accent"]};
}}

QCheckBox {{
    background-color: transparent;
    color: {t["text"]};
    spacing: 8px;
}}

QCheckBox:disabled {{
    color: {t["disabled"]};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    background-color: {t["check"]};
    border: 1px solid {t["check_border"]};
    border-radius: 4px;
}}

QCheckBox::indicator:hover {{
    background-color: {t["subtle_hover"]};
}}

QCheckBox::indicator:checked {{
    background-color: {a["accent"]};
    border-color: {a["accent"]};
    image: url("{_ASSETS}/check-{scheme}.svg");
}}

QCheckBox::indicator:disabled {{
    border-color: {t["disabled"]};
    background-color: transparent;
}}

QCheckBox::indicator:checked:disabled {{
    background-color: {t["disabled"]};
    image: url("{_ASSETS}/check-disabled.svg");
}}


QProgressBar {{
    border: none;
    border-radius: 2px;
    background-color: {t["progress_track"]};
    max-height: 4px;
    font-size: 1px;
    color: transparent;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {a["accent"]};
    border-radius: 2px;
}}

QScrollBar:vertical {{
    width: 8px;
    background: transparent;
    margin: 2px;
}}

QScrollBar::handle:vertical {{
    min-height: 32px;
    background: {t["scroll"]};
    border-radius: 2px;
}}

QScrollBar:horizontal {{
    height: 8px;
    background: transparent;
    margin: 2px;
}}

QScrollBar::handle:horizontal {{
    min-width: 32px;
    background: {t["scroll"]};
    border-radius: 2px;
}}

QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    width: 0px;
    height: 0px;
    background: transparent;
    border: none;
}}

QSplitter::handle {{
    background-color: {t["divider"]};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}


ImageLabel {{
    background-color: {t["well"]};
    color: {t["tertiary"]};
    border: 1px solid {t["card_border"]};
    border-radius: 4px;
}}

QWidget#ToolbarPanel {{
    background-color: {t["window"]};
    border-bottom: 1px solid {t["divider"]};
}}

QWidget#StatusBarPanel {{
    background-color: {t["layer"]};
    border-top: 1px solid {t["divider"]};
}}

#ProjectSection {{
    background-color: {t["window"]};
}}

#PreviewSection {{
    background-color: {t["layer"]};
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


def themed_icon(name: str) -> QIcon:
    with open(os.path.join(_ASSETS, "toolbar", f"{name}.svg"), encoding="utf-8") as f:
        svg = f.read().replace("currentColor", tokens()["text"])
    pixmap = QPixmap()
    pixmap.loadFromData(svg.encode("utf-8"), "SVG")
    return QIcon(pixmap)


def _ui_font() -> QFont:
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    if "Segoe UI Variable Text" in QFontDatabase.families():
        font.setFamily("Segoe UI Variable Text")
    font.setPointSize(9)
    return font


class _TitleBarFilter(QObject):
    def eventFilter(self, obj, event):
        if (
            event.type() == QEvent.Type.Show
            and isinstance(obj, QWidget)
            and obj.windowType() in (Qt.WindowType.Window, Qt.WindowType.Dialog)
        ):
            win_window.apply_title_bar_theme(obj, is_dark())
        return False


_installed = False
_title_bar_filter: _TitleBarFilter | None = None


def _apply_current(app: QApplication) -> None:
    app.setStyleSheet(build_stylesheet(tokens(), accent_colors(is_dark())))
    for widget in app.topLevelWidgets():
        if widget.isVisible() and widget.windowType() in (Qt.WindowType.Window, Qt.WindowType.Dialog):
            win_window.apply_title_bar_theme(widget, is_dark())
    theme_changed.emit()


def apply(app: QApplication) -> None:
    global _installed
    app.setStyle("Fusion")
    app.setFont(_ui_font())
    _apply_current(app)

    if not _installed:
        _installed = True
        app.styleHints().colorSchemeChanged.connect(lambda _scheme: _apply_current(app))
        combo_popup.install(app)
        global _title_bar_filter
        _title_bar_filter = _TitleBarFilter(app)
        app.installEventFilter(_title_bar_filter)
