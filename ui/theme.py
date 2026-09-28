"""Photoshop-inspired dark theme constants and stylesheet for PBRCELAIN."""
from __future__ import annotations

COLOR_BG = "#282828"
COLOR_CANVAS = "#181818"
COLOR_PANEL = "#323232"
COLOR_PANEL_HEADER = "#282828"
COLOR_PANEL_INSPECTOR = "#2a2a2a"
COLOR_INPUT_BG = "#1a1a1a"
COLOR_BORDER = "#202020"
COLOR_BORDER_LIGHT = "#3c3c3c"
COLOR_BORDER_FOCUS = "#1473e6"
COLOR_TEXT = "#dedede"
COLOR_TEXT_DIM = "#8a8a8a"
COLOR_TEXT_BRIGHT = "#ffffff"
COLOR_BTN_BG = "#3e3e3e"
COLOR_BTN_HOVER = "#4c4c4c"
COLOR_BTN_PRESSED = "#282828"
COLOR_ACCENT = "#1473e6"
COLOR_ACCENT_HOVER = "#2688f2"
COLOR_ACCENT_PRESSED = "#0d66d0"
COLOR_DISABLED = "#555555"
COLOR_DISABLED_BG = "#2c2c2c"

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 11px;
}}

QToolBar {{
    background-color: {COLOR_PANEL};
    border-bottom: 1px solid {COLOR_BORDER};
    spacing: 2px;
    padding: 3px 6px;
    min-height: 32px;
}}

QToolButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    padding: 4px 8px;
    border-radius: 2px;
    border: 1px solid transparent;
    font-size: 11px;
}}

QToolButton:hover {{
    background-color: {COLOR_BTN_HOVER};
    border-color: #555555;
    color: {COLOR_TEXT_BRIGHT};
}}

QToolButton:pressed {{
    background-color: {COLOR_BTN_PRESSED};
    border-color: {COLOR_BORDER};
}}

QMenuBar {{
    background-color: {COLOR_PANEL};
    color: {COLOR_TEXT};
    border-bottom: 1px solid {COLOR_BORDER};
    font-size: 11px;
}}

QMenuBar::item {{
    padding: 4px 8px;
    background: transparent;
}}

QMenuBar::item:selected {{
    background-color: {COLOR_BTN_HOVER};
    color: {COLOR_TEXT_BRIGHT};
}}

QMenu {{
    background-color: #2c2c2c;
    color: {COLOR_TEXT};
    border: 1px solid #1c1c1c;
    padding: 4px 0px;
    font-size: 11px;
}}

QMenu::item {{
    padding: 5px 24px 5px 20px;
}}

QMenu::item:selected {{
    background-color: {COLOR_ACCENT};
    color: {COLOR_TEXT_BRIGHT};
}}

QMenu::separator {{
    height: 1px;
    background-color: {COLOR_BORDER_LIGHT};
    margin: 4px 8px;
}}

QStatusBar {{
    background-color: {COLOR_PANEL_HEADER};
    color: {COLOR_TEXT_DIM};
    border-top: 1px solid {COLOR_BORDER};
    font-size: 11px;
}}

QLabel {{
    background-color: transparent;
    color: {COLOR_TEXT};
}}

QLabel[role="dim"] {{
    background-color: transparent;
    color: {COLOR_TEXT_DIM};
}}

QLabel[role="panel-title"] {{
    background-color: transparent;
    color: {COLOR_TEXT_BRIGHT};
    font-weight: 600;
    font-size: 11px;
}}

QLabel[role="panel-heading"] {{
    color: #b0b0b0;
    background-color: {COLOR_PANEL_HEADER};
    border-bottom: 1px solid #1c1c1c;
    padding: 6px 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

QLabel[role="viewport-label"] {{
    color: #b0b0b0;
    background-color: #242424;
    border-bottom: 1px solid #1a1a1a;
    padding: 6px 12px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

QGroupBox {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    background-color: #2d2d2d;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: {COLOR_TEXT_BRIGHT};
    font-weight: 600;
    font-size: 11px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QPushButton {{
    background-color: {COLOR_BTN_BG};
    color: {COLOR_TEXT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
    padding: 5px 12px;
    font-size: 11px;
}}

QPushButton:hover {{
    background-color: {COLOR_BTN_HOVER};
    border-color: #555555;
    color: {COLOR_TEXT_BRIGHT};
}}

QPushButton:pressed {{
    background-color: {COLOR_BTN_PRESSED};
    border-color: {COLOR_BORDER};
}}

QPushButton:disabled {{
    color: {COLOR_DISABLED};
    background-color: {COLOR_DISABLED_BG};
    border-color: #222222;
}}

QPushButton[role="generate"] {{
    background-color: {COLOR_ACCENT};
    color: {COLOR_TEXT_BRIGHT};
    font-weight: 600;
    border: 1px solid #0d66d0;
    padding: 7px 12px;
    border-radius: 3px;
    font-size: 11px;
}}

QPushButton[role="generate"]:hover {{
    background-color: {COLOR_ACCENT_HOVER};
    border-color: {COLOR_ACCENT};
}}

QPushButton[role="generate"]:pressed {{
    background-color: {COLOR_ACCENT_PRESSED};
    border-color: #0952a5;
}}

QPushButton[role="generate"]:disabled {{
    background-color: #26384e;
    color: #6d7f95;
    border-color: #1a2535;
}}

QPushButton[role="generate-all"] {{
    background-color: #12805c;
    color: {COLOR_TEXT_BRIGHT};
    font-weight: 600;
    border: 1px solid #0e6649;
    padding: 7px 12px;
    border-radius: 3px;
    font-size: 11px;
}}

QPushButton[role="generate-all"]:hover {{
    background-color: #1aa87a;
    border-color: #12805c;
}}

QPushButton[role="generate-all"]:pressed {{
    background-color: #0d5e43;
    border-color: #09402d;
}}

QPushButton[role="generate-all"]:disabled {{
    background-color: #1f3a2f;
    color: #5d7a6e;
    border-color: #15261f;
}}

QPushButton[role="slot-btn"] {{
    background-color: #282828;
    color: #a0a0a0;
    border: 1px solid #1e1e1e;
    border-radius: 3px;
    padding: 6px 4px;
    font-size: 10px;
    font-weight: 600;
    text-align: center;
    min-height: 42px;
}}

QPushButton[role="slot-btn"]:hover {{
    background-color: #383838;
    border-color: #444444;
    color: {COLOR_TEXT_BRIGHT};
}}

QPushButton[role="slot-btn"]:checked {{
    background-color: #383838;
    border-top: 3px solid {COLOR_ACCENT};
    border-bottom: 1px solid #1e1e1e;
    border-left: 1px solid #1e1e1e;
    border-right: 1px solid #1e1e1e;
    color: {COLOR_TEXT_BRIGHT};
    font-weight: 700;
}}

QPushButton[role="subtab"] {{
    background-color: transparent;
    color: {COLOR_TEXT_DIM};
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0px;
    padding: 6px 14px;
    font-weight: 600;
    font-size: 11px;
}}

QPushButton[role="subtab"]:hover {{
    color: #d0d0d0;
    background-color: rgba(255, 255, 255, 0.03);
}}

QPushButton[role="subtab"]:checked {{
    color: {COLOR_TEXT_BRIGHT};
    border-bottom: 2px solid {COLOR_ACCENT};
    background-color: rgba(20, 115, 230, 0.06);
}}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {COLOR_INPUT_BG};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 2px;
    padding: 3px 6px;
    color: {COLOR_TEXT_BRIGHT};
    font-size: 11px;
    selection-background-color: {COLOR_ACCENT};
    selection-color: {COLOR_TEXT_BRIGHT};
    min-height: 20px;
}}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
    border-color: #555555;
}}

QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border: 1px solid {COLOR_BORDER_FOCUS};
    background-color: #161616;
}}

QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{
    background-color: #262626;
    color: {COLOR_DISABLED};
    border-color: #2e2e2e;
}}

QComboBox QAbstractItemView {{
    background-color: #262626;
    border: 1px solid #1c1c1c;
    selection-background-color: {COLOR_ACCENT};
    selection-color: {COLOR_TEXT_BRIGHT};
    color: {COLOR_TEXT};
    padding: 2px;
}}

QCheckBox, QRadioButton {{
    background-color: transparent;
    color: {COLOR_TEXT};
    spacing: 6px;
    font-size: 11px;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width: 13px;
    height: 13px;
    background-color: {COLOR_INPUT_BG};
    border: 1px solid #484848;
    border-radius: 2px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {COLOR_ACCENT};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT};
}}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {{
    background-color: #262626;
    border-color: #383838;
}}

QSlider::groove:horizontal {{
    height: 3px;
    background: #161616;
    border: 1px solid #282828;
    border-radius: 1px;
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_ACCENT};
    border-radius: 1px;
}}

QSlider::handle:horizontal {{
    background: #888888;
    border: 1px solid #202020;
    width: 10px;
    height: 14px;
    margin: -6px 0;
    border-radius: 2px;
}}

QSlider::handle:horizontal:hover {{
    background: #cccccc;
    border-color: {COLOR_ACCENT};
}}

QSlider::handle:horizontal:pressed {{
    background: {COLOR_ACCENT};
    border-color: {COLOR_ACCENT_PRESSED};
}}

QProgressBar {{
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 2px;
    text-align: center;
    background-color: {COLOR_INPUT_BG};
    height: 10px;
    font-size: 9px;
    color: {COLOR_TEXT_BRIGHT};
}}

QProgressBar::chunk {{
    background-color: {COLOR_ACCENT};
    border-radius: 1px;
}}

QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    background-color: {COLOR_PANEL};
}}

QTabBar::tab {{
    background: {COLOR_PANEL_HEADER};
    color: {COLOR_TEXT_DIM};
    border: 1px solid #1c1c1c;
    border-bottom: none;
    padding: 6px 14px;
    font-weight: 600;
    font-size: 11px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background: {COLOR_PANEL};
    color: {COLOR_TEXT_BRIGHT};
    border-top: 2px solid {COLOR_ACCENT};
}}

QTabBar::tab:hover:!selected {{
    background: #2f2f2f;
    color: {COLOR_TEXT};
}}

QSplitter::handle:horizontal {{
    width: 3px;
    background-color: #1e1e1e;
}}

QSplitter::handle:vertical {{
    height: 3px;
    background-color: #1e1e1e;
}}

QSplitter::handle:hover {{
    background-color: {COLOR_ACCENT};
}}

QScrollBar:vertical {{
    width: 8px;
    background: transparent;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    min-height: 24px;
    background: #4a4a4a;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #606060;
}}

QScrollBar::handle:vertical:pressed {{
    background: #787878;
}}

QScrollBar:horizontal {{
    height: 8px;
    background: transparent;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    min-width: 24px;
    background: #4a4a4a;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: #606060;
}}

QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{
    width: 0px;
    height: 0px;
    background: transparent;
    border: none;
}}

#ProjectSection {{
    background-color: {COLOR_PANEL};
    border-right: 1px solid #1c1c1c;
}}

#PreviewSection {{
    background-color: {COLOR_BG};
}}

#DynamicPanelHost, #PreviewPropertiesPanel {{
    background-color: {COLOR_PANEL};
    border: none;
}}

#MapInspector {{
    background-color: {COLOR_PANEL_INSPECTOR};
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
}}

#ViewportFrame {{
    background-color: {COLOR_CANVAS};
    border: 1px solid #141414;
}}
"""
