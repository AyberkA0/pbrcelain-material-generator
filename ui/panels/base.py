from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.theme import IS_MAC, PLATFORM_THEME, pick

COLUMNS = 3


class PropertyPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._row = 0
        self._col = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        self._grid = QGridLayout(content)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(12)
        self._grid.setContentsMargins(*pick(mac=(14, 12, 14, 14), windows=(16, 12, 16, 16), legacy=(10, 10, 10, 10)))
        for c in range(COLUMNS):
            self._grid.setColumnStretch(c, 1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def add_param(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        label = QLabel(label_text)
        label.setProperty("role", "dim")
        v.addWidget(label)
        v.addWidget(widget)
        if PLATFORM_THEME != "legacy":
            combos = widget.findChildren(QComboBox)
            if isinstance(widget, QComboBox):
                combos.append(widget)
            for combo in combos:
                combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(4)
        self._grid.addWidget(container, self._row, self._col)
        self._advance()
        return container

    def add_section(self, title: str) -> None:
        label = QLabel(title)
        label.setProperty("role", "panel-title")
        self.add_wide(label)

    def add_wide(self, widget: QWidget) -> None:
        if self._col != 0:
            self._row += 1
            self._col = 0
        self._grid.addWidget(widget, self._row, 0, 1, COLUMNS)
        self._row += 1

    def add_stretch(self) -> None:
        self._grid.setRowStretch(self._row + 1, 1)

    def _advance(self) -> None:
        self._col += 1
        if self._col >= COLUMNS:
            self._col = 0
            self._row += 1


class GeneratePropertiesContainer(QWidget):
    pageChanged = pyqtSignal(str)
    propertiesChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.generate_page = PropertyPanel()
        self.properties_page = PropertyPanel()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs_row = QHBoxLayout()
        tabs_row.setContentsMargins(6, 6, 6, 0)
        if IS_MAC:
            tabs_row.setContentsMargins(14, 4, 14, 2)
            segment = QWidget()
            segment.setObjectName("SubtabBar")
            segment.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            segment_layout = QHBoxLayout(segment)
            segment_layout.setContentsMargins(2, 2, 2, 2)
            segment_layout.setSpacing(2)
        self.generate_tab_btn = QPushButton("Generate")
        self.properties_tab_btn = QPushButton("Adjustment")
        for btn in (self.generate_tab_btn, self.properties_tab_btn):
            btn.setCheckable(True)
            btn.setProperty("role", "subtab")
        self.generate_tab_btn.setChecked(True)
        self.generate_tab_btn.clicked.connect(lambda: self.select_page("generate"))
        self.properties_tab_btn.clicked.connect(lambda: self.select_page("properties"))
        if IS_MAC:
            segment_layout.addWidget(self.generate_tab_btn)
            segment_layout.addWidget(self.properties_tab_btn)
            tabs_row.addWidget(segment, 1)
        else:
            tabs_row.addWidget(self.generate_tab_btn)
            tabs_row.addWidget(self.properties_tab_btn)
            tabs_row.addStretch(1)
        layout.addLayout(tabs_row)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.generate_page)
        self.stack.addWidget(self.properties_page)
        layout.addWidget(self.stack, 1)

    def select_page(self, name: str) -> None:
        if name == "properties":
            self.stack.setCurrentWidget(self.properties_page)
            self.properties_tab_btn.setChecked(True)
            self.generate_tab_btn.setChecked(False)
        else:
            name = "generate"
            self.stack.setCurrentWidget(self.generate_page)
            self.generate_tab_btn.setChecked(True)
            self.properties_tab_btn.setChecked(False)
        self.pageChanged.emit(name)

    def current_page(self) -> str:
        return "properties" if self.stack.currentWidget() is self.properties_page else "generate"
