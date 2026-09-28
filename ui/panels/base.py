"""Base class for the dynamic, per-map-type property panels.

Lays parameters out in a 3-column grid (wrapping to a new row every third
item), wrapped in a scroll area so the panel stays usable at small window
sizes. `add_wide()` breaks out of the 3-column grid for a full-width row
(section groups, notes, buttons).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
        self._grid.setContentsMargins(10, 10, 10, 10)
        for c in range(COLUMNS):
            self._grid.setColumnStretch(c, 1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def add_param(self, label_text: str, widget: QWidget) -> QWidget:
        """Place `widget` (with a label above it) in the next 3-column cell."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        label = QLabel(label_text)
        label.setProperty("role", "dim")
        v.addWidget(label)
        v.addWidget(widget)
        self._grid.addWidget(container, self._row, self._col)
        self._advance()
        return container

    def add_section(self, title: str) -> None:
        """Bold section-header label spanning the full width, starting a fresh row."""
        label = QLabel(title)
        label.setProperty("role", "panel-title")
        self.add_wide(label)

    def add_wide(self, widget: QWidget) -> None:
        """Place `widget` spanning the full row width, starting a fresh row."""
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
    """Splits a map type's parameters into two independent pages:

    - Generate: parameters that require a real (possibly slow) generation
      process — e.g. depth model / chunk settings. Bottom-bar "Generate
      Map" / "Kill Process" apply here.
    - Properties: parameters that only ever need an instant, local
      recompute (no model) — post-processing on already-produced data.
      Bottom-bar becomes "Save" / "Revert" here. Edits still preview live
      (concrete panels emit `propertiesChanged`, debounced upstream into a
      draft recompute), but only Save commits the draft as the map's
      official/exported result — Revert discards it back to that.

    Each page is its own `PropertyPanel` (3-column grid); concrete panels
    build into `self.generate_page` / `self.properties_page` instead of
    adding params to `self` directly.
    """

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
        self.generate_tab_btn = QPushButton("Generate")
        self.properties_tab_btn = QPushButton("Adjustment")
        for btn in (self.generate_tab_btn, self.properties_tab_btn):
            btn.setCheckable(True)
            btn.setProperty("role", "subtab")
        self.generate_tab_btn.setChecked(True)
        self.generate_tab_btn.clicked.connect(lambda: self.select_page("generate"))
        self.properties_tab_btn.clicked.connect(lambda: self.select_page("properties"))
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
