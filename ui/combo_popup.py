"""Let combo-box drop-down lists grow wider than their (possibly narrow)
combo box, so long entries are never cut off. Installed app-wide by the
macOS and Windows themes."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QApplication, QComboBox


def fit_popup_to_contents(combo: QComboBox) -> None:
    view = combo.view()
    fm = view.fontMetrics()
    widest = max((fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())), default=0)
    has_icons = any(not combo.itemIcon(i).isNull() for i in range(combo.count()))
    icon_w = combo.iconSize().width() + 6 if has_icons else 0
    scrollbar = view.verticalScrollBar().sizeHint().width() if combo.count() > combo.maxVisibleItems() else 0
    # Item padding (QSS) + popup frame/padding + a little breathing room.
    view.setMinimumWidth(max(combo.width(), widest + icon_w + scrollbar + 40))


class _PopupWidthFilter(QObject):
    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress) and isinstance(obj, QComboBox):
            fit_popup_to_contents(obj)
        return False


_filter: _PopupWidthFilter | None = None


def install(app: QApplication) -> None:
    global _filter
    if _filter is None:
        _filter = _PopupWidthFilter(app)
        app.installEventFilter(_filter)
