"""Qt adapter for the headless core data model.

DataModel adds Qt signals on top of tsdt_core.CoreDataModel: the core's
_notify_* observer hooks are overridden to emit the signals the rest of
the UI is wired to. All data logic lives in tsdt_core.core.

AnnotationSegment and OperationRecord are re-exported for compatibility
with existing imports.
"""
from __future__ import annotations

from PySide6 import QtCore

from tsdt_core.core import CoreDataModel
from tsdt_core.models import AnnotationSegment, OperationRecord

__all__ = ["AnnotationSegment", "DataModel", "OperationRecord"]


class DataModel(QtCore.QObject, CoreDataModel):
    """CoreDataModel with Qt signal notification."""

    dataChanged = QtCore.Signal()
    annotationsChanged = QtCore.Signal()
    statusMessage = QtCore.Signal(str)
    historyChanged = QtCore.Signal()

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        QtCore.QObject.__init__(self, parent)
        CoreDataModel.__init__(self)

    def _notify_data_changed(self) -> None:
        self.dataChanged.emit()

    def _notify_annotations_changed(self) -> None:
        self.annotationsChanged.emit()

    def _notify_history_changed(self) -> None:
        self.historyChanged.emit()

    def _notify_status(self, message: str) -> None:
        self.statusMessage.emit(message)
