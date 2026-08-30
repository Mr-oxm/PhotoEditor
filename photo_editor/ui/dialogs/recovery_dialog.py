"""Offer documents left behind by a session that did not exit cleanly."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)


class RecoveryDialog(QDialog):
    """Lists recoverable documents and lets the user restore or discard them."""

    def __init__(self, entries: list, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recover Unsaved Work")
        self.setModal(True)
        self.resize(520, 320)
        self._entries = list(entries)

        layout = QVBoxLayout(self)
        heading = QLabel(
            "Basera closed unexpectedly. These documents had unsaved changes."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)

        self._list = QListWidget()
        self._list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        for entry in self._entries:
            label = f"{entry.name}   —   autosaved {entry.describe_age()}"
            if entry.original_path:
                label += f"\n{entry.original_path}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self._list.addItem(item)
        self._list.selectAll()
        layout.addWidget(self._list, 1)

        note = QLabel(
            "Recovered documents open as unsaved copies — save them where "
            "you want them."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")
        layout.addWidget(note)

        buttons = QHBoxLayout()
        self._discard_btn = QPushButton("Discard Selected")
        self._discard_btn.clicked.connect(self._on_discard)
        buttons.addWidget(self._discard_btn)
        buttons.addStretch()

        box = QDialogButtonBox()
        self._recover_btn = box.addButton(
            "Recover Selected", QDialogButtonBox.ButtonRole.AcceptRole)
        box.addButton("Not Now", QDialogButtonBox.ButtonRole.RejectRole)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        buttons.addWidget(box)
        layout.addLayout(buttons)

    # ---- API ---------------------------------------------------------------

    def selected_entries(self) -> list:
        return [item.data(Qt.ItemDataRole.UserRole)
                for item in self._list.selectedItems()]

    def _on_discard(self) -> None:
        for item in self._list.selectedItems():
            entry = item.data(Qt.ItemDataRole.UserRole)
            entry.discard()
            self._list.takeItem(self._list.row(item))
        if self._list.count() == 0:
            self.reject()
