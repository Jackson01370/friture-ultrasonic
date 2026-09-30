#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 puppy

# This file is part of Friture.
#
# Friture is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
# Friture is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Friture.  If not, see <http://www.gnu.org/licenses/>.

"""What RecordingEvents.qml shows: the events, newest first, and the filters.

The rows are a list MODEL updated in place, keyed by event, for the reason
learnt on the Band Survey list: a list property is rebuilt wholesale on
every change, and a row destroyed between press and release swallows the
click.
"""

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot

from friture.replay_view_model import _prop


class EventListModel(QtCore.QAbstractListModel):
    """Rows keyed by "key", in the order given (newest first)."""

    FIELDS = ("key", "start", "time_text", "kind", "kind_label", "detail", "brief", "protected")
    countChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self._roles = {QtCore.Qt.UserRole + 1 + i: f.encode() for i, f in enumerate(self.FIELDS)}
        self._field = {QtCore.Qt.UserRole + 1 + i: f for i, f in enumerate(self.FIELDS)}

    def roleNames(self):
        return self._roles

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        f = self._field.get(role)
        return self._rows[index.row()].get(f) if f else None

    @pyqtProperty(int, notify=countChanged)
    def count(self):
        return len(self._rows)

    @pyqtSlot(int, result='QVariant')
    def get(self, row):
        return dict(self._rows[row]) if 0 <= row < len(self._rows) else {}

    def set_rows(self, rows) -> None:
        wanted = list(rows)
        keys = [r["key"] for r in wanted]
        # removals, from the back
        for i in range(len(self._rows) - 1, -1, -1):
            if self._rows[i]["key"] not in keys:
                self.beginRemoveRows(QtCore.QModelIndex(), i, i)
                del self._rows[i]
                self.endRemoveRows()
        # insertions and updates, in the wanted order
        for pos, want in enumerate(wanted):
            have = next((i for i, r in enumerate(self._rows) if r["key"] == want["key"]), None)
            if have is None:
                self.beginInsertRows(QtCore.QModelIndex(), pos, pos)
                self._rows.insert(pos, dict(want))
                self.endInsertRows()
            else:
                if have != pos:
                    self.beginMoveRows(QtCore.QModelIndex(), have, have, QtCore.QModelIndex(),
                                       pos if pos < have else pos + 1)
                    self._rows.insert(pos, self._rows.pop(have))
                    self.endMoveRows()
                if self._rows[pos] != want:
                    self._rows[pos] = dict(want)
                    idx = self.index(pos, 0)
                    self.dataChanged.emit(idx, idx, list(self._roles.keys()))
        self.countChanged.emit()


class RecordingEventsViewModel(QtCore.QObject):
    summary_text_changed, summary_text = _prop("summary_text", str, "")
    protected_text_changed, protected_text = _prop("protected_text", str, "")
    status_text_changed, status_text = _prop("status_text", str, "")
    show_loud_changed, show_loud = _prop("show_loud", bool, True)
    show_voice_changed, show_voice = _prop("show_voice", bool, True)
    show_brief_changed, show_brief = _prop("show_brief", bool, True)
    show_lines_changed, show_lines = _prop("show_lines", bool, True)
    protected_only_changed, protected_only = _prop("protected_only", bool, False)

    play_requested = pyqtSignal(str)
    protect_requested = pyqtSignal(str)
    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events = EventListModel(self)

    @pyqtProperty(QtCore.QObject, constant=True)
    def events(self):
        return self._events

    @pyqtSlot(str)
    def play(self, key):
        self.play_requested.emit(key)

    @pyqtSlot(str)
    def toggle_protect(self, key):
        self.protect_requested.emit(key)

    @pyqtSlot()
    def refresh(self):
        self.refresh_requested.emit()

    # the filters are set from QML checkboxes
    @pyqtSlot(str, bool)
    def set_filter(self, name, on):
        setattr(self, name, bool(on))
