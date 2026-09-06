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

"""What BandSurvey.qml shows: the lines found, and the shape of the noise.

Display only, plus one way back: ``tune`` sends the Listen band to a line,
which is what makes the list actionable -- find something, click it, and
every other dock that follows the band is already looking at it.

★ THE LINES ARE A MODEL, NOT A LIST PROPERTY ★

    They were a QVariantList first, and clicking one usually did nothing.
    A QVariantList is replaced wholesale on every change, so QML destroys
    and rebuilds every delegate -- including the one under the cursor,
    between the press and the release. Measured on a steady two-tone room:
    the list changed on 16 of 28 refreshes, about twice a second, because
    the levels wobble by a tenth of a dB even when the lines do not move.

    A QAbstractListModel keyed by frequency updates a row IN PLACE instead:
    rows survive, only the changed cells repaint, and a click always lands
    on the row it was aimed at. Rows are inserted and removed only when a
    line really appears or disappears.
"""

from PyQt5 import QtCore


class SurveyLineModel(QtCore.QAbstractListModel):
    """The lines, kept in place across updates -- see the module docstring.

    Rows are ordered by FREQUENCY, lowest first, so a line does not jump
    around the list as its level wobbles. Which lines are in the model is
    still decided by how far each stands over its own floor; this is only
    the order they are shown in.
    """

    FrequencyRole = QtCore.Qt.UserRole + 1
    FrequencyTextRole = QtCore.Qt.UserRole + 2
    ExcessRole = QtCore.Qt.UserRole + 3
    ExcessTextRole = QtCore.Qt.UserRole + 4
    LevelTextRole = QtCore.Qt.UserRole + 5
    SteadyRole = QtCore.Qt.UserRole + 6
    SteadinessTextRole = QtCore.Qt.UserRole + 7
    WidthTextRole = QtCore.Qt.UserRole + 8
    WidthRole = QtCore.Qt.UserRole + 9

    # Two candidates this close together are the same line drifting by a
    # bin, not a new one: matching them keeps the row alive.
    SAME_LINE_HZ = 40.0

    _ROLES = {
        FrequencyRole: b"frequency",
        FrequencyTextRole: b"frequency_text",
        ExcessRole: b"excess",
        ExcessTextRole: b"excess_text",
        LevelTextRole: b"level_text",
        SteadyRole: b"steady",
        SteadinessTextRole: b"steadiness_text",
        WidthTextRole: b"width_text",
        WidthRole: b"width",
    }
    _FIELDS = {
        FrequencyRole: "frequency",
        FrequencyTextRole: "frequency_text",
        ExcessRole: "excess",
        ExcessTextRole: "excess_text",
        LevelTextRole: "level_text",
        SteadyRole: "steady",
        SteadinessTextRole: "steadiness_text",
        WidthTextRole: "width_text",
        WidthRole: "width",
    }

    countChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []

    def roleNames(self):
        return self._ROLES

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        field = self._FIELDS.get(role)
        return self._rows[index.row()].get(field) if field else None

    @QtCore.pyqtProperty(int, notify=countChanged)
    def count(self):
        return len(self._rows)

    @QtCore.pyqtSlot(int, result='QVariant')
    def get(self, row):
        """One row as an object, for tests and for QML that wants the whole thing."""
        return dict(self._rows[row]) if 0 <= row < len(self._rows) else {}

    def set_rows(self, rows) -> None:
        """Take the new set of lines, keeping the rows that are still there.

        ``rows`` arrives in any order; it is sorted by frequency here, which
        is the order the list is shown in.
        """
        wanted = sorted(rows, key=lambda r: r["frequency"])
        # Remove the rows that no longer have a line near them, from the
        # back so the indices ahead of each removal stay valid.
        for i in range(len(self._rows) - 1, -1, -1):
            if not any(abs(w["frequency"] - self._rows[i]["frequency"]) < self.SAME_LINE_HZ
                       for w in wanted):
                self.beginRemoveRows(QtCore.QModelIndex(), i, i)
                del self._rows[i]
                self.endRemoveRows()
        changed = False
        for want in wanted:
            match = next((i for i, have in enumerate(self._rows)
                          if abs(have["frequency"] - want["frequency"]) < self.SAME_LINE_HZ), None)
            if match is None:
                at = next((i for i, have in enumerate(self._rows)
                           if have["frequency"] > want["frequency"]), len(self._rows))
                self.beginInsertRows(QtCore.QModelIndex(), at, at)
                self._rows.insert(at, dict(want))
                self.endInsertRows()
                changed = True
            elif self._rows[match] != want:
                self._rows[match] = dict(want)
                idx = self.index(match, 0)
                self.dataChanged.emit(idx, idx, list(self._ROLES.keys()))
        if changed or len(self._rows) != len(wanted):
            self.countChanged.emit()


class BandSurveyViewModel(QtCore.QObject):
    statusTextChanged = QtCore.pyqtSignal(str)
    rangeTextChanged = QtCore.pyqtSignal(str)
    shapeChanged = QtCore.pyqtSignal()
    helpTextChanged = QtCore.pyqtSignal(str)
    tuneRequested = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status_text = ""
        self._range_text = ""
        self._shape: list = []
        self._help_text = ""
        self._lines = SurveyLineModel(self)

    @QtCore.pyqtProperty(QtCore.QObject, constant=True)
    def lines(self):
        return self._lines

    @QtCore.pyqtProperty(str, notify=statusTextChanged)
    def status_text(self):
        return self._status_text

    @status_text.setter  # type: ignore
    def status_text(self, value):
        if self._status_text != value:
            self._status_text = value
            self.statusTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=rangeTextChanged)
    def range_text(self):
        return self._range_text

    @range_text.setter  # type: ignore
    def range_text(self, value):
        if self._range_text != value:
            self._range_text = value
            self.rangeTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=helpTextChanged)
    def help_text(self):
        return self._help_text

    @help_text.setter  # type: ignore
    def help_text(self, value):
        if self._help_text != value:
            self._help_text = value
            self.helpTextChanged.emit(value)

    @QtCore.pyqtProperty('QVariantList', notify=shapeChanged)
    def shape(self):
        return self._shape

    def set_shape(self, rows) -> None:
        rows = list(rows)
        if rows == self._shape:
            return
        self._shape = rows
        self.shapeChanged.emit()

    def set_lines(self, rows) -> None:
        self._lines.set_rows(rows)

    @QtCore.pyqtSlot(float, float)
    def tune(self, frequency_hz: float, width_hz: float) -> None:
        """Clicked a line: point the shared Listen band at it, this wide."""
        self.tuneRequested.emit(float(frequency_hz), float(width_hz))
