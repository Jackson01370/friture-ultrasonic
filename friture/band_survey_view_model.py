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
"""

from PyQt5 import QtCore


class BandSurveyViewModel(QtCore.QObject):
    statusTextChanged = QtCore.pyqtSignal(str)
    rangeTextChanged = QtCore.pyqtSignal(str)
    linesChanged = QtCore.pyqtSignal()
    shapeChanged = QtCore.pyqtSignal()
    helpTextChanged = QtCore.pyqtSignal(str)
    tuneRequested = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status_text = ""
        self._range_text = ""
        self._lines: list = []
        self._shape: list = []
        self._help_text = ""

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

    # One entry per line: frequency, what it stands out by, how steady it is.
    @QtCore.pyqtProperty('QVariantList', notify=linesChanged)
    def lines(self):
        return self._lines

    def set_lines(self, rows) -> None:
        rows = list(rows)
        if rows == self._lines:
            return
        self._lines = rows
        self.linesChanged.emit()

    @QtCore.pyqtProperty('QVariantList', notify=shapeChanged)
    def shape(self):
        return self._shape

    def set_shape(self, rows) -> None:
        rows = list(rows)
        if rows == self._shape:
            return
        self._shape = rows
        self.shapeChanged.emit()

    @QtCore.pyqtSlot(float)
    def tune(self, frequency_hz: float) -> None:
        """Clicked a line: point the shared Listen band at it."""
        self.tuneRequested.emit(float(frequency_hz))
