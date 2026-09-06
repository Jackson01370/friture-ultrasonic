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

"""What DigitalDecode.qml shows: strings, flags, the symbol strip's data
and the analog waveform.

Display only. The decoder writes its readouts as plain attributes on the
GUI thread and the widget copies them here on each canvasUpdate; the QML
binds to these properties and nothing else.
"""

from PyQt5 import QtCore


class DigitalDecodeViewModel(QtCore.QObject):
    modeTextChanged = QtCore.pyqtSignal(str)
    modeHintChanged = QtCore.pyqtSignal(str)
    bandTextChanged = QtCore.pyqtSignal(str)
    rangeTextChanged = QtCore.pyqtSignal(str)
    signalTextChanged = QtCore.pyqtSignal(str)
    decodeTextChanged = QtCore.pyqtSignal(str)
    bitsChanged = QtCore.pyqtSignal(str)
    lockedChanged = QtCore.pyqtSignal(bool)
    symbolsChanged = QtCore.pyqtSignal()
    helpTextChanged = QtCore.pyqtSignal(str)
    analogChanged = QtCore.pyqtSignal(bool)
    levelsChanged = QtCore.pyqtSignal(int)
    traceChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode_text = ""
        self._mode_hint = ""
        self._band_text = ""
        self._range_text = ""
        self._signal_text = ""
        self._decode_text = ""
        self._bits = ""
        self._locked = False
        self._symbols: list = []
        self._agreements: list = []
        self._help_text = ""
        self._analog = False
        self._levels = 2
        self._trace: list = []
        self._trace_hi_text = ""
        self._trace_lo_text = ""

    @QtCore.pyqtProperty(str, notify=modeTextChanged)
    def mode_text(self):
        return self._mode_text

    @mode_text.setter  # type: ignore
    def mode_text(self, value):
        if self._mode_text != value:
            self._mode_text = value
            self.modeTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=modeHintChanged)
    def mode_hint(self):
        return self._mode_hint

    @mode_hint.setter  # type: ignore
    def mode_hint(self, value):
        if self._mode_hint != value:
            self._mode_hint = value
            self.modeHintChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=bandTextChanged)
    def band_text(self):
        return self._band_text

    @band_text.setter  # type: ignore
    def band_text(self, value):
        if self._band_text != value:
            self._band_text = value
            self.bandTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=rangeTextChanged)
    def range_text(self):
        return self._range_text

    @range_text.setter  # type: ignore
    def range_text(self, value):
        if self._range_text != value:
            self._range_text = value
            self.rangeTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=signalTextChanged)
    def signal_text(self):
        return self._signal_text

    @signal_text.setter  # type: ignore
    def signal_text(self, value):
        if self._signal_text != value:
            self._signal_text = value
            self.signalTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=decodeTextChanged)
    def decode_text(self):
        return self._decode_text

    @decode_text.setter  # type: ignore
    def decode_text(self, value):
        if self._decode_text != value:
            self._decode_text = value
            self.decodeTextChanged.emit(value)

    @QtCore.pyqtProperty(str, notify=bitsChanged)
    def bits(self):
        return self._bits

    @bits.setter  # type: ignore
    def bits(self, value):
        if self._bits != value:
            self._bits = value
            self.bitsChanged.emit(value)

    @QtCore.pyqtProperty(bool, notify=lockedChanged)
    def locked(self):
        return self._locked

    @locked.setter  # type: ignore
    def locked(self, value):
        value = bool(value)
        if self._locked != value:
            self._locked = value
            self.lockedChanged.emit(value)

    # The two lists change together, once per canvas update, so they share
    # one notification: the strip repaints once, not twice.
    @QtCore.pyqtProperty('QVariantList', notify=symbolsChanged)
    def symbols(self):
        return self._symbols

    @QtCore.pyqtProperty('QVariantList', notify=symbolsChanged)
    def agreements(self):
        return self._agreements

    def set_symbols(self, symbols, agreements) -> None:
        symbols = [int(s) for s in symbols]
        agreements = [float(a) for a in agreements]
        if symbols == self._symbols and agreements == self._agreements:
            return
        self._symbols = symbols
        self._agreements = agreements
        self.symbolsChanged.emit()

    @QtCore.pyqtProperty(str, notify=helpTextChanged)
    def help_text(self):
        return self._help_text

    @help_text.setter  # type: ignore
    def help_text(self, value):
        if self._help_text != value:
            self._help_text = value
            self.helpTextChanged.emit(value)

    # -- analog modes ------------------------------------------------------

    @QtCore.pyqtProperty(bool, notify=analogChanged)
    def analog(self):
        return self._analog

    @analog.setter  # type: ignore
    def analog(self, value):
        value = bool(value)
        if self._analog != value:
            self._analog = value
            self.analogChanged.emit(value)

    # how many levels the symbol strip draws (2 or 4)
    @QtCore.pyqtProperty(int, notify=levelsChanged)
    def levels(self):
        return self._levels

    @levels.setter  # type: ignore
    def levels(self, value):
        value = max(2, int(value))
        if self._levels != value:
            self._levels = value
            self.levelsChanged.emit(value)

    # The waveform and its axis labels change together, once per canvas
    # update, so they share one notification.
    @QtCore.pyqtProperty('QVariantList', notify=traceChanged)
    def trace(self):
        return self._trace

    @QtCore.pyqtProperty(str, notify=traceChanged)
    def trace_hi_text(self):
        return self._trace_hi_text

    @QtCore.pyqtProperty(str, notify=traceChanged)
    def trace_lo_text(self):
        return self._trace_lo_text

    def set_trace(self, values, hi_text: str, lo_text: str) -> None:
        values = [float(v) for v in values]
        if values == self._trace and hi_text == self._trace_hi_text and lo_text == self._trace_lo_text:
            return
        self._trace = values
        self._trace_hi_text = hi_text
        self._trace_lo_text = lo_text
        self.traceChanged.emit()
