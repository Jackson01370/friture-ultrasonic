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

"""What ReplayBar.qml shows, and the requests it sends back.

Display state only; friture.replay_control.ReplayController does the work
and writes the results here.
"""

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot


def _prop(name, kind, default):
    """A notifying Qt property backed by self._<name>."""
    signal = pyqtSignal(kind)

    def getter(self):
        return getattr(self, "_" + name, default)

    def setter(self, value):
        if getattr(self, "_" + name, default) != value:
            setattr(self, "_" + name, value)
            getattr(self, name + "_changed").emit(value)

    return signal, pyqtProperty(kind, fget=getter, fset=setter, notify=signal)


class ReplayViewModel(QtCore.QObject):
    active_changed, active = _prop("active", bool, False)
    playing_changed, playing = _prop("playing", bool, False)
    position_text_changed, position_text = _prop("position_text", str, "")
    start_text_changed, start_text = _prop("start_text", str, "")
    end_text_changed, end_text = _prop("end_text", str, "")
    fraction_changed, fraction = _prop("fraction", float, 0.0)
    speed_changed, speed = _prop("speed", float, 1.0)
    status_text_changed, status_text = _prop("status_text", str, "")
    folder_text_changed, folder_text = _prop("folder_text", str, "")
    current_protected_changed, current_protected = _prop("current_protected", bool, False)

    ranges_changed = pyqtSignal()
    protect_requested = pyqtSignal()

    # requests, handled by ReplayController
    toggle_requested = pyqtSignal()
    play_pause_requested = pyqtSignal()
    seek_requested = pyqtSignal(float)          # a fraction of the span
    step_requested = pyqtSignal(float)          # seconds, either way
    speed_requested = pyqtSignal(float)
    latest_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ranges = []

    # stretches of the span that hold audio, as [from, to] fractions: the
    # bar draws them so the gaps in the recording are visible
    @pyqtProperty('QVariantList', notify=ranges_changed)
    def ranges(self):
        return self._ranges

    def set_ranges(self, ranges) -> None:
        ranges = [list(r) for r in ranges]
        if ranges != self._ranges:
            self._ranges = ranges
            self.ranges_changed.emit()

    @pyqtSlot()
    def toggle(self):
        self.toggle_requested.emit()

    @pyqtSlot()
    def play_pause(self):
        self.play_pause_requested.emit()

    @pyqtSlot(float)
    def seek(self, fraction):
        self.seek_requested.emit(float(fraction))

    @pyqtSlot(float)
    def step(self, seconds):
        self.step_requested.emit(float(seconds))

    @pyqtSlot(float)
    def set_speed(self, speed):
        self.speed_requested.emit(float(speed))

    @pyqtSlot()
    def latest(self):
        self.latest_requested.emit()

    @pyqtSlot()
    def toggle_protect(self):
        self.protect_requested.emit()
