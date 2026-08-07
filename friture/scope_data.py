#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2021 Timothée Lecomte

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

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtProperty
from PyQt5.QtQml import QQmlListProperty # type: ignore

from friture.axis import Axis
from friture.curve import Curve
from friture.listen.listen_band_view_model import ListenBandViewModel

class Scope_Data(QtCore.QObject):
    show_color_axis_changed = QtCore.pyqtSignal(bool)
    show_legend_changed = QtCore.pyqtSignal(bool)
    plot_items_changed = QtCore.pyqtSignal()
    listen_band_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._plot_items = []
        self._horizontal_axis = Axis(self)
        self._vertical_axis = Axis(self)
        self._color_axis = Axis(self)
        self._show_color_axis = False
        self._show_legend = True
        self._listen_band = None
        self._freq_axis = ""

    @pyqtProperty(QQmlListProperty, notify=plot_items_changed) # type: ignore
    def plot_items(self):
        return QQmlListProperty(Curve, self, self._plot_items)

    def insert_plot_item(self, index, plot_item):
        self._plot_items.insert(index, plot_item)
        self.plot_items_changed.emit()

    def add_plot_item(self, plot_item):
        self._plot_items.append(plot_item)
        plot_item.setParent(self) # take ownership
        self.plot_items_changed.emit()

    def remove_plot_item(self, plot_item):
        self._plot_items.remove(plot_item)
        self.plot_items_changed.emit()

    @pyqtProperty(Axis, constant=True) # type: ignore
    def horizontal_axis(self):
        return self._horizontal_axis

    @pyqtProperty(Axis, constant=True) # type: ignore
    def vertical_axis(self):
        return self._vertical_axis

    @pyqtProperty(Axis, constant=True)
    def color_axis(self):
        return self._color_axis
    
    @pyqtProperty(bool, notify=show_color_axis_changed)
    def show_color_axis(self):
        return self._show_color_axis
    
    @show_color_axis.setter
    def show_color_axis(self, show_color_axis):
        if self._show_color_axis != show_color_axis:
            self._show_color_axis = show_color_axis
            self.show_color_axis_changed.emit(show_color_axis)
    
    # Band listening. Only the plots that actually have a frequency axis set
    # these; everywhere else listen_band stays null and the plot behaves
    # exactly as it did before.
    def set_listen_band(self, listen_band, freq_axis):
        """freq_axis is "vertical" or "horizontal": which axis is in Hz."""
        self._listen_band = listen_band
        self._freq_axis = freq_axis
        self.listen_band_changed.emit()

    @pyqtProperty(ListenBandViewModel, notify=listen_band_changed) # type: ignore
    def listen_band(self):
        return self._listen_band

    @pyqtProperty(str, notify=listen_band_changed) # type: ignore
    def freq_axis(self):
        return self._freq_axis

    @pyqtProperty(bool, notify=show_legend_changed) # type: ignore
    def show_legend(self):
        return self._show_legend

    @show_legend.setter # type: ignore
    def show_legend(self, show_legend):
        if self._show_legend != show_legend:
            self._show_legend = show_legend
            self.show_legend_changed.emit(show_legend)
