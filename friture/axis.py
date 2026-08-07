from PyQt5 import QtCore
from PyQt5.QtCore import pyqtProperty, pyqtSlot

from friture.plotting.coordinateTransform import CoordinateTransform
from friture.plotting.scaleDivision import ScaleDivision

class Axis(QtCore.QObject):
    name_changed = QtCore.pyqtSignal(str)
    range_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._name = "Axis Name"
        self._formatter = lambda x: str(x)
        self._scale_division = ScaleDivision(-1., 1., self)
        self._coordinate_transform = CoordinateTransform(-1, 1, 1., 0, 0, self)
        self._show_minor_grid_lines = False

    @pyqtProperty(str, notify=name_changed) # type: ignore
    def name(self):
        return self._name

    @name.setter # type: ignore
    def name(self, name):
        if self._name != name:
            self._name = name
            self.name_changed.emit(name)

    def setTrackerFormatter(self, formatter):
        if self._formatter != formatter:
            self._formatter = formatter

    @pyqtSlot(float, result=str) # type: ignore
    def formatTracker(self, value):
        return self._formatter(value)

    def setRange(self, scale_min, scale_max):
        self._scale_division.setRange(scale_min, scale_max)
        self._coordinate_transform.setRange(scale_min, scale_max)
        self.range_changed.emit()

    def setScale(self, scale):
        self._scale_division.setScale(scale)
        self._coordinate_transform.setScale(scale)
        self.range_changed.emit()

    # coordinate_transform maps values through slots, and QML cannot see the
    # axis state a slot reads. These two exist so a binding that calls
    # toScreen() can name the dependency and be re-evaluated when the
    # displayed range or scale changes.
    @pyqtProperty(float, notify=range_changed) # type: ignore
    def range_min(self):
        return self._coordinate_transform.coord_min

    @pyqtProperty(float, notify=range_changed) # type: ignore
    def range_max(self):
        return self._coordinate_transform.coord_max

    @pyqtProperty(ScaleDivision, constant=True) # type: ignore
    def scale_division(self):
        return self._scale_division

    @pyqtProperty(bool, constant=True) # type: ignore
    def show_minor_grid_lines(self) -> bool:
        return self._show_minor_grid_lines

    @show_minor_grid_lines.setter # type: ignore
    def show_minor_grid_lines(self, show: bool):
        self._show_minor_grid_lines = show

    @pyqtProperty(CoordinateTransform, constant=True) # type: ignore
    def coordinate_transform(self):
        return self._coordinate_transform
