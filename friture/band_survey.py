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

"""The Band Survey dock: what stands out in the spectrum, and how steady it is.

Answers the question the spectrum plot cannot: "there is a hump here, is it
anything?". The noise floor of this microphone slopes by 15 dB across the
band and the room adds humps of its own, so height on the screen says
nothing on its own. Every line here is measured against the spectrum a few
hundred hertz either side of it, and carries how much its level moved over
the seconds it was watched -- which is what separates a device that is
simply on from an event, and both from ripple.

Clicking a line points the shared Listen band at it, so the Digital Decode
dock, the Listen output and the plot overlays all follow with no retyping.

The analysis is friture.demod.survey; this file is the dock plumbing.
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject

from friture.audiobackend import SAMPLING_RATE
from friture.band_survey_view_model import BandSurveyViewModel
from friture.demod.survey import SpectrumSurvey
from friture.listen.listen_band_view_model import (
    MIN_WIDTH_HZ,
    NYQUIST_HZ,
    GetListenBand,
)

DEFAULT_FROM_HZ = 100
DEFAULT_TO_HZ = int(NYQUIST_HZ)
DEFAULT_TOP = 12
DEFAULT_MIN_EXCESS_DB = 4.0
# The band handed to the other docks when a line is clicked is the line's
# OWN measured extent (see SpectrumSurvey._extent), widened by this much so
# the skirts are inside rather than on the edge, and then held between these
# limits: narrower than the filter can resolve is pointless, and wider than
# the decoder's baseband cannot be looked at anyway.
TUNE_MARGIN = 1.6
TUNE_MIN_WIDTH_HZ = 400
TUNE_MAX_WIDTH_HZ = 20000

HELP_TEXT = (
    "Every line is measured against the spectrum a few hundred hertz either side "
    "of it, not against one floor for the whole band: this microphone's own noise "
    "rises by 15 dB towards 25 kHz, so height alone says nothing. "
    "\"Over time\" is how much the line's level moved while it was watched -- "
    "steady means a device that is simply on, and a wide spread means something "
    "that comes and goes. Click a line to point the Listen band at it."
)

# The bands the shape readout is broken into: ordinary listening ranges
# below 10 kHz, then the ultrasonic decades.
SHAPE_EDGES = [100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000,
               15000, 20000, 26000, 32000, 40000, 60000, 90000, 125000]


class BandSurvey_Widget(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.audiobuffer = None
        self._survey = SpectrumSurvey(SAMPLING_RATE)
        self._band = GetListenBand()
        self._paused = False
        self._from_hz = DEFAULT_FROM_HZ
        self._to_hz = DEFAULT_TO_HZ
        self._top = DEFAULT_TOP
        self._min_excess_db = DEFAULT_MIN_EXCESS_DB
        # The lines are recomputed on a timer of their own: the medians and
        # the sort cost a few milliseconds, which is fine twice a second and
        # not fine at the canvas rate.
        self._updates_since_lines = 0

        self._view_model = BandSurveyViewModel(self)
        self._view_model.help_text = HELP_TEXT
        self._view_model.tuneRequested.connect(self._on_tune)
        self._refresh_range_text()

        self.settings_dialog = BandSurvey_Settings_Dialog(parent, self)

    # -- the dock's duck-typed interface ----------------------------------

    def view_model(self):
        return self._view_model

    def qml_file_name(self):
        return "BandSurvey.qml"

    def set_buffer(self, buffer):
        self.audiobuffer = buffer

    def handle_new_data(self, floatdata):
        if self._paused:
            return
        self._survey.process(floatdata[0, :])

    def canvasUpdate(self):
        self._updates_since_lines += 1
        if self._updates_since_lines < 5:
            return
        self._updates_since_lines = 0
        self._refresh()

    def pause(self):
        self._paused = True

    def restart(self):
        # A gap in the stream would average two different rooms together.
        self._survey.reset()
        self._paused = False

    def settings_called(self, checked):
        self.settings_dialog.show()

    def saveState(self, settings):
        self.settings_dialog.saveState(settings)

    def restoreState(self, settings):
        self.settings_dialog.restoreState(settings)

    # -- what the settings dialog drives ------------------------------------

    @property
    def survey(self) -> SpectrumSurvey:
        return self._survey

    def set_range(self, from_hz: float, to_hz: float) -> None:
        self._from_hz, self._to_hz = float(from_hz), float(to_hz)
        self._refresh_range_text()

    def set_top(self, top: int) -> None:
        self._top = max(1, int(top))

    def set_min_excess_db(self, value: float) -> None:
        self._min_excess_db = float(value)

    def restart_survey(self) -> None:
        self._survey.reset()
        self._refresh()

    # -- the readout ---------------------------------------------------------

    def _refresh_range_text(self) -> None:
        self._view_model.range_text = "%.1f - %.1f kHz" % (self._from_hz / 1e3, self._to_hz / 1e3)

    def _refresh(self) -> None:
        vm = self._view_model
        survey = self._survey
        if not survey.ready:
            vm.status_text = "listening ... (%.1f s so far; a first answer needs about a second)" % survey.seconds_held
            vm.set_lines([])
            vm.set_shape([])
            return

        lines = survey.lines(self._from_hz, self._to_hz, top=self._top,
                             min_excess_db=self._min_excess_db)
        vm.status_text = ("%d line%s standing at least %.0f dB over its own neighbourhood, "
                          "averaged over %s (steadiness from the last %.0f s)"
                          % (len(lines), "" if len(lines) == 1 else "s", self._min_excess_db,
                             ("%.0f s" % survey.seconds_held if survey.seconds_held < 90
                              else "%.1f min" % (survey.seconds_held / 60.0)),
                             survey.seconds_watched))
        # The model sorts by frequency; what is IN it is still chosen by how
        # far each line stands over its own floor.
        vm.set_lines([{
            "frequency": line.frequency_hz,
            "frequency_text": ("%.3f kHz" % (line.frequency_hz / 1e3) if line.frequency_hz >= 1000
                               else "%.1f Hz" % line.frequency_hz),
            "width": self._tune_width(line),
            "width_text": self._width_text(line),
            "excess": line.excess_db,
            "excess_text": "%+.1f dB" % line.excess_db,
            "level_text": "%.1f dB" % line.level_db,
            "steady": line.steady,
            "steadiness_text": ("steady" if line.steady
                                else "comes and goes (%.0f dB)" % line.spread_db),
        } for line in lines])

        shape = survey.band_shape(SHAPE_EDGES)
        if shape:
            levels = [row[2] for row in shape]
            lo, hi = min(levels), max(levels)
            span = max(hi - lo, 1e-9)
            vm.set_shape([{
                "band_text": ("%.1f - %.1f kHz" % (a / 1e3, b / 1e3) if a >= 1000
                              else "%.0f - %.0f Hz" % (a, b)),
                "bar": (median - lo) / span,
                "detail_text": "median %6.1f dB   peak %6.1f dB at %8.3f kHz" % (median, peak, f_peak / 1e3),
            } for a, b, median, peak, f_peak in shape])

    @staticmethod
    def _tune_width(line) -> float:
        """The band to give this line: its own extent, widened and clamped."""
        want = line.extent_hz * TUNE_MARGIN
        return float(min(max(want, TUNE_MIN_WIDTH_HZ, MIN_WIDTH_HZ), TUNE_MAX_WIDTH_HZ))

    @classmethod
    def _width_text(cls, line) -> str:
        """What the width column says: the signal's own extent, not the band."""
        extent = line.extent_hz
        if extent < 1.0:
            return "a tone"
        return "%.0f Hz" % extent if extent < 1000 else "%.1f kHz" % (extent / 1e3)

    def _on_tune(self, frequency_hz: float, width_hz: float) -> None:
        """A line was clicked: move the shared Listen band onto it.

        ★ CENTRE, WIDTH, CENTRE AGAIN ★ The band keeps itself inside the
        capture, so each setter can move the other value: asking for a
        width while the band still sits at its OLD centre clamps the width
        against that old position (measured: a 8 kHz band asked for at
        120 kHz came back 2 kHz wide, because the band was still at 1 kHz
        when the width was set), and asking for a centre while the band is
        still its OLD width can clamp the centre. Setting the centre,
        then the width, then the centre once more leaves both where they
        were asked for, and costs one extra assignment.

        A line near either END of the range cannot have a band wider than
        twice its distance from that end without running past it, and the
        band would then slide inward and miss the line -- measured: 124 kHz
        asked for 20 kHz wide came back centred at 115 kHz. Its width is
        reduced instead, which is the honest answer: that is all the room
        there is on that side.
        """
        want = max(width_hz, MIN_WIDTH_HZ)
        headroom = 2.0 * min(frequency_hz, NYQUIST_HZ - frequency_hz)
        want = min(want, max(headroom, MIN_WIDTH_HZ))
        self._band.set_centre_hz(int(frequency_hz))
        self._band.set_width_hz(int(want))
        self._band.set_centre_hz(int(frequency_hz))


class BandSurvey_Settings_Dialog(QtWidgets.QDialog):

    def __init__(self, parent, widget: BandSurvey_Widget):
        super().__init__(parent)

        self._widget = widget
        self.setWindowTitle("Band survey settings")
        self.form_layout = QtWidgets.QFormLayout(self)

        self.spin_from = QtWidgets.QSpinBox(self)
        self.spin_from.setRange(0, int(NYQUIST_HZ) - 1000)
        self.spin_from.setSingleStep(100)
        self.spin_from.setSuffix(" Hz")
        self.spin_from.setValue(DEFAULT_FROM_HZ)
        self.form_layout.addRow("Look from:", self.spin_from)

        self.spin_to = QtWidgets.QSpinBox(self)
        self.spin_to.setRange(1000, int(NYQUIST_HZ))
        self.spin_to.setSingleStep(1000)
        self.spin_to.setSuffix(" Hz")
        self.spin_to.setValue(DEFAULT_TO_HZ)
        self.form_layout.addRow("Look to:", self.spin_to)

        self.spin_top = QtWidgets.QSpinBox(self)
        self.spin_top.setRange(1, 40)
        self.spin_top.setValue(DEFAULT_TOP)
        self.spin_top.setToolTip("How many lines to list, strongest first.")
        self.form_layout.addRow("Lines to list:", self.spin_top)

        self.spin_excess = QtWidgets.QDoubleSpinBox(self)
        self.spin_excess.setRange(1.0, 40.0)
        self.spin_excess.setSingleStep(1.0)
        self.spin_excess.setDecimals(0)
        self.spin_excess.setSuffix(" dB")
        self.spin_excess.setValue(DEFAULT_MIN_EXCESS_DB)
        self.spin_excess.setToolTip(
            "How far a line must stand over the spectrum either side of it to be listed. "
            "Measured in this room: real devices reach 10-27 dB, ripple stays under 6 dB.")
        self.form_layout.addRow("List a line from:", self.spin_excess)

        self.button_restart = QtWidgets.QPushButton("Start again", self)
        self.button_restart.setToolTip(
            "Throw away what has been gathered and start a fresh survey -- "
            "after moving the microphone, or switching something on.")
        self.form_layout.addRow("", self.button_restart)

        self.setLayout(self.form_layout)

        self.spin_from.valueChanged.connect(self._on_range_changed)
        self.spin_to.valueChanged.connect(self._on_range_changed)
        self.spin_top.valueChanged.connect(widget.set_top)
        self.spin_excess.valueChanged.connect(widget.set_min_excess_db)
        self.button_restart.clicked.connect(lambda: widget.restart_survey())

    def _on_range_changed(self, _value=None) -> None:
        lo, hi = self.spin_from.value(), self.spin_to.value()
        if hi <= lo:
            return
        self._widget.set_range(lo, hi)

    # method
    def saveState(self, settings):
        settings.setValue("from_hz", self.spin_from.value())
        settings.setValue("to_hz", self.spin_to.value())
        settings.setValue("top", self.spin_top.value())
        settings.setValue("min_excess_db", self.spin_excess.value())

    # method
    def restoreState(self, settings):
        self.spin_from.setValue(settings.value("from_hz", DEFAULT_FROM_HZ, type=int))
        self.spin_to.setValue(settings.value("to_hz", DEFAULT_TO_HZ, type=int))
        self.spin_top.setValue(settings.value("top", DEFAULT_TOP, type=int))
        self.spin_excess.setValue(settings.value("min_excess_db", DEFAULT_MIN_EXCESS_DB, type=float))
