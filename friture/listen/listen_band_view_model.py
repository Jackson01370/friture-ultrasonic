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

"""The one band that every plot, the player and the live monitor share.

Friture's plots are docks the user creates by hand, so there may be no
spectrogram, or three of them. The band therefore cannot live inside a
widget: it is a process-wide singleton, reached through GetListenBand(),
that each plot binds to as it is built. Click a peak in any of them and
they all move together.
"""

import math

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot

from friture.audiobackend import OUTPUT_SAMPLING_RATE, SAMPLING_RATE
from friture.listen.band_dsp import (
    BANDPASS,
    HETERODYNE,
    group_delay_seconds,
    min_bandwidth,
    taps_for_bandwidth,
)

NYQUIST_HZ = SAMPLING_RATE / 2.0

# The narrowest band the longest filter can resolve. A real floor set by the
# filter, not an arbitrary UI limit -- and going below it would not help
# anyway, since a narrower band would be nothing but roll-off.
MIN_WIDTH_HZ = int(math.ceil(min_bandwidth(SAMPLING_RATE)))

# Where bats are. A default in the audible range would be a strange place to
# start on a microphone bought to hear above it.
DEFAULT_CENTRE_HZ = 45000
DEFAULT_WIDTH_HZ = 10000

__band_instance = None


def GetListenBand():
    global __band_instance
    if __band_instance is None:
        __band_instance = ListenBandViewModel()
    return __band_instance


class ListenBandViewModel(QtCore.QObject):
    # Anything the DSP has to react to: a new mode, edge or width. Rebuilding
    # a filter resets its stream state, so this is also the "expect a click"
    # signal.
    band_changed = pyqtSignal()

    enabled_changed = pyqtSignal(bool)
    monitoring_changed = pyqtSignal(bool)
    agc_changed = pyqtSignal(bool)
    agc_gain_changed = pyqtSignal()
    denoise_changed = pyqtSignal(bool)
    gate_changed = pyqtSignal(bool)
    clipping_changed = pyqtSignal()
    limiter_changed = pyqtSignal(bool)
    limiter_reduction_changed = pyqtSignal()
    mode_changed = pyqtSignal(int)
    centre_changed = pyqtSignal()
    width_changed = pyqtSignal()
    gain_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._enabled = False
        self._monitoring = False
        # On by default: the request this answers is "make quiet things
        # audible", and an AGC that has to be found and switched on does not.
        self._agc_enabled = True
        self._agc_gain_db = 0.0
        self._gain_source = None
        # Both off by default. Each changes what you hear in ways that are
        # not always wanted -- noise reduction can smear a faint call into
        # the background it is being subtracted from, and a gate can clip
        # one entirely. Neither should be discovered by accident.
        self._denoise_enabled = False
        self._gate_enabled = False
        self._clipping = False
        # On by default. Below the ceiling it does nothing but delay by
        # 2.3 ms; above it, it is the difference between loud and ruined.
        self._limiter_enabled = True
        self._limiter_reduction_db = 0.0
        # Heterodyne, not band-pass: above ~24 kHz band-pass has nothing to
        # give, since the sound card cannot reproduce what it selects. Moving
        # the band down is the only way to hear anything up there.
        self._mode = HETERODYNE
        self._gain_db = 0
        # One tuple, not two floats: the audio thread reads this while the GUI
        # writes it, and a single rebind cannot be caught half-done the way two
        # assignments can.
        self._band_hz = self._moved(DEFAULT_CENTRE_HZ, DEFAULT_WIDTH_HZ)

    # -- the band itself ----------------------------------------------------
    #
    # Two ways to change it, with opposite priorities:
    #
    #   moving it   keeps the width -- you aimed at a frequency, and the band
    #               you were listening through should be the one that lands
    #               there. At an edge the centre gives way.
    #
    #   resizing it keeps the centre -- you aimed first and are now adjusting
    #               how much around it you hear. At an edge the width gives
    #               way rather than sliding the target out from under you.

    @staticmethod
    def _moved(centre: float, width: float):
        """Fit (centre, width) inside [0, Nyquist], preserving the width."""
        width = min(max(float(width), MIN_WIDTH_HZ), NYQUIST_HZ)
        f_lo = min(max(centre - width / 2.0, 0.0), NYQUIST_HZ - width)
        return f_lo + width / 2.0, width

    @staticmethod
    def _resized(centre: float, width: float):
        """Fit (centre, width) inside [0, Nyquist], preserving the centre."""
        room = 2.0 * min(centre, NYQUIST_HZ - centre)
        return centre, min(max(float(width), MIN_WIDTH_HZ), room)

    def _apply(self, band) -> None:
        centre, width = band
        old_centre, old_width = self._band_hz
        if centre == old_centre and width == old_width:
            return

        self._band_hz = band
        if centre != old_centre:
            self.centre_changed.emit()
        if width != old_width:
            self.width_changed.emit()
        self.band_changed.emit()

    @pyqtSlot(float)
    def click_center(self, frequency: float) -> None:
        """Centre the band on a frequency, leaving the width alone.

        Centred rather than edge-aligned because the heterodyne maps
        [f_lo, f_lo+width] onto [0, width]: a centred target lands at width/2,
        clear of the DC edge where anything at the band bottom becomes 0 Hz
        and vanishes, and clear of the roll-off at the top.

        Also the drag handler -- dragging the band is a stream of these, and
        the filter retunes without breaking, so there is nothing to throttle.
        """
        if not math.isfinite(frequency):
            return
        self._apply(self._moved(frequency, self._band_hz[1]))

    @pyqtSlot(float)
    def drag_edge(self, frequency: float) -> None:
        """Pull an edge to a frequency, keeping the centre where it is.

        Which edge was grabbed does not need saying: the centre is fixed, so
        an edge at `frequency` means a half-width of |frequency - centre|
        either way.
        """
        if not math.isfinite(frequency):
            return
        centre = self._band_hz[0]
        self._apply(self._resized(centre, 2.0 * abs(frequency - centre)))

    def band_snapshot(self):
        """(f_lo, width, mode) for the audio thread, in one consistent read.

        The audio path asks for this once per block rather than reacting to
        band_changed, which is what keeps a drag from mattering: however many
        times the mouse moves in between, the filter is retuned once, to
        wherever the band ended up.
        """
        centre, width = self._band_hz
        return centre - width / 2.0, width, self._mode

    # band_changed, not centre_changed: the edges move when the width changes
    # too, and the overlay drawn from them has to follow both.
    @pyqtProperty(float, notify=band_changed)  # type: ignore
    def f_lo(self) -> float:
        centre, width = self._band_hz
        return centre - width / 2.0

    @pyqtProperty(float, notify=band_changed)  # type: ignore
    def f_hi(self) -> float:
        centre, width = self._band_hz
        return centre + width / 2.0

    def get_centre_hz(self) -> int:
        return int(round(self._band_hz[0]))

    def set_centre_hz(self, centre: int) -> None:
        self.click_center(float(centre))

    centre_hz = pyqtProperty(int, fget=get_centre_hz, fset=set_centre_hz, notify=centre_changed)

    def get_width_hz(self) -> int:
        return int(round(self._band_hz[1]))

    def set_width_hz(self, width: int) -> None:
        self._apply(self._resized(self._band_hz[0], float(width)))

    width_hz = pyqtProperty(int, fget=get_width_hz, fset=set_width_hz, notify=width_changed)

    # -- modes and switches -------------------------------------------------

    def get_mode(self) -> int:
        return self._mode

    def set_mode(self, mode: int) -> None:
        mode = HETERODYNE if mode == HETERODYNE else BANDPASS
        if self._mode != mode:
            self._mode = mode
            self.mode_changed.emit(mode)
            self.band_changed.emit()

    mode = pyqtProperty(int, fget=get_mode, fset=set_mode, notify=mode_changed)

    def get_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        if self._enabled == enabled:
            return
        self._enabled = enabled
        self.enabled_changed.emit(enabled)
        if not enabled:
            # The monitor exists to play the band; with no band there is
            # nothing for it to play, and leaving it running would put raw
            # input on the speakers, which is a howl waiting to happen.
            self.set_monitoring(False)

    enabled = pyqtProperty(bool, fget=get_enabled, fset=set_enabled, notify=enabled_changed)

    def get_monitoring(self) -> bool:
        return self._monitoring

    def set_monitoring(self, monitoring: bool) -> None:
        if monitoring and not self._enabled:
            self.set_enabled(True)
        if self._monitoring != monitoring:
            self._monitoring = monitoring
            self.monitoring_changed.emit(monitoring)

    monitoring = pyqtProperty(bool, fget=get_monitoring, fset=set_monitoring, notify=monitoring_changed)

    def get_agc_enabled(self) -> bool:
        return self._agc_enabled

    def set_agc_enabled(self, enabled: bool) -> None:
        if self._agc_enabled != enabled:
            self._agc_enabled = enabled
            self.agc_changed.emit(enabled)

    agc_enabled = pyqtProperty(bool, fget=get_agc_enabled, fset=set_agc_enabled, notify=agc_changed)

    def get_denoise_enabled(self) -> bool:
        return self._denoise_enabled

    def set_denoise_enabled(self, enabled: bool) -> None:
        if self._denoise_enabled != enabled:
            self._denoise_enabled = enabled
            self.denoise_changed.emit(enabled)

    denoise_enabled = pyqtProperty(bool, fget=get_denoise_enabled,
                                   fset=set_denoise_enabled, notify=denoise_changed)

    def get_gate_enabled(self) -> bool:
        return self._gate_enabled

    def set_gate_enabled(self, enabled: bool) -> None:
        if self._gate_enabled != enabled:
            self._gate_enabled = enabled
            self.gate_changed.emit(enabled)

    gate_enabled = pyqtProperty(bool, fget=get_gate_enabled,
                                fset=set_gate_enabled, notify=gate_changed)

    def register_gain_source(self, processor) -> None:
        """Whose AGC the readout shows.

        The first to register wins, and there is no attempt to show both: the
        live monitor and the player see the same band through the same
        settings, so their gains track each other closely enough that picking
        one is not a lie.
        """
        if self._gain_source is None:
            self._gain_source = processor
            timer = QtCore.QTimer(self)
            # Polled rather than pushed: the gain moves every block, 122 times
            # a second at 250 kHz, and a readout that changes that fast is
            # unreadable as well as wasteful.
            timer.setInterval(200)
            timer.timeout.connect(self._refresh_meters)
            timer.start()

    def _refresh_meters(self) -> None:
        if self._gain_source is None:
            return

        clipping = self._gain_source.take_peak() > 1.0
        if clipping != self._clipping:
            self._clipping = clipping
            self.clipping_changed.emit()

        reduction = self._gain_source.take_reduction_db()
        if abs(reduction - self._limiter_reduction_db) >= 0.5:
            self._limiter_reduction_db = reduction
            self.limiter_reduction_changed.emit()

        gain_db = self._gain_source.agc_gain_db
        # A tenth of a dB is below the point of caring, and holding the text
        # still is worth more than tracking it exactly.
        if abs(gain_db - self._agc_gain_db) >= 0.5:
            self._agc_gain_db = gain_db
            self.agc_gain_changed.emit()

    def get_limiter_enabled(self) -> bool:
        return self._limiter_enabled

    def set_limiter_enabled(self, enabled: bool) -> None:
        if self._limiter_enabled != enabled:
            self._limiter_enabled = enabled
            self.limiter_changed.emit(enabled)

    limiter_enabled = pyqtProperty(bool, fget=get_limiter_enabled,
                                   fset=set_limiter_enabled, notify=limiter_changed)

    @pyqtProperty(int, notify=limiter_reduction_changed)  # type: ignore
    def limiter_reduction_db(self) -> int:
        """How hard the limiter is working, as a negative number of dB.

        Worth showing for the same reason a mixing desk shows it: it says how
        far past the ceiling the gain has been pushed, which nothing else on
        screen reveals once the limiter is stopping it from being audible.
        """
        return int(round(self._limiter_reduction_db))

    @pyqtProperty(bool, notify=clipping_changed)  # type: ignore
    def clipping(self) -> bool:
        """Whether the output is running past full scale.

        Past that point more gain is not more volume: the peaks are already
        flat, and what grows is the distortion. Worth saying plainly, because
        the slider goes far enough that this is easy to reach.
        """
        return self._clipping

    @pyqtProperty(int, notify=agc_gain_changed)  # type: ignore
    def agc_gain_db(self) -> int:
        return int(round(self._agc_gain_db))

    def get_gain_db(self) -> int:
        return self._gain_db

    def set_gain_db(self, gain_db: int) -> None:
        if self._gain_db != gain_db:
            self._gain_db = int(gain_db)
            self.gain_changed.emit(self._gain_db)

    gain_db = pyqtProperty(int, fget=get_gain_db, fset=set_gain_db, notify=gain_changed)

    # -- read-only surface for QML ------------------------------------------

    @pyqtProperty(int, constant=True)  # type: ignore
    def min_width_hz(self) -> int:
        return MIN_WIDTH_HZ

    @pyqtProperty(int, constant=True)  # type: ignore
    def max_freq_hz(self) -> int:
        return int(NYQUIST_HZ)

    @pyqtProperty(int, notify=width_changed)  # type: ignore
    def latency_ms(self) -> int:
        """How far behind the input a narrow band comes out.

        Worth showing: it is the price of the width the user just asked for,
        it climbs steeply at the narrow end, and while listening live it is
        the difference between a sound and its echo.
        """
        taps = taps_for_bandwidth(self._band_hz[1], SAMPLING_RATE)
        return int(round(1000.0 * group_delay_seconds(taps, SAMPLING_RATE)))

    # The highest frequency that can survive playback. Everything above it is
    # removed on the way to the sound card, which is a rate conversion, not a
    # setting -- see friture.listen.playout.
    AUDIBLE_TOP_HZ = OUTPUT_SAMPLING_RATE / 2.0

    @pyqtProperty(str, notify=band_changed)  # type: ignore
    def playback_note(self) -> str:
        """Why the selected band may not be heard, or "" when it will be.

        Band-pass leaves the band where it is, and on a 250 kHz capture that
        is usually somewhere no speaker can follow. Saying so beats letting
        someone conclude the bats are not calling.
        """
        if self._mode == HETERODYNE:
            return ""
        if self.f_lo >= self.AUDIBLE_TOP_HZ:
            return "silent on playback — use heterodyne"
        if self.f_hi > self.AUDIBLE_TOP_HZ:
            return "only below %s is heard" % _khz(self.AUDIBLE_TOP_HZ)
        return ""

    @pyqtProperty(bool, notify=mode_changed)  # type: ignore
    def shifts_frequency(self) -> bool:
        """Whether what you hear sits somewhere other than where you clicked."""
        return self._mode == HETERODYNE

    @pyqtProperty(str, notify=band_changed)  # type: ignore
    def status_text(self) -> str:
        span = "%s – %s" % (_khz(self.f_lo), _khz(self.f_hi))
        if self._mode == HETERODYNE:
            return "%s → 0 – %s" % (span, _khz(self._band_hz[1]))
        return span


def _khz(hz: float) -> str:
    if hz < 1000.0:
        return "%d Hz" % int(round(hz))
    return "%.2f kHz" % (hz / 1000.0)
