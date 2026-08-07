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

"""Turns the shared band into a filter, and applies it to blocks of audio.

process() runs on whichever thread is producing audio -- the PortAudio
output callback for playback, the GUI thread for the live monitor -- while
the band is retuned from the GUI thread by a click. So a retune is *staged*
and swapped in at the next block boundary, never in the middle of a pass.

One processor per audio path. The player and the live monitor each get their
own, because a filter's tail is the tail of one particular stream and mixing
two streams through it would smear one into the other.
"""

import logging
import threading
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject

from friture.audiobackend import SAMPLING_RATE
from friture.listen.agc import Agc
from friture.listen.band_dsp import BandFilter, make_filter
from friture.listen.denoise import SpectralDenoiser
from friture.listen.gate import NoiseGate
from friture.listen.limiter import SoftLimiter
from friture.listen.listen_band_view_model import ListenBandViewModel

logger = logging.getLogger(__name__)


class BandProcessor(QObject):

    def __init__(self, parent: Optional[QObject], band: ListenBandViewModel) -> None:
        super().__init__(parent)

        self._band = band
        self._lock = threading.Lock()
        self._filter: Optional[BandFilter] = None
        self._tuned_to = None    # the (f_lo, width, mode) the filter is set to
        self._gain = 1.0
        # Runs at the capture rate, on the band-limited signal: measuring the
        # level of the band being listened to is the whole point, and measuring
        # it before the filter would just track whatever else is in the room.
        self._agc = Agc(SAMPLING_RATE)
        self._denoiser = SpectralDenoiser(SAMPLING_RATE)
        self._gate = NoiseGate(SAMPLING_RATE)
        # Loudest sample since the meter last looked. With the gain reaching
        # +150 dB there has to be something on screen saying when the output
        # has stopped being louder and started being a square wave.
        self._peak = 0.0
        self._limiter = SoftLimiter(SAMPLING_RATE)
        # Deepest gain reduction since the meter last looked. Peak-hold, the
        # way a limiter meter always is: the reductions that matter last a
        # couple of milliseconds and would never be caught by sampling.
        self._reduction_db = 0.0

        band.gain_changed.connect(self._on_gain_changed)
        self._on_gain_changed(band.gain_db)
        band.register_gain_source(self)

    @property
    def active(self) -> bool:
        return self._band.enabled

    def _on_gain_changed(self, gain_db: int) -> None:
        with self._lock:
            self._gain = float(10.0 ** (gain_db / 20.0))

    def take_reduction_db(self) -> float:
        """Deepest gain reduction since the last call, then start again."""
        reduction, self._reduction_db = self._reduction_db, 0.0
        return reduction

    def take_peak(self) -> float:
        """Loudest output sample since the last call, then start again.

        Read from the GUI thread while process() runs on the audio one. A
        float read and a float write, so the worst case is losing one
        window's peak to a race -- and the meter is polled five times a
        second, so it would be replaced before anyone saw it.
        """
        peak, self._peak = self._peak, 0.0
        return peak

    @property
    def agc_gain_db(self) -> float:
        """What the AGC is currently adding, for the readout in the bar."""
        return self._agc.gain_db

    def reset(self) -> None:
        """Forget the stream tail -- call when the audio stream restarts."""
        with self._lock:
            self._filter = None
            self._tuned_to = None
            self._agc.reset()
            self._denoiser.reset()
            self._gate.reset()
            self._limiter.reset()

    def process(self, mono: np.ndarray) -> np.ndarray:
        """Band-limit one block. Same length in and out.

        The band is sampled here rather than pushed in on a signal, so a drag
        that moves it a hundred times between blocks costs one retune. And a
        retune, unlike a rebuild, does not restart the stream: the mixer phase
        and the filter history both carry on, which is what lets the band be
        dragged around while the audio keeps playing.

        Returns the input untouched if the band cannot be built, which the
        clamping in ListenBandViewModel should already have made impossible;
        passing audio through beats dropping into silence with no clue why.
        """
        with self._lock:
            wanted = self._band.band_snapshot()
            if wanted != self._tuned_to:
                self._tune(wanted)

            if self._filter is None:
                return mono

            band = self._filter.process(mono)

            # Order matters. Noise reduction first, so the gate and the AGC
            # both judge the level of what is actually wanted rather than of
            # the room's steady interference. The gate next, so the AGC can
            # be told to stop adapting while there is nothing to listen to.
            if self._band.denoise_enabled:
                band = self._denoiser.process(band)

            gate_open = True
            if self._band.gate_enabled:
                band = self._gate.process(band)
                gate_open = self._gate.is_open

            if self._band.agc_enabled:
                band = self._agc.process(band, adapt=gate_open)

            # The manual gain stays a trim on top rather than an alternative:
            # the AGC lands the level in the right range and this moves it to
            # taste. They do not fight -- the AGC targets its own output, and
            # this scales what comes after.
            out = band * self._gain

            # Last, after the trim: this is about what leaves for the sound
            # card, and everything before it can push the level up.
            if self._band.limiter_enabled:
                out = self._limiter.process(out)
                self._reduction_db = min(self._reduction_db, self._limiter.reduction_db)

            if out.size:
                self._peak = max(self._peak, float(np.max(np.abs(out))))
            return out

    def _tune(self, wanted) -> None:
        f_lo, width, mode = wanted
        rebuild = self._filter is None or mode != self._tuned_to[2]
        try:
            if rebuild:
                # A new mode is a different structure, so the stream does
                # restart here -- the only case that still clicks.
                band_filter = make_filter(mode)
                band_filter.configure(f_lo, width, SAMPLING_RATE)
                self._filter = band_filter
            else:
                self._filter.retune(f_lo, width)
        except ValueError:
            logger.exception("Cannot listen to %.0f Hz + %.0f Hz", f_lo, width)
            self._filter = None
        self._tuned_to = wanted
