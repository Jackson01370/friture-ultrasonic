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

"""Capture rate in, playback rate out.

Nothing captured at 250 kHz can reach a speaker as it is: no sound card takes
that rate, and five sixths of the band is above hearing anyway. Everything
audible therefore passes through here, whether it is the live monitor or the
history player.

Two jobs, done by the one resampler:

  rate       250 kHz down to 48 kHz, which is 125/24 -- not an integer, so
             this is a real fractional resampler rather than a decimator.

  anti-alias soxr lowpasses before it decimates. Without that the whole
             ultrasonic half of the spectrum would fold down into the audible
             band and be heard as a wash of noise that is not there.

What it does NOT do is make ultrasound audible. A 40 kHz call arrives here
above the new Nyquist and is removed, correctly. Moving it down first is what
the heterodyne mode is for; this stage only carries the result out.
"""

from __future__ import annotations

import numpy as np
import soxr

from friture.audiobackend import OUTPUT_SAMPLING_RATE, SAMPLING_RATE
from friture.listen.audio_fifo import AudioFifo


class Playout:
    """Push capture-rate mono, pop playback-rate blocks.

    A queue sits between the two because the resampler returns whatever
    number of samples the ratio happens to produce for a block, while an
    output callback must be handed exactly the number it asked for.
    """

    def __init__(self, capacity: int, prebuffer: int = 0):
        self._fifo = AudioFifo(capacity, prebuffer)
        self._resampler = self._new_resampler()

    @staticmethod
    def _new_resampler():
        return soxr.ResampleStream(SAMPLING_RATE, OUTPUT_SAMPLING_RATE, 1, dtype="float32")

    @property
    def available(self) -> int:
        """Playback-rate samples ready to be popped."""
        return self._fifo.occupancy

    @property
    def n_underruns(self) -> int:
        return self._fifo.n_underruns

    @property
    def n_dropped(self) -> int:
        return self._fifo.n_dropped

    def reset(self) -> None:
        """Start a new stream: drop what is queued and the resampler's tail."""
        self._resampler = self._new_resampler()
        self._fifo.clear()

    def push(self, mono: np.ndarray) -> None:
        resampled = self._resampler.resample_chunk(np.asarray(mono, dtype=np.float32).reshape(-1))
        if resampled.size:
            self._fifo.push(resampled)

    def pop_into(self, out: np.ndarray) -> int:
        return self._fifo.pop_into(out)
