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

"""Keep the output inside full scale by turning it down, not by cutting it off.

With the gain reaching +150 dB, something has to happen at the top. Doing
nothing means the conversion to int16 flattens every peak: measured on a
quiet 40-50 kHz band at +150 dB, 99.99% of samples came back flat, which is
not a loud signal but a square wave. A waveshaper -- tanh and friends -- only
rounds the corners of that, and a signal four hundred times over the ceiling
is still a square wave with rounded corners.

So this reduces the gain instead. The waveform keeps its shape; only its
level moves. A peak 52 dB over the ceiling costs 52 dB of gain reduction for
as long as it lasts and nothing at all afterwards.

Look-ahead is what makes that work on a bat call. The gain has to be down
BEFORE the peak arrives, or the first samples of every call clip while the
limiter is still reacting -- and a bat call is nothing but its first samples.
The audio is therefore delayed by the look-ahead, and the gain is worked out
from a window that reaches that far into the future.

    x --> delay(L) --------------------> * --> out
      \                                  ^
       -> |x| -> needed gain -> min over |
          per segment          the next L

The running minimum is taken over segments rather than samples: the exact
sample a peak lands on does not matter, only that the gain is already down
when it gets there, and a segment at a time is ~500 times less arithmetic.
Between segments the gain is a straight line, so nothing steps.

Release is slow and deliberate. Recovering fast after every peak is what
makes a limiter breathe audibly on a signal full of clicks.
"""

from __future__ import annotations

import numpy as np

# Just under full scale. The resampler downstream interpolates, and an
# interpolated sample can sit slightly above the ones either side of it, so
# the last fraction of a dB is left alone.
DEFAULT_CEILING = 0.95

# 2 ms at 250 kHz. Long enough to get ahead of a call's attack, short enough
# not to matter next to the 6 ms the noise reduction already costs.
DEFAULT_LOOKAHEAD_S = 0.002

# How quickly the gain comes back once the peak has gone.
DEFAULT_RELEASE_S = 0.050

# Gain is decided once per segment and interpolated between. 64 samples is
# 0.26 ms at 250 kHz -- finer than anything audible as a level change.
DEFAULT_SEGMENT = 64


class SoftLimiter:
    """Peak limiter with look-ahead. n samples in, n samples out, delayed."""

    def __init__(
        self,
        fs: float,
        ceiling: float = DEFAULT_CEILING,
        lookahead_s: float = DEFAULT_LOOKAHEAD_S,
        release_s: float = DEFAULT_RELEASE_S,
        segment: int = DEFAULT_SEGMENT,
    ):
        if fs <= 0:
            raise ValueError("fs must be > 0, got %r" % fs)
        if not 0.0 < ceiling <= 1.0:
            raise ValueError("ceiling must be in (0, 1], got %r" % ceiling)
        if segment < 1:
            raise ValueError("segment must be >= 1, got %r" % segment)

        self.fs = float(fs)
        self.ceiling = float(ceiling)
        self.segment = int(segment)
        self.lookahead_segments = max(1, int(round(lookahead_s * fs / self.segment)))
        # One more segment than the look-ahead is held back, because the gain
        # for a segment has to be settled at BOTH its ends before it can be
        # emitted -- see _emit_ready_segments.
        self._hold = self.lookahead_segments + 1
        self.release_s = float(release_s)

        self._release_alpha = (
            1.0 if release_s <= 0
            else 1.0 - float(np.exp(-(self.segment / self.fs) / release_s)))

        self.reset()

    @property
    def latency_samples(self) -> int:
        return self._hold * self.segment

    @property
    def latency_s(self) -> float:
        return self.latency_samples / self.fs

    @property
    def reduction_db(self) -> float:
        """How far the limiter is currently pulling the level down."""
        return 20.0 * float(np.log10(max(self._gain, 1e-12)))

    def reset(self) -> None:
        self._pending = np.zeros(0)      # audio not yet emitted
        self._needed = []                # per-segment gain, oldest first
        self._gain = 1.0
        self._started = False
        # Primed with the delay, so callers get back as many samples as they
        # hand over from the very first block.
        self._ready = np.zeros(self.latency_samples)

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x

        self._pending = np.concatenate((self._pending, x))
        self._measure_new_segments()
        self._emit_ready_segments()

        if self._ready.size < x.size:
            # Only while the look-ahead is still filling.
            self._ready = np.concatenate(
                (np.zeros(x.size - self._ready.size), self._ready))

        out, self._ready = self._ready[:x.size], self._ready[x.size:]
        return out

    def _measure_new_segments(self) -> None:
        measured = len(self._needed) * self.segment
        complete = (self._pending.size // self.segment) * self.segment
        if complete <= measured:
            return

        fresh = self._pending[measured:complete].reshape(-1, self.segment)
        peaks = np.max(np.abs(fresh), axis=1)
        # 1 where the segment already fits, below 1 by exactly as much as it
        # overshoots. Never above 1: this only ever turns things down.
        self._needed.extend(np.minimum(1.0, self.ceiling / np.maximum(peaks, 1e-20)))

    def _emit_ready_segments(self) -> None:
        """Emit segments whose gain is settled at both ends.

        The gain during a segment is a straight line between two values, and
        BOTH have to be low enough for the whole segment, or a peak early in
        it slips out while the line is still on its way down. So the value at
        the start is the minimum needed across this segment and the look-ahead
        after it, and the value at the end is that window with one more
        segment of look-ahead on the end. Both include this segment's own
        requirement, so every point on the line between them is under it.
        """
        while len(self._needed) > self._hold:
            starts_at = min(self._needed[:self._hold])
            # Both windows include THIS segment's own requirement. Leaving it
            # out of the second one lets the gain start recovering while the
            # loud part is still playing, and a burst that ends mid-segment
            # then punches through -- measured at 2.4x over the ceiling
            # before this line included index 0.
            ends_at = min(self._needed[:self._hold + 1])

            start = min(self._gain, starts_at) if self._started else starts_at
            self._started = True

            if ends_at < start:
                # Straight down. Safe because the look-ahead means the peak
                # that demanded it has not arrived yet.
                end = ends_at
            else:
                # Back up slowly, and never past what is actually needed.
                end = min(ends_at, start + self._release_alpha * (ends_at - start))

            block = self._pending[:self.segment]
            self._ready = np.concatenate(
                (self._ready, block * np.linspace(start, end, self.segment)))

            self._gain = end
            self._pending = self._pending[self.segment:]
            self._needed.pop(0)
