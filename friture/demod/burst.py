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

"""ON/OFF extraction from an envelope -- OOK / ASK demodulation.

Ported from ultraScan's dsp/burst.py (M23), unchanged in behaviour. Slice the
AM envelope at a threshold and the keying falls out.

ADAPTIVE FLOOR: the threshold rides on a one-pole EMA of the envelope rather
than sitting at a fixed level, so it works whether the room is quiet or
humming.

    burst OFF : floor += a_off * (env[n] - floor)      tau_off_s = 0.05  (fast)
    burst ON  : floor += a_on  * (env[n] - floor)      tau_on_s  = 2.0   (slow)
                ... but ONLY WHEN IT WOULD LOWER THE FLOOR.

The floor never rises inside a burst: the pulse is the loudest thing in the
block, so letting the floor chase it would raise the OFF threshold until the
detector cut its own burst in half.

HYSTERESIS + DEBOUNCE, so a wobbling envelope produces one burst and not forty:

    ON  when env[n] >= floor * ratio_on    (default 6.0, about +15.6 dB)
    OFF when env[n] <  floor * ratio_off   (default 3.0, about +9.5 dB)
    an OFF shorter than min_off_ms is bridged (the burst stays ONE burst)
    an ON  shorter than min_on_ms  is discarded (a noise spike is not a pulse)

Block streaming: everything that matters across a block edge (floor, state
machine, burst start, running peak/sum, the absolute sample counter, the
pending OFF run) is carried, so splitting the stream cannot split a burst.

The per-sample ON/OFF decision is kept as ``last_state``: that is the trace
the symbol decoder reads. A bare ``env > floor * ratio`` was tried in
ultraScan first and does not survive a keyed carrier -- the floor chases the
OFF level down between symbols, the ratio degenerates, and the trace reads
either all-ON or a rate that is neither the baud nor a harmonic of it
(measured there: 452 Bd for a 600 Bd signal). Using the state machine's own
decision means the bits cannot disagree with the burst list about what was
keyed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Burst:
    """One detected ON pulse."""

    start_sample: int    # absolute sample index, counted from the last reset()
    end_sample: int      # absolute sample index, EXCLUSIVE
    duration_s: float
    peak_amp: float      # max envelope inside [start, end)
    mean_amp: float      # mean envelope inside [start, end), bridged dips included


class BurstDetector:
    """Envelope -> ON/OFF bursts (OOK / ASK demodulation)."""

    def __init__(
        self,
        fs_bb: float,
        ratio_on: float = 6.0,
        ratio_off: float = 3.0,
        tau_off_s: float = 0.05,
        tau_on_s: float = 2.0,
        min_on_ms: float = 0.2,
        min_off_ms: float = 0.2,
    ) -> None:
        """
        fs_bb      : envelope sample rate [Hz] (> 0)
        ratio_on   : env/floor needed to OPEN a burst (must exceed ratio_off)
        ratio_off  : env/floor below which a burst starts CLOSING
        tau_off_s  : floor EMA time constant while no burst is open (fast)
        tau_on_s   : floor EMA time constant while a burst is open, downward only
        min_on_ms  : bursts shorter than this are discarded
        min_off_ms : gaps shorter than this are bridged (kept inside one burst)
        """
        if fs_bb <= 0:
            raise ValueError(f"fs_bb must be > 0, got {fs_bb!r}")
        if ratio_off <= 0:
            raise ValueError(f"ratio_off must be > 0, got {ratio_off!r}")
        if ratio_on <= ratio_off:
            raise ValueError(
                f"ratio_on must be > ratio_off (hysteresis), got "
                f"ratio_on={ratio_on!r} ratio_off={ratio_off!r}")
        if tau_off_s < 0 or tau_on_s < 0:
            raise ValueError("tau_off_s / tau_on_s must be >= 0")
        if min_on_ms < 0 or min_off_ms < 0:
            raise ValueError("min_on_ms / min_off_ms must be >= 0")

        self.fs_bb = float(fs_bb)
        self.ratio_on = float(ratio_on)
        self.ratio_off = float(ratio_off)
        self._a_off = self._alpha(tau_off_s)
        self._a_on = self._alpha(tau_on_s)
        self.min_on_samples = int(round(float(min_on_ms) * 1e-3 * self.fs_bb))
        # At least one OFF sample must be seen, otherwise nothing could ever close.
        self.min_off_samples = max(
            1, int(round(float(min_off_ms) * 1e-3 * self.fs_bb)))
        self.reset()

    def _alpha(self, tau_s: float) -> float:
        tau = float(tau_s)
        if tau <= 0.0:
            return 1.0                      # zero time constant = follow instantly
        return float(1.0 - np.exp(-1.0 / (tau * self.fs_bb)))

    @property
    def floor(self) -> float | None:
        """Current noise-floor estimate (None until the first non-empty block)."""
        return self._floor

    def reset(self) -> None:
        """Clear the floor, the state machine and the absolute sample counter."""
        self._floor: float | None = None
        self._on = False
        self._start = 0
        self._peak = 0.0          # accumulators for the burst being built
        self._sum = 0.0
        self._cnt = 0
        self._off_run = 0         # consecutive OFF samples seen inside the burst
        self._g_peak = 0.0        # ... and their accumulators, held aside until
        self._g_sum = 0.0         # we know whether the gap is bridged or final
        self._g_cnt = 0
        self._n = 0               # absolute samples consumed since reset()
        # The per-sample ON/OFF decision of the LAST process() call -- the
        # symbol decoder's input. See the module docstring.
        self.last_state = np.zeros(0, dtype=bool)

    def process(self, env: np.ndarray) -> list[Burst]:
        """Consume one envelope block; return the bursts that CLOSED in it.

        A burst still open at the end of the block is not returned; it is carried
        into the next call. An empty block returns ``[]`` and changes nothing at
        all (not even the floor initialisation, which waits for real samples).
        """
        e = np.asarray(env, dtype=np.float64).reshape(-1)
        if e.size == 0:
            self.last_state = np.zeros(0, dtype=bool)
            return []
        if self._floor is None:
            # First real samples define the floor. The median is used, not the
            # mean, so a block that already contains pulses still yields the
            # BACKGROUND as long as the duty cycle is under 50 %.
            self._floor = float(np.median(e))

        out: list[Burst] = []
        state = np.zeros(e.size, dtype=bool)
        floor = self._floor
        n0 = self._n
        on_level = self.ratio_on
        off_level = self.ratio_off

        for i in range(e.size):
            x = float(e[i])
            if not self._on:
                if x >= floor * on_level:
                    self._on = True
                    self._start = n0 + i
                    self._peak = x
                    self._sum = x
                    self._cnt = 1
                    self._off_run = 0
                    self._g_peak = self._g_sum = 0.0
                    self._g_cnt = 0
            elif x < floor * off_level:
                self._off_run += 1
                if x > self._g_peak:
                    self._g_peak = x
                self._g_sum += x
                self._g_cnt += 1
                if self._off_run >= self.min_off_samples:
                    # Long enough to be a real gap: the burst ended where the
                    # OFF run began, and the OFF run itself is not part of it.
                    burst = self._close(n0 + i + 1 - self._off_run)
                    if burst is not None:
                        out.append(burst)
                    self._on = False
                    self._off_run = 0
                    self._g_peak = self._g_sum = 0.0
                    self._g_cnt = 0
            else:
                if self._off_run:
                    # Too short to be a gap: fold the dip back into the burst so
                    # one pulse with a wobble stays ONE pulse.
                    if self._g_peak > self._peak:
                        self._peak = self._g_peak
                    self._sum += self._g_sum
                    self._cnt += self._g_cnt
                    self._off_run = 0
                    self._g_peak = self._g_sum = 0.0
                    self._g_cnt = 0
                if x > self._peak:
                    self._peak = x
                self._sum += x
                self._cnt += 1

            state[i] = self._on
            # Floor update, judged on the state AFTER this sample was classified.
            if self._on:
                if x < floor:                       # downward only -- see docstring
                    floor += self._a_on * (x - floor)
            else:
                floor += self._a_off * (x - floor)

        self._floor = floor
        self._n = n0 + e.size
        self.last_state = state
        return out

    def flush(self) -> list[Burst]:
        """End of stream: close a still-open burst and return it (or ``[]``)."""
        out: list[Burst] = []
        if self._on:
            burst = self._close(self._n - self._off_run)
            if burst is not None:
                out.append(burst)
            self._on = False
            self._off_run = 0
            self._g_peak = self._g_sum = 0.0
            self._g_cnt = 0
        return out

    def _close(self, end: int) -> Burst | None:
        """Finish the open burst at ``end`` (exclusive); None if it is too short."""
        length = end - self._start
        burst = None
        if length >= self.min_on_samples and length > 0 and self._cnt > 0:
            burst = Burst(
                start_sample=int(self._start),
                end_sample=int(end),
                duration_s=length / self.fs_bb,
                peak_amp=float(self._peak),
                mean_amp=float(self._sum / self._cnt),
            )
        self._peak = self._sum = 0.0
        self._cnt = 0
        return burst
