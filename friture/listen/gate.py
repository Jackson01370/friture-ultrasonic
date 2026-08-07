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

"""Silence between the calls.

A bat is not a continuous sound. It is a few milliseconds of shriek every
tenth of a second, and the rest is the microphone's own hiss -- which the AGC
will happily lift to a comfortable listening level, because that is what an
AGC does. This closes the gap.

The threshold is relative, not absolute: the gate tracks the quietest thing
it has heard lately and opens for anything that stands far enough above it.
Set in dBFS instead, it would need retuning for every gain setting, every
band and every room.

Which means it needs a moment to learn. Switched on during a loud passage the
floor starts high and the gate stays shut until the level drops and drags it
down -- a second or so, once. Switched on during ordinary room noise, which
is the normal case, it is right immediately.

Two details decide whether it sounds like a bat detector or like a fault:

  hold   a call's tail is quieter than its peak, and a gate that shuts the
         instant the level dips clips every call short and chatters on the
         ones near the threshold. Once open it stays open for hold_s.

  ramp   the gain moves as a per-sample ramp across the block, never a step
         between blocks -- the same rule as everywhere else in this chain.
         Opening is fast enough not to swallow the front of a call, closing
         slow enough not to click.

What it is not: this improves nothing about the signal-to-noise ratio inside
a call. It removes the noise you hear *between* calls, which is most of the
time and most of the fatigue.
"""

from __future__ import annotations

import numpy as np

# How far above the tracked floor the level must rise for the gate to open,
# and how far it must fall to close again. The gap between them is what keeps
# a signal hovering at the threshold from chattering.
DEFAULT_OPEN_DB = 12.0
DEFAULT_CLOSE_DB = 6.0

# How quickly the floor estimate follows. Slow, and biased towards falling:
# the floor is the quiet part, so it should be dragged down by silence and
# barely moved by a loud passage.
DEFAULT_FLOOR_FALL_S = 0.5
DEFAULT_FLOOR_RISE_S = 8.0

# Once open, stay open this long past the last sample above the threshold.
DEFAULT_HOLD_S = 0.150

# Gain movement. Fast open, gentle close.
DEFAULT_ATTACK_S = 0.002
DEFAULT_RELEASE_S = 0.050

# What "closed" means. Not zero: a gate that slams to digital silence is more
# noticeable than one that ducks, and -40 dB is already inaudible next to the
# call that opened it.
DEFAULT_CLOSED_DB = -40.0


class NoiseGate:
    """Passes the loud parts, ducks the rest, and says which it is doing."""

    def __init__(
        self,
        fs: float,
        open_db: float = DEFAULT_OPEN_DB,
        close_db: float = DEFAULT_CLOSE_DB,
        floor_fall_s: float = DEFAULT_FLOOR_FALL_S,
        floor_rise_s: float = DEFAULT_FLOOR_RISE_S,
        hold_s: float = DEFAULT_HOLD_S,
        attack_s: float = DEFAULT_ATTACK_S,
        release_s: float = DEFAULT_RELEASE_S,
        closed_db: float = DEFAULT_CLOSED_DB,
        eps: float = 1e-9,
    ):
        if fs <= 0:
            raise ValueError("fs must be > 0, got %r" % fs)
        if close_db > open_db:
            raise ValueError("close_db must not exceed open_db, or the gate chatters")

        self.fs = float(fs)
        self.open_ratio = float(10.0 ** (open_db / 20.0))
        self.close_ratio = float(10.0 ** (close_db / 20.0))
        self.floor_fall_s = float(floor_fall_s)
        self.floor_rise_s = float(floor_rise_s)
        self.hold_s = float(hold_s)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)
        self.closed = float(10.0 ** (closed_db / 20.0))
        self.eps = float(eps)

        self.reset()

    def reset(self) -> None:
        self._floor = None
        self._gain = 1.0
        self._is_open = False
        self._hold_left_s = 0.0

    @property
    def is_open(self) -> bool:
        """Whether the gate is currently passing audio.

        The AGC asks: a gain that keeps adapting while the gate is shut winds
        up on the noise floor and then blasts the first call that opens it.
        """
        return self._is_open

    @property
    def gain(self) -> float:
        return self._gain

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        n = x.size
        if n == 0:
            return x

        level = max(float(np.sqrt(np.mean(x ** 2))), self.eps)
        block_s = n / self.fs

        if self._floor is None:
            self._floor = level
        else:
            tau = self.floor_fall_s if level < self._floor else self.floor_rise_s
            alpha = 1.0 if tau <= 0 else 1.0 - float(np.exp(-block_s / tau))
            self._floor += alpha * (level - self._floor)
        self._floor = max(self._floor, self.eps)

        # Hysteresis: a higher bar to open than to stay open.
        if self._is_open:
            still_loud = level > self._floor * self.close_ratio
        else:
            still_loud = level > self._floor * self.open_ratio

        if still_loud:
            self._hold_left_s = self.hold_s
            self._is_open = True
        else:
            self._hold_left_s = max(0.0, self._hold_left_s - block_s)
            self._is_open = self._hold_left_s > 0.0

        wanted = 1.0 if self._is_open else self.closed
        tau = self.attack_s if wanted > self._gain else self.release_s
        alpha = 1.0 if tau <= 0 else 1.0 - float(np.exp(-block_s / tau))
        new_gain = self._gain + alpha * (wanted - self._gain)

        ramp = np.linspace(self._gain, new_gain, n)
        self._gain = float(new_gain)
        return x * ramp
