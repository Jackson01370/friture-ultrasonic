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

"""Automatic gain control: bring a quiet band up to a usable level.

Ported from ultraScan's AGCGain, defaults and all -- the tuning below was
learned on this same microphone and there is nothing to rediscover.

Per block:
  1. measure the block's RMS
  2. want = target_rms / level, capped at max_gain
  3. move the running gain toward `want` through a one-pole filter whose time
     constant is ASYMMETRIC: fast when the gain must come DOWN (a loud sound
     has arrived and must not reach the speaker as a blast), slow when it must
     go UP (chasing every dip is what makes an AGC pump)
  4. apply it as a per-sample ramp from the old gain to the new one

Step 4 is what keeps it silent between blocks. A single gain per block would
step the waveform at every boundary, which at 122 blocks a second is not a
level change but a buzz.

What it cannot do
-----------------
Gain lifts the noise floor along with the signal. This helps a signal that is
merely quiet; it does nothing for one buried *under* the noise, which needs
gating or spectral subtraction rather than amplification. During silence the
gain climbs to its ceiling and the microphone's own hiss rises to the target
-- that is the mechanism working, not failing.

Which is why the ceiling is 12x rather than something larger: ultraScan
shipped +40 dB first and found on real hardware that a quiet band's noise
floor came up as a roar that buried speech. +21.6 dB still lifts anything
above about 0.017 RMS to the target, and leaves the floor where it belongs.
"""

from __future__ import annotations

import numpy as np

# Comfortable output level for float audio in [-1, 1].
DEFAULT_TARGET_RMS = 0.2

# Fast down, slow up. See the module docstring.
DEFAULT_ATTACK_S = 0.010
DEFAULT_RELEASE_S = 0.300

# 12x is +21.6 dB. Do not raise this without listening to a quiet band first.
DEFAULT_MAX_GAIN = 12.0


class Agc:
    """Steers a block-streaming signal toward a target level."""

    def __init__(
        self,
        fs: float,
        target_rms: float = DEFAULT_TARGET_RMS,
        attack_s: float = DEFAULT_ATTACK_S,
        release_s: float = DEFAULT_RELEASE_S,
        max_gain: float = DEFAULT_MAX_GAIN,
        eps: float = 1e-6,
    ):
        if fs <= 0:
            raise ValueError("fs must be > 0, got %r" % fs)
        if target_rms <= 0:
            raise ValueError("target_rms must be > 0, got %r" % target_rms)
        if attack_s < 0 or release_s < 0:
            raise ValueError("attack_s / release_s must be >= 0")
        if max_gain <= 0:
            raise ValueError("max_gain must be > 0, got %r" % max_gain)

        self.fs = float(fs)
        self.target_rms = float(target_rms)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)
        self.max_gain = float(max_gain)
        self.eps = float(eps)

        # Starts at unity so the first samples are not slammed; it converges
        # over the first attack or release.
        self._gain = 1.0

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def gain_db(self) -> float:
        return 20.0 * float(np.log10(max(self._gain, 1e-12)))

    def reset(self) -> None:
        """Back to unity. For a new stream, not for a new band.

        Retuning deliberately does not reset: the band is retuned continuously
        while it is dragged, and dropping the gain to unity on every mouse move
        would be its own kind of pumping. The AGC re-adapts within a release.
        """
        self._gain = 1.0

    def process(self, audio: np.ndarray, adapt: bool = True) -> np.ndarray:
        """Apply the gain. With adapt=False, hold it where it is.

        The gate passes adapt=False while it is shut. Otherwise the AGC spends
        every silence measuring the ducked signal, winds all the way up to its
        ceiling, and blasts the first call that opens the gate.
        """
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        n = x.size
        if n == 0:
            return x

        if adapt:
            level = max(float(np.sqrt(np.mean(x ** 2))), self.eps)
            wanted = min(self.target_rms / level, self.max_gain)
        else:
            wanted = self._gain

        # The coefficient uses the real block length, so the time constants
        # stay honest whatever size the blocks arrive in.
        tau = self.attack_s if wanted < self._gain else self.release_s
        alpha = 1.0 if tau <= 0.0 else 1.0 - float(np.exp(-(n / self.fs) / tau))
        new_gain = self._gain + alpha * (wanted - self._gain)

        ramp = np.linspace(self._gain, new_gain, n)
        self._gain = float(new_gain)
        return x * ramp
