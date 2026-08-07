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

"""What has always been there, per frequency bin -- and what to do about it.

Measured in this room at 250 kHz: a near-pure 25 kHz tone standing 12 dB over
its surroundings (an ultrasonic pest repeller), a second at 96 kHz, and an
otherwise smooth microphone floor that drifts less than 0.1 dB over four
seconds. Perfectly stationary noise, and a bat call is anything but -- so the
rule that separates them is simply "how far is this bin above where it
usually sits".

That one rule handles both problems with no list of frequencies to maintain:
a steady tone is absorbed into its own background and attenuated like any
other constant, and if the repeller is switched off, or a new one appears,
the estimate follows without being told.

Used in two places from here, and they want different things from it:

  the audio    wants the steady part quieter, so it takes `process`, which
               returns a gain to attenuate by (denoise.py)

  the display  wants the steady part *flat*, which is not the same thing.
               Attenuating a steady tone and the steady floor around it by
               the same amount leaves the tone standing exactly as far above
               its surroundings as before -- measured here at +6.7 dB either
               way, so the interfering line stayed just as visible. What
               removes it from a picture is dividing each bin by its own
               background, which puts everything steady at one level and
               leaves only what is new. That is `flatten_gain`.

Two things keep it from eating what it is meant to preserve:

  the update slows right down in a bin that is currently loud, so a call
  lasting a second is not gradually adopted as background

  the gain is smoothed across neighbouring bins and across time before it is
  applied. Subtracting bin by bin leaves isolated survivors flickering in and
  out -- "musical noise", which is more distracting than the hiss it replaced
"""

from __future__ import annotations

import numpy as np

# How fast the background follows the room. Seconds.
DEFAULT_TAU_S = 1.5

# ...and how fast it follows while the bin is loud. Much slower, so a long
# call is not absorbed into the estimate that is supposed to remove noise.
DEFAULT_ACTIVE_TAU_S = 60.0

# A bin more than this far above its background counts as loud, for the
# purpose of slowing the update above.
DEFAULT_ACTIVE_MARGIN_DB = 6.0

# Where attenuation starts and where it stops: a bin at or below its
# background is attenuated fully, one this far above it passes untouched.
DEFAULT_OPEN_DB = 3.0
DEFAULT_KNEE_DB = 9.0

# Never attenuate beyond this. Digital silence between surviving bins is what
# makes musical noise obvious; leaving a floor keeps the result sounding like
# a quieter version of the same room.
DEFAULT_FLOOR_DB = -18.0

# Gain smoothing: bins either side, and how fast the gain may rise and fall.
DEFAULT_SMOOTH_BINS = 3
DEFAULT_GAIN_ATTACK_S = 0.005
DEFAULT_GAIN_RELEASE_S = 0.060


class SpectralBackground:
    """Tracks a per-bin background and turns it into a per-bin gain."""

    def __init__(
        self,
        n_bins: int,
        frame_rate: float,
        tau_s: float = DEFAULT_TAU_S,
        active_tau_s: float = DEFAULT_ACTIVE_TAU_S,
        active_margin_db: float = DEFAULT_ACTIVE_MARGIN_DB,
        open_db: float = DEFAULT_OPEN_DB,
        knee_db: float = DEFAULT_KNEE_DB,
        floor_db: float = DEFAULT_FLOOR_DB,
        smooth_bins: int = DEFAULT_SMOOTH_BINS,
        gain_attack_s: float = DEFAULT_GAIN_ATTACK_S,
        gain_release_s: float = DEFAULT_GAIN_RELEASE_S,
    ):
        if n_bins <= 0:
            raise ValueError("n_bins must be > 0, got %r" % n_bins)
        if frame_rate <= 0:
            raise ValueError("frame_rate must be > 0, got %r" % frame_rate)
        if knee_db <= 0:
            raise ValueError("knee_db must be > 0, got %r" % knee_db)

        self.n_bins = int(n_bins)
        self.open_db = float(open_db)
        self.knee_db = float(knee_db)
        self.active_margin_db = float(active_margin_db)
        self.floor = float(10.0 ** (floor_db / 20.0))
        self.smooth_bins = int(smooth_bins)

        # Per-frame coefficients from the time constants.
        self._alpha_quiet = _one_pole_alpha(tau_s, frame_rate)
        self._alpha_loud = _one_pole_alpha(active_tau_s, frame_rate)
        self._alpha_gain_up = _one_pole_alpha(gain_attack_s, frame_rate)
        self._alpha_gain_down = _one_pole_alpha(gain_release_s, frame_rate)

        self._background = None
        self._gain = np.ones(self.n_bins)

    @property
    def background(self):
        """The current estimate, or None until the first frame."""
        return self._background

    def reset(self) -> None:
        self._background = None
        self._gain = np.ones(self.n_bins)

    def flatten_gain(self) -> np.ndarray:
        """Gain that levels the background out, for a display.

        Each bin is scaled so its background lands on the median background,
        which erases both a steady interfering line and the overall tilt of
        the noise floor while leaving anything above background sticking out
        by exactly as much as before. Returns ones until a background exists.

        Bounded either side: a bin whose background is near zero would
        otherwise be multiplied without limit and fill the display with
        amplified nothing.
        """
        if self._background is None:
            return np.ones(self.n_bins)
        reference = float(np.median(self._background))
        if reference <= 0.0:
            return np.ones(self.n_bins)
        return np.clip(reference / (self._background + 1e-20), 0.01, 100.0)

    def process(self, magnitude: np.ndarray) -> np.ndarray:
        """One frame of magnitudes in, one frame of gains in [floor, 1] out."""
        magnitude = np.asarray(magnitude, dtype=np.float64).reshape(-1)
        if magnitude.size != self.n_bins:
            raise ValueError("expected %d bins, got %d" % (self.n_bins, magnitude.size))

        if self._background is None:
            # Start from the first frame rather than from zero: otherwise the
            # first second of audio is compared against nothing and passes
            # whatever is in it straight through.
            self._background = magnitude.copy()
            return np.ones(self.n_bins)

        excess_db = 20.0 * np.log10((magnitude + 1e-20) / (self._background + 1e-20))

        # Slow the update where the bin is loud -- see the module docstring.
        alpha = np.where(excess_db > self.active_margin_db,
                         self._alpha_loud, self._alpha_quiet)
        self._background += alpha * (magnitude - self._background)

        wanted = np.clip((excess_db - self.open_db) / self.knee_db, 0.0, 1.0)
        wanted = self.floor + (1.0 - self.floor) * wanted

        if self.smooth_bins > 0:
            wanted = _smooth_across_bins(wanted, self.smooth_bins)

        # Rising quickly and falling slowly: a call must not be clipped at its
        # start, and the gain must not chatter as a bin hovers at the knee.
        rising = wanted > self._gain
        alpha_gain = np.where(rising, self._alpha_gain_up, self._alpha_gain_down)
        self._gain += alpha_gain * (wanted - self._gain)

        return self._gain.copy()


def _one_pole_alpha(tau_s: float, frame_rate: float) -> float:
    if tau_s <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-(1.0 / frame_rate) / tau_s))


def _smooth_across_bins(gain: np.ndarray, half_width: int) -> np.ndarray:
    """Moving average over 2*half_width+1 bins, edges held rather than zeroed."""
    width = 2 * half_width + 1
    padded = np.pad(gain, half_width, mode="edge")
    kernel = np.ones(width) / width
    return np.convolve(padded, kernel, mode="valid")
