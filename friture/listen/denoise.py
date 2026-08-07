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

"""Spectral noise reduction for the audio path.

SpectralBackground decides what to keep; this puts a signal through it and
gets one back. Short-time Fourier transform in, per-bin gain applied to the
complex spectrum, overlap-add out.

Applying the gain to the complex spectrum rather than rebuilding from
magnitudes is what keeps the phase: the original phase is exactly right for
the part being kept, and inventing one is what makes denoisers sound hollow.

Latency is frame minus hop: 6.1 ms at 250 kHz with a 2048-point frame. That
is the price of asking a frequency-domain question at all -- the transform
cannot say what is in a frame until it has most of the frame.

The window is a periodic Hann at 75% overlap, applied on the way in and again
on the way out. Four overlapping Hann-squared frames sum to a constant, and
that constant is measured from the window in __init__ rather than quoted from
a table -- if the window or the overlap is ever changed, the normalisation
follows instead of quietly becoming wrong.
"""

from __future__ import annotations

import numpy as np

from friture.listen.spectral_background import SpectralBackground

# 2048 at 250 kHz gives 122 Hz bins and 6.1 ms of latency. Enough resolution
# to isolate a pure interfering tone without smearing a call that lasts a few
# milliseconds across several frames.
DEFAULT_FRAME = 2048
OVERLAP = 4  # hop = frame / 4


class SpectralDenoiser:
    """Block-streaming spectral noise reduction. n samples in, n samples out.

    Output runs frame - hop behind the input: the samples returned for a
    block are the ones that entered that long ago. Inherent, not an
    implementation detail to be tidied away.
    """

    def __init__(self, fs: float, frame: int = DEFAULT_FRAME, **background_kwargs):
        if fs <= 0:
            raise ValueError("fs must be > 0, got %r" % fs)
        if frame < OVERLAP or frame % OVERLAP:
            raise ValueError("frame must be a multiple of %d, got %r" % (OVERLAP, frame))

        self.fs = float(fs)
        self.frame = int(frame)
        self.hop = self.frame // OVERLAP
        self._window = np.hanning(self.frame + 1)[:-1]  # periodic, so it sums flat

        # What the overlapping windows add up to, measured from the window
        # itself. Constant across the hop when the window really is COLA.
        squared = self._window ** 2
        overlapped = sum(squared[k * self.hop:(k + 1) * self.hop] for k in range(OVERLAP))
        self._cola = float(np.mean(overlapped))
        if not np.allclose(overlapped, self._cola, rtol=1e-6):
            raise ValueError("window and overlap do not satisfy COLA; output would ripple")

        self.background = SpectralBackground(
            n_bins=self.frame // 2 + 1,
            frame_rate=self.fs / self.hop,
            **background_kwargs)

        self.reset()

    def reset(self) -> None:
        self.background.reset()
        # Samples waiting to fill the next frame.
        self._pending = np.zeros(0)
        # Where finished frames are summed. Only its first hop is complete at
        # any moment; the rest is still waiting for later frames.
        self._overlap = np.zeros(self.frame)
        # Output that is finished but not yet asked for, primed with the
        # silence the delay amounts to. Priming here rather than padding on
        # the first call makes the delay exactly frame - hop whatever size
        # blocks the caller happens to use.
        self._ready = np.zeros(self.frame - self.hop)

    @property
    def latency_s(self) -> float:
        """How far the output runs behind the input.

        frame - hop, not frame: the first hop of a frame is complete as soon
        as that frame is transformed, so the wait is for the rest of it.
        """
        return (self.frame - self.hop) / self.fs

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x

        self._pending = np.concatenate((self._pending, x))
        while self._pending.size >= self.frame:
            self._consume_one_frame()

        if self._ready.size < x.size:
            # Happens only while the first frame is still filling: pad with
            # the silence that the delay amounts to, rather than returning a
            # short block, so callers can rely on n in / n out.
            self._ready = np.concatenate(
                (np.zeros(x.size - self._ready.size), self._ready))

        out, self._ready = self._ready[:x.size], self._ready[x.size:]
        return out

    def _consume_one_frame(self) -> None:
        frame = self._pending[:self.frame] * self._window
        self._pending = self._pending[self.hop:]

        spectrum = np.fft.rfft(frame)
        gain = self.background.process(np.abs(spectrum))
        # Applied to the complex spectrum, so the original phase is kept.
        # Rebuilding from magnitudes alone means inventing a phase, which is
        # what makes a denoiser sound hollow.
        cleaned = np.fft.irfft(spectrum * gain, n=self.frame) * self._window

        self._overlap += cleaned
        # Only the first hop is done: every later sample still has frames to
        # come. Emit it, slide the accumulator along, and open up fresh space.
        self._ready = np.concatenate((self._ready, self._overlap[:self.hop] / self._cola))
        self._overlap = np.concatenate(
            (self._overlap[self.hop:], np.zeros(self.hop)))
