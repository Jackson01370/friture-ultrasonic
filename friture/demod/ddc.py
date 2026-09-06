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

"""Complex down-converter: the selected band as a complex baseband stream.

Cuts [f_lo, f_lo + width] out of the real capture, slides it down so that
f_lo sits at 0 Hz, and decimates. What comes out is COMPLEX and stays that
way: the demodulators in friture.demod.detectors need the analytic signal
(instantaneous frequency is the phase advance between samples, and that is
only defined on a complex signal).

This is ultraScan's ComplexDdc (dsp/ddc_display.py) on numpy alone -- direct
convolution with the filter history carried, which is bit-for-bit what
scipy's lfilter with zi does for an FIR. The listen path in
friture.listen.band_dsp is a different DDC for a different job: it goes back
to real audio at the capture rate, and is built to be retuned mid-stream
without a click. This one is built to be simple and exact; a retune is a
rebuild, since a change of band is a different signal and the decoder
downstream has to start over anyway.

FREQUENCY REFERENCE -- baseband 0 Hz is f_lo, the band's LOWER EDGE, not its
centre. The demodulators report frequencies relative to that; add f_lo to get
back to the plot's axis.

ONE-SIDED -- the lowpass is a real lowpass at cutoff/2 modulated up by
cutoff/2, so its passband is [0, cutoff] and it rejects negative baseband
frequencies. That is what keeps whatever sits just BELOW f_lo out of the
band: with a symmetric lowpass it would land at negative frequencies and
alias into the tone histogram and the carrier estimate.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def lowpass_taps(cutoff_hz: float, fs: float, n_taps: int) -> np.ndarray:
    """Hamming-windowed sinc lowpass, unity gain at DC, linear phase.

    The same design as friture.listen.band_dsp uses for the listen filter.
    """
    k = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff_hz * k / fs) * np.hamming(n_taps)
    return h / h.sum()


class ComplexDdc:
    """Real stream (fs_in) -> complex baseband of [f_lo, f_lo+width] (fs_in/decim)."""

    def __init__(self, f_lo: float, width: float, fs_in: float, decim: int,
                 n_taps: int | None = None) -> None:
        if fs_in <= 0:
            raise ValueError(f"fs_in must be > 0, got {fs_in!r}")
        if decim < 1:
            raise ValueError(f"decim must be >= 1, got {decim!r}")
        if width <= 0:
            raise ValueError(f"width must be > 0, got {width!r}")
        if not 0.0 <= f_lo < fs_in / 2.0:
            raise ValueError(f"f_lo must be in [0, fs_in/2), got {f_lo!r}")
        fs_out = float(fs_in) / decim
        if f_lo + min(float(width), fs_out / 2.0) > fs_in / 2.0 + 1e-6:
            raise ValueError(
                f"[f_lo, f_lo+cutoff] exceeds Nyquist {fs_in / 2.0:.0f} Hz "
                f"(f_lo={f_lo!r}, width={width!r})")

        self.f_lo = float(f_lo)
        self.width = float(width)
        self.fs_in = float(fs_in)
        self.decim = int(decim)
        # Cutoff = the band width, capped at the decimated Nyquist: nothing
        # above fs_out/2 survives the decimation anyway.
        self.cutoff = min(float(width), fs_out / 2.0)

        # Anti-alias taps scale with decim so the transition is narrow enough
        # that content above fs_out/2 is gone before the decimation folds it
        # back into the band. Odd count -> linear phase, integer delay.
        if n_taps is None:
            n_taps = 24 * self.decim + 1
        self.n_taps = int(n_taps) | 1

        # One-sided complex FIR, see the module docstring. Gain 2 puts back
        # the half that keeping one of a real tone's two lines took away.
        half = self.cutoff / 2.0
        b = lowpass_taps(half, self.fs_in, self.n_taps)
        k = np.arange(self.n_taps)
        self._taps = 2.0 * b * np.exp(1j * _TWO_PI * half * k / self.fs_in)

        self._dphi = _TWO_PI * self.f_lo / self.fs_in
        self.reset()

    @property
    def fs_out(self) -> float:
        return self.fs_in / self.decim

    @property
    def group_delay_s(self) -> float:
        return (self.n_taps - 1) / 2.0 / self.fs_in

    def reset(self) -> None:
        """Drop carried state: filter history, mixer phase, decimation phase."""
        self._history = np.zeros(self.n_taps - 1, dtype=np.complex128)
        self._phase = 0.0
        self._offset = 0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Real block (fs_in) -> complex64 block (fs_out). State carries across calls."""
        s = np.asarray(block, dtype=np.float64).reshape(-1)
        if s.size == 0:
            return np.empty(0, dtype=np.complex64)
        n = s.size

        # 1. complex NCO + mix; the phase continues where the last block ended
        ph = self._phase + self._dphi * np.arange(n)
        x = s * np.exp(-1j * ph)
        self._phase = float((self._phase + self._dphi * n) % _TWO_PI)

        # 2. one-sided lowpass, history carried -> y[i] = sum_k h[k] x[i-k]
        stream = np.concatenate((self._history, x))
        y = np.convolve(stream, self._taps, mode="valid")
        self._history = stream[-(self.n_taps - 1):]

        # 3. integer decimation with a carried pick phase, so the kept
        #    samples sit on one grid however the stream is chopped
        picks = y[self._offset::self.decim]
        self._offset = (self._offset - n) % self.decim
        return picks.astype(np.complex64)
