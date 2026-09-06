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

"""Find what stands out in the whole spectrum, and say how steady it is.

This is the survey that finding the 25 kHz repeller, the 16 kHz whine and
the 1 kHz-spaced spurs took by hand, made repeatable. Two ideas do the work.

★ A LINE IS MEASURED AGAINST ITS OWN NEIGHBOURHOOD, NOT AGAINST THE BAND ★

    The room's noise slopes: this microphone's own floor rises from -20 dB
    at 60-120 kHz to -5 dB at 22-26 kHz, and the room adds a hump at 3-4 kHz
    on top. Against a single wideband floor, everything in those humps
    "stands out" by 10-15 dB and the real lines are buried in the list --
    while 5.8 kHz, which sits in a valley, looks like nothing when it is
    merely quiet. What reads as a bump on the screen is what rises above the
    spectrum a few hundred hertz either side of it, so that is what is
    measured: the median over +-FLOOR_HZ, excluding the line's own bins.

★ STEADINESS SEPARATES A DEVICE FROM AN EVENT ★

    A line that is always there is a machine; one that comes and goes is
    something happening; one that appears only in the average is ripple. So
    every line carries the spread of its level across the individual
    windows, and the 10th-to-90th percentile is used rather than the full
    range -- one door slam otherwise makes a dead-steady line look like an
    event.

numpy only, no Qt: the same code runs in the dock and in
scripts/verify/band_anomalies.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurveyLine:
    """One line that stands out, and what is known about it."""

    frequency_hz: float
    excess_db: float        # over the median of its own neighbourhood
    level_db: float         # absolute, in the same units as the spectrum
    spread_db: float        # 10th-to-90th percentile of its level over time
    n_windows: int

    @property
    def steady(self) -> bool:
        """True for a device that is simply on, False for something that comes and goes."""
        return self.spread_db < 6.0

    def describe(self) -> str:
        return ("%.3f kHz  %+.1f dB over its neighbourhood, %s"
                % (self.frequency_hz / 1e3, self.excess_db,
                   "steady" if self.steady else "varies %.0f dB" % self.spread_db))


class SpectrumSurvey:
    """Accumulate a spectrum over a rolling history and report what stands out.

    Blocks in, at the capture rate; nothing else is needed. The windows are
    kept individually (only the last ``history`` of them) because the
    steadiness of a line cannot be recovered from an average.
    """

    # How far either side of a candidate the floor is measured. 250 Hz is
    # wide enough to average over the ripple of a 0.26 s window and narrow
    # enough to follow the microphone's own slope.
    FLOOR_HZ = 250.0
    # The line's own bins are excluded from its floor: a strong line spills
    # into its neighbours, and including them would hide it from itself.
    EXCLUDE_BINS = 3
    # The floor is evaluated on a subsampled grid and interpolated: an exact
    # median at every bin of a 32769-bin spectrum costs about 100 ms, which
    # is too much ten times a second, and the floor is smooth by
    # construction.
    FLOOR_STRIDE = 16

    def __init__(self, fs: float, nfft: int = 1 << 16, history: int = 32) -> None:
        if fs <= 0:
            raise ValueError(f"fs must be > 0, got {fs!r}")
        if nfft < 256 or (nfft & (nfft - 1)):
            raise ValueError(f"nfft must be a power of two >= 256, got {nfft!r}")
        if history < 2:
            raise ValueError(f"history must be >= 2, got {history!r}")
        self.fs = float(fs)
        self.nfft = int(nfft)
        self.history = int(history)
        self.hop = self.nfft // 2
        self._window = np.hanning(self.nfft)
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.fs)
        self.bin_hz = float(self.freqs[1])
        self.reset()

    def reset(self) -> None:
        self._carry = np.zeros(0, dtype=np.float64)
        # TWO ACCUMULATORS, because the two questions want different lengths.
        # The AVERAGE wants every window there has ever been: a line 4 dB out
        # of the noise needs minutes of averaging to become certain, and
        # nothing is gained by throwing the old windows away. STEADINESS
        # wants the recent ones, kept individually -- it cannot be recovered
        # from an average, and holding a 32769-bin window costs 128 kB, so
        # the count has to be bounded.
        self._sum = np.zeros(self.freqs.size, dtype=np.float64)
        self._windows: list[np.ndarray] = []
        self.n_windows_total = 0

    @property
    def ready(self) -> bool:
        """At least two windows: one cannot say anything about steadiness."""
        return len(self._windows) >= 2

    @property
    def seconds_held(self) -> float:
        """How much spectrum the AVERAGE is made of, in seconds."""
        return (self.n_windows_total + 1) * self.hop / self.fs if self.n_windows_total else 0.0

    @property
    def seconds_watched(self) -> float:
        """How much of it the steadiness figures are made of."""
        return (len(self._windows) + 1) * self.hop / self.fs if self._windows else 0.0

    def process(self, block: np.ndarray) -> int:
        """Feed one capture block; returns how many windows it completed."""
        b = np.asarray(block, dtype=np.float64).reshape(-1)
        if b.size == 0:
            return 0
        s = np.concatenate((self._carry, b))
        made = 0
        while s.size >= self.nfft:
            spec = np.abs(np.fft.rfft(s[:self.nfft] * self._window)) ** 2
            self._sum += spec
            self._windows.append(spec.astype(np.float32))
            made += 1
            self.n_windows_total += 1
            s = s[self.hop:]
        if len(self._windows) > self.history:
            self._windows = self._windows[-self.history:]
        self._carry = s
        return made

    def spectrum_db(self) -> np.ndarray:
        """The averaged spectrum over EVERY window since the last reset, in dB."""
        if self.n_windows_total == 0:
            return np.full(self.freqs.size, -300.0)
        return 10.0 * np.log10(np.maximum(self._sum / self.n_windows_total, 1e-30))

    def local_floor_db(self, db: np.ndarray | None = None) -> np.ndarray:
        """The median of the spectrum +-FLOOR_HZ around each bin, in dB.

        Evaluated every FLOOR_STRIDE bins and interpolated -- see
        FLOOR_STRIDE. The line's own EXCLUDE_BINS are dropped from each
        window so a tall line does not raise the floor it is measured
        against.
        """
        if db is None:
            db = self.spectrum_db()
        half = max(1, int(self.FLOOR_HZ / self.bin_hz))
        n = db.size
        grid = np.arange(0, n, self.FLOOR_STRIDE)
        floor = np.empty(grid.size)
        keep = np.ones(2 * half + 1, dtype=bool)
        keep[half - self.EXCLUDE_BINS:half + self.EXCLUDE_BINS + 1] = False
        for j, i in enumerate(grid):
            a, b = i - half, i + half + 1
            if a < 0 or b > n:
                a, b = max(a, 0), min(b, n)
                seg = db[a:b]
            else:
                seg = db[a:b][keep]
            floor[j] = np.median(seg)
        return np.interp(np.arange(n), grid, floor)

    def lines(self, f_from: float = 100.0, f_to: float | None = None,
              top: int = 12, min_separation_hz: float = 60.0,
              min_excess_db: float = 4.0) -> list[SurveyLine]:
        """The lines that stand out, strongest first.

        f_from / f_to      the range to look in
        top                how many to return
        min_separation_hz  two candidates closer than this are one line
        min_excess_db      below this a candidate is not reported at all
        """
        if not self.ready:
            return []
        f_to = self.freqs[-1] if f_to is None else f_to
        db = self.spectrum_db()
        floor = self.local_floor_db(db)
        excess = db - floor
        sel = (self.freqs >= f_from) & (self.freqs <= f_to)
        idx = np.flatnonzero(sel)
        if idx.size < 8:
            return []
        # interior local maxima only: a shoulder is not a line
        inner = idx[1:-1]
        is_peak = (db[inner] > db[inner - 1]) & (db[inner] >= db[inner + 1])
        cands = inner[is_peak & (excess[inner] >= min_excess_db)]
        if cands.size == 0:
            return []
        order = cands[np.argsort(excess[cands])[::-1]]

        stack = np.array(self._windows, dtype=np.float64)
        out: list[SurveyLine] = []
        taken: list[float] = []
        for k in order:
            f0 = float(self.freqs[k])
            if any(abs(f0 - t) < min_separation_hz for t in taken):
                continue
            taken.append(f0)
            a, b = max(k - 1, 0), min(k + 2, self.freqs.size)
            per_window = 10.0 * np.log10(np.maximum(stack[:, a:b].sum(axis=1), 1e-30))
            spread = float(np.percentile(per_window, 90) - np.percentile(per_window, 10))
            out.append(SurveyLine(f0, float(excess[k]), float(db[k]), spread, len(self._windows)))
            if len(out) >= top:
                break
        return out

    def band_shape(self, edges) -> list[tuple[float, float, float, float, float]]:
        """(lo, hi, median dB, peak dB, peak Hz) for each band, for the readout."""
        db = self.spectrum_db()
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (self.freqs >= lo) & (self.freqs < hi)
            if not s.any():
                continue
            seg, fseg = db[s], self.freqs[s]
            k = int(np.argmax(seg))
            rows.append((float(lo), float(hi), float(np.median(seg)),
                         float(seg[k]), float(fseg[k])))
        return rows
