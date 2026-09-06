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

"""Symbol clock recovery and bit slicing -- "how fast, on what grid, which bits".

Ported from ultraScan's dsp/symbols.py (M26). The detectors in
friture.demod.detectors answer "which tone / which side" for every SAMPLE and
stop there. This module turns that per-sample trace into SYMBOLS and BITS
without being told the baud rate.

  estimate_symbol_clock  unknown baud -> rate + timing phase + a number saying
                         how much to believe it (stateless)
  SymbolClockTracker     the same estimate kept up to date on a rolling
                         history, for live use
  SymbolSlicer           rate + phase -> one decision per symbol, with the
                         per-symbol agreement that IS the eye
  differential_bits      symbols -> bits when only the CHANGES carry data
                         (DBPSK, and any NRZI-style coding)
  bits_to_text           the readout

WHY THE LOWEST PEAK, AND NOT THE BIGGEST

    A symbol trace changes only on the symbol grid, so its transitions sit at
    p_i = phi + k_i*T for integer k_i: the DATA decides WHICH grid points get
    a transition, never where the grid is. Write the transitions as an impulse
    train and its DFT at frequency r is

        X(r) = sum_i exp(-2 pi i * p_i * r)

    which is a "do these transition times agree on a period" score, for
    free, at every candidate rate at once. Substituting p_i = phi + k_i*T:

        r = 1/T   ->  exp(-2 pi i phi/T) * exp(-2 pi i k_i)   -> ALL ALIGN
        r = 2/T   ->  same, k_i doubles                       -> ALL ALIGN
        r = 1/2T  ->  exp(-pi i k_i) = +-1 by parity          -> CANCELS

    So the spectrum has lines at every HARMONIC k/T and nothing at the
    sub-harmonics. The fundamental is therefore the LOWEST line, not the
    tallest -- on a pattern with few adjacent-equal symbols the second
    harmonic is routinely taller. Picking the maximum would report double the
    true baud, and a doubled baud still slices into a plausible-looking bit
    stream, which is the failure that matters most. Hence the scan goes
    upward and takes the first line that clears the floor.

WHAT THE CONFIDENCE NUMBER IS FOR

    confidence_db is the winning line's height over the MEDIAN of the
    searched band -- a peak-to-floor ratio, not an SNR. Noise has no
    preferred period, so its transition train is flat and every candidate
    scores about the same: single-digit dB, and a rate that moves every
    window. Real keying sits tens of dB up and stays put.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_TWO_PI = 2.0 * np.pi

# A trace with fewer transitions than this cannot support a clock estimate:
# the peak-to-floor ratio of a handful of impulses is decided by WHICH
# handful, not by whether they share a period.
MIN_TRANSITIONS = 24

# How far below the band's tallest line the fundamental is still accepted.
# Real keying whose second harmonic dominates still puts the fundamental
# within a few dB of it; a noise ripple that happens to sit lower does not.
_HARMONIC_MARGIN_DB = 6.0
# Below this peak-to-floor ratio the band has no line at all, only ripple.
_MIN_PEAK_DB = 6.0
# The neighbourhood the confidence is measured against: half an octave
# each side of the line, and at least this many bins of it.
_LOCAL_FLOOR_SPAN_DOWN = 0.71
_LOCAL_FLOOR_SPAN_UP = 1.41
_LOCAL_FLOOR_MIN_BINS = 8


@dataclass(frozen=True)
class SymbolClock:
    """The recovered clock, plus what it is worth.

    rate_hz       : symbols per second. 0.0 when nothing was found.
    phase_samples : offset of a symbol boundary from sample 0 of the trace the
                    estimate was made on, in [0, period_samples).
    confidence_db : the winning spectral line over the band's median floor.
                    THE number to gate a decoder on.
    n_transitions : how many symbol changes the estimate is based on.
    fs            : sample rate the estimate was made at [Hz].
    """

    rate_hz: float
    phase_samples: float
    confidence_db: float
    n_transitions: int
    fs: float = field(default=1.0)

    @property
    def period_samples(self) -> float:
        """Samples per symbol, or 0.0 when rate_hz is 0."""
        return 0.0 if self.rate_hz <= 0.0 else self.fs / self.rate_hz


def transition_train(sym: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """1.0 where the symbol label changes, 0.0 elsewhere (float64, len(sym)).

    EACH VALID SAMPLE IS COMPARED WITH THE PREVIOUS VALID ONE, NOT ITS
    IMMEDIATE NEIGHBOUR. The obvious rule -- a change between two ADJACENT
    samples that are both valid -- silently deletes the data: in FSK the
    invalid region IS the transition (while the tone slides from one to the
    other it matches neither), so the trace reads 0 0 0 -1 -1 -1 1 1 1 and
    under the adjacent rule not one of those is a transition. ultraScan
    measured three transitions in 149504 samples of plainly keyed signal.

    Comparing against the previous VALID sample finds it, and still ignores
    what the adjacent rule was protecting against: an edge into or out of a
    gated region where the label did NOT change contributes nothing, so a
    gate opening and closing on its own rhythm cannot put a line at the
    gate's period.

    sym[0] has no predecessor and is never a transition.
    """
    s = np.asarray(sym).reshape(-1)
    if s.size == 0:
        return np.empty(0, dtype=np.float64)
    v = (np.ones(s.size, dtype=bool) if valid is None
         else np.asarray(valid, dtype=bool).reshape(-1))
    if v.size != s.size:
        raise ValueError(
            f"sym and valid must be the same length, got {s.size} and {v.size}")
    t = np.zeros(s.size, dtype=np.float64)
    where = np.flatnonzero(v)
    if where.size < 2:
        return t
    seen = s[where]
    t[where[1:][seen[1:] != seen[:-1]]] = 1.0
    return t


def estimate_symbol_clock(
    sym: np.ndarray,
    valid: np.ndarray | None,
    fs: float,
    *,
    baud_min: float,
    baud_max: float,
    pad: int = 4,
) -> SymbolClock:
    """Recover rate + timing phase from a symbol trace, baud unknown (stateless).

    sym      : per-sample symbol labels (ints: a tone index, an OOK on/off, a
               DBPSK side -- anything that only changes on the grid)
    valid    : companion mask, or None for "all valid"
    fs       : sample rate of sym [Hz]
    baud_min : lowest rate to consider [Hz]. A REAL commitment: a fundamental
               below this limit is invisible and the scan will return its
               second harmonic instead.
    baud_max : highest rate to consider [Hz]. Must be < fs/2, and in practice
               well under it -- fewer than ~4 samples per symbol leaves
               nothing for the slicer's guard band.
    pad      : FFT zero-padding factor (>= 1)

    Returns a SymbolClock. rate_hz == 0.0 (and confidence_db == 0.0) means
    "no clock here": too few transitions, or no line clearing the floor. That
    is a legitimate and common answer, and it is what noise must produce.

    Raises ValueError on impossible parameters only. Ordinary bad data never
    raises: noise in gives "no clock" out.
    """
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs!r}")
    if not (0 < baud_min < baud_max):
        raise ValueError(
            f"need 0 < baud_min < baud_max, got {baud_min!r} and {baud_max!r}")
    if baud_max >= fs / 2.0:
        raise ValueError(
            f"baud_max {baud_max!r} must be below Nyquist {fs / 2.0!r}")
    if pad < 1:
        raise ValueError(f"pad must be >= 1, got {pad!r}")

    t = transition_train(sym, valid)
    n_tr = int(t.sum())
    none = SymbolClock(0.0, 0.0, 0.0, n_tr, fs=float(fs))
    if t.size < 2 or n_tr < MIN_TRANSITIONS:
        return none

    # Remove the mean so the transition DENSITY (a large DC term) does not
    # share a spectrum with the lines. No taper: the train is impulses, not a
    # continuous waveform.
    x = t - t.mean()
    nfft = int(2 ** np.ceil(np.log2(max(x.size * pad, 2))))
    spec = np.fft.rfft(x, n=nfft)
    mag = np.abs(spec)

    lo = int(np.ceil(baud_min * nfft / fs))
    hi = int(np.floor(baud_max * nfft / fs))
    lo = max(lo, 1)
    hi = min(hi, mag.size - 2)
    if hi <= lo + 2:
        return none
    band = mag[lo:hi + 1]

    floor = float(np.median(band))
    if floor <= 0.0:
        return none

    peak_db = 20.0 * np.log10(float(band.max()) / floor)
    if peak_db < _MIN_PEAK_DB:
        return none
    # Anchor the threshold to the band's OWN tallest line. That is what makes
    # this work on a pattern whose second harmonic dominates: the fundamental
    # of real keying is never far below its own harmonic, while ripple is.
    thresh = floor * 10.0 ** ((peak_db - _HARMONIC_MARGIN_DB) / 20.0)

    interior = np.arange(1, band.size - 1)
    is_peak = ((band[interior] > band[interior - 1])
               & (band[interior] >= band[interior + 1]))
    cands = interior[is_peak & (band[interior] >= thresh)]
    if cands.size == 0:
        return none
    k = int(cands[0]) + lo

    # Parabolic interpolation on the log magnitude: the true rate almost
    # never lands on a bin centre, and a half-bin error at 1 kBd is a few Bd,
    # enough to walk a symbol boundary across a 100 ms trace.
    y0 = np.log(max(mag[k - 1], 1e-30))
    y1 = np.log(max(mag[k], 1e-30))
    y2 = np.log(max(mag[k + 1], 1e-30))
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0 if denom == 0.0 else 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    rate = float((k + delta) * fs / nfft)
    if not (baud_min <= rate <= baud_max):
        return none

    period = fs / rate
    # THE CONFIDENCE IS THE LINE OVER ITS OWN NEIGHBOURHOOD, not over the
    # whole band's median. A noisy trace whose labels flip slowly -- a
    # constellation turning at a residual carrier error, a random walk --
    # has a RED transition spectrum, and against the band's median its
    # lowest bump stands 20 dB tall while being no taller than the bins
    # beside it. Measured on DBPSK at 10 dB SNR: 41 Bd at 23 dB, bits
    # garbage. A real line rises out of a flat neighbourhood; the bump does
    # not. Half an octave each side, the line's own bins excluded.
    n_lo = max(lo, int(k * _LOCAL_FLOOR_SPAN_DOWN))
    n_hi = min(hi, int(np.ceil(k * _LOCAL_FLOOR_SPAN_UP)))
    around = np.concatenate((mag[n_lo:max(n_lo, k - 2)], mag[min(hi + 1, k + 3):n_hi + 1]))
    local = float(np.median(around)) if around.size >= _LOCAL_FLOOR_MIN_BINS else floor
    conf = 20.0 * np.log10(float(mag[k]) / max(floor, local))

    # THE PHASE IS NOT READ OFF THE BIN. angle(spec[k]) looks like the timing
    # offset and is not usable as one: the bin sits up to half a bin away
    # from the interpolated rate, and that residual is multiplied by the MEAN
    # SYMBOL INDEX before it reaches the phase (ultraScan measured 12 samples
    # of 250 from a rate error of 0.002%). The transitions themselves have no
    # such lever arm, so the phase is the CIRCULAR MEAN of the transition
    # positions modulo the refined period: every transition voting once, no
    # extrapolation.
    pos = np.flatnonzero(t).astype(np.float64)
    vec = np.exp(1j * _TWO_PI * pos / period).mean()
    if not np.isfinite(vec) or vec == 0:
        return none
    phase = float((np.angle(vec) / _TWO_PI) * period) % period
    return SymbolClock(rate, phase, conf, n_tr, fs=float(fs))


class SymbolClockTracker:
    """estimate_symbol_clock kept current on a rolling history (live).

    The estimator needs far more samples than one audio block holds (a
    2048-sample capture block is 410 baseband samples; a 200 Bd signal puts
    under two symbols in that). So the trace is accumulated in a ring of
    ``history`` samples and re-estimated every ``interval`` samples --
    decoupling "how often the clock is refreshed" from "how often audio
    arrives".

    The published rate is EMA-smoothed and the phase is carried as an
    ABSOLUTE sample index, so a jittering estimate slides the symbol grid
    instead of teleporting it. An estimate that comes back empty does not
    erase the last good one at once: confidence_db halves and the rate is
    kept for ``hold`` estimates, so one bad window inside a burst does not
    restart the decoder.
    """

    # An estimate further than this from the published rate replaces it
    # instead of being blended in -- see _estimate.
    JUMP_FRACTION = 0.02

    def __init__(self, fs: float, *, baud_min: float, baud_max: float,
                 history: int = 8192, interval: int = 2048,
                 rate_ema: float = 0.3, hold: int = 4):
        if history < 64:
            raise ValueError(f"history must be >= 64, got {history!r}")
        if not (0 < interval <= history):
            raise ValueError(
                f"need 0 < interval <= history, got {interval!r} and {history!r}")
        if not (0.0 < rate_ema <= 1.0):
            raise ValueError(f"rate_ema must be in (0, 1], got {rate_ema!r}")
        if hold < 1:
            raise ValueError(f"hold must be >= 1, got {hold!r}")
        self.fs = float(fs)
        self.baud_min = float(baud_min)
        self.baud_max = float(baud_max)
        self.history = int(history)
        self.interval = int(interval)
        self.rate_ema = float(rate_ema)
        self.hold = int(hold)
        self.reset()

    def reset(self) -> None:
        self._sym = np.zeros(0, dtype=np.int16)
        self._valid = np.zeros(0, dtype=bool)
        self._since = 0
        self._n_total = 0          # absolute index one past the last fed sample
        self._buf_start = 0        # absolute index of self._sym[0]
        self.rate_hz = 0.0
        self.phase_abs = 0.0       # absolute sample index of a symbol boundary
        self.confidence_db = 0.0
        self.n_transitions = 0
        self.n_estimates = 0
        self._misses = 0

    @property
    def period_samples(self) -> float:
        return 0.0 if self.rate_hz <= 0.0 else self.fs / self.rate_hz

    @property
    def locked(self) -> bool:
        return self.rate_hz > 0.0

    def process(self, sym: np.ndarray, valid: np.ndarray | None = None) -> bool:
        """Feed one block. Returns True if the clock was re-estimated here."""
        s = np.asarray(sym).reshape(-1).astype(np.int16, copy=False)
        if s.size == 0:
            return False
        v = (np.ones(s.size, dtype=bool) if valid is None
             else np.asarray(valid, dtype=bool).reshape(-1))
        if v.size != s.size:
            raise ValueError(
                f"sym and valid must be the same length, got {s.size} and {v.size}")
        self._sym = np.concatenate((self._sym, s))
        self._valid = np.concatenate((self._valid, v))
        self._n_total += s.size
        if self._sym.size > self.history:
            drop = self._sym.size - self.history
            self._sym = self._sym[drop:]
            self._valid = self._valid[drop:]
            self._buf_start += drop
        self._since += s.size
        if self._since < self.interval or self._sym.size < self.history // 2:
            return False
        self._since = 0
        self._estimate()
        return True

    def _estimate(self) -> None:
        est = estimate_symbol_clock(
            self._sym, self._valid, self.fs,
            baud_min=self.baud_min, baud_max=self.baud_max)
        self.n_estimates += 1
        self.n_transitions = est.n_transitions
        if est.rate_hz <= 0.0:
            self._misses += 1
            if self._misses >= self.hold:
                self.rate_hz = 0.0
                self.confidence_db = 0.0
            else:
                self.confidence_db *= 0.5
            return
        self._misses = 0
        self.confidence_db = est.confidence_db
        if (self.rate_hz <= 0.0
                or abs(est.rate_hz - self.rate_hz) > self.JUMP_FRACTION * self.rate_hz):
            # A different clock, not a refinement of this one: take it. The
            # smoothing below is for the jitter between estimates of ONE
            # clock; blending across two would publish a rate that is
            # neither. Measured on synthetic OOK through a room model: a
            # first, 5 dB estimate of 41.8 Bd followed by 22 dB estimates of
            # 50 Bd was published as 44, 46, 47... for a second and a half.
            self.rate_hz = est.rate_hz
        else:
            self.rate_hz = ((1.0 - self.rate_ema) * self.rate_hz
                            + self.rate_ema * est.rate_hz)
        # est.phase_samples is relative to sample 0 of the history buffer;
        # carry it as an ABSOLUTE index so the grid does not jump when the
        # buffer slides out from under it.
        period = self.fs / self.rate_hz
        self.phase_abs = float((self._buf_start + est.phase_samples) % period)


class SymbolSlicer:
    """Rate + phase -> one decision per symbol, plus the agreement behind it.

    Each symbol's decision is the MAJORITY label over the middle
    ``1 - 2*guard`` of its period. The guard is not decoration: a transition
    takes real time (the FM smoother, the FIR skirt, the transmitter's own
    rise time), so the samples next to a boundary belong to neither symbol.

    ``agreement`` -- the winning label's share of the valid samples in the
    guard window -- is this decoder's eye opening. 1.0 is a symbol every
    sample agreed on; 0.5 on two-level data is a coin flip. It is reported
    PER SYMBOL, so a burst that decodes cleanly in the middle and frays at
    the edges says so.

    Streaming: a symbol is emitted only once its whole period has arrived,
    and the leftover samples are carried. Absolute sample indices are used
    throughout, so changing the clock mid-stream slides the grid rather than
    renumbering everything that came before -- see set_clock for why that
    has to be done by TIME and not by symbol number.
    """

    def __init__(self, n_levels: int = 2, guard: float = 0.25, min_samples: int = 3):
        if n_levels < 2:
            raise ValueError(f"n_levels must be >= 2, got {n_levels!r}")
        if not (0.0 <= guard < 0.5):
            raise ValueError(f"guard must be in [0, 0.5), got {guard!r}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {min_samples!r}")
        self.n_levels = int(n_levels)
        self.guard = float(guard)
        self.min_samples = int(min_samples)
        self.period = 0.0
        self.phase = 0.0
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        self._val = np.zeros(0, dtype=bool)
        self._start = 0            # absolute index of _buf[0]
        self._next_abs: float | None = None   # absolute start of the next symbol to emit
        self.n_symbols = 0
        self.n_undecided = 0

    def set_clock(self, period_samples: float, phase_abs: float) -> None:
        """Adopt a new clock. A symbol already scheduled keeps its place in TIME.

        The tracker republishes the clock every block, and each estimate
        moves the grid a little: a rate refinement changes T, a phase
        refinement slides phi -- and phi is published modulo T, so between
        two estimates of the SAME grid it can flip from ~T to ~0. Numbering
        symbols by k on the new grid would then re-emit or skip a whole
        symbol: measured on synthetic 200 Bd OOK through the live chain, the
        readout carried one duplicated bit for each clock update that
        crossed the wrap (a 51/64 match against the sent bits, the halves
        offset by one). So the next symbol's start is carried as an absolute
        sample position and snapped to the NEAREST boundary of the new grid:
        a small refinement slides it, and a wrap lands on the same boundary.
        """
        self.period = float(period_samples)
        self.phase = float(phase_abs)
        if self.period > 0.0 and self._next_abs is not None:
            k = np.round((self._next_abs - self.phase) / self.period)
            self._next_abs = self.phase + float(k) * self.period

    def process(self, sym: np.ndarray, valid: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Feed one block; get every symbol that COMPLETED inside it.

        Returns ``(labels, agreement)``:
            labels    : int8, the majority label, or -1 for a symbol whose
                        guard window held fewer than ``min_samples`` valid
                        samples -- undecided, not guessed
            agreement : float32 in [0, 1], the winner's share (0.0 when -1)

        Both are empty while no symbol finished.
        """
        s = np.asarray(sym).reshape(-1).astype(np.int16, copy=False)
        v = (np.ones(s.size, dtype=bool) if valid is None
             else np.asarray(valid, dtype=bool).reshape(-1))
        if v.size != s.size:
            raise ValueError(
                f"sym and valid must be the same length, got {s.size} and {v.size}")
        if s.size:
            self._buf = np.concatenate((self._buf, s))
            self._val = np.concatenate((self._val, v))
        if self.period <= 0.0:
            # No clock: DISCARD, do not accumulate. The live path keeps
            # feeding an unlocked slicer so the absolute sample numbering
            # stays shared with the clock tracker; that must not grow a
            # buffer without bound. The index still advances, so the grid
            # means the same thing when a clock arrives.
            self._start += self._buf.size
            self._buf = self._buf[:0]
            self._val = self._val[:0]
            self._next_abs = None
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32)
        if self._buf.size == 0:
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32)

        T, g = self.period, self.guard
        end_abs = self._start + self._buf.size
        if self._next_abs is None or self._next_abs + T < self._start:
            # Nothing scheduled (a fresh lock), or scheduled so far back that
            # its samples are gone (a long gap): start at the first boundary
            # inside the buffer.
            self._next_abs = self.phase + float(np.ceil((self._start - self.phase) / T)) * T

        labels: list[int] = []
        agrees: list[float] = []
        while True:
            a = self._next_abs
            if a + T > end_abs:                 # this symbol has not finished
                break
            lo = int(np.ceil(a + g * T))
            hi = int(np.ceil(a + (1.0 - g) * T))
            if lo < self._start:                # its window slid out of the buffer
                self._next_abs = a + T
                continue
            i0 = lo - self._start
            i1 = min(max(hi - self._start, i0 + 1), self._buf.size)
            win, wv = self._buf[i0:i1], self._val[i0:i1]
            good = win[wv & (win >= 0)]
            if good.size < self.min_samples:
                labels.append(-1)
                agrees.append(0.0)
                self.n_undecided += 1
            else:
                counts = np.bincount(good, minlength=self.n_levels)
                top = int(np.argmax(counts))
                labels.append(top)
                agrees.append(float(counts[top]) / float(good.size))
            self.n_symbols += 1
            self._next_abs = a + T

        # Drop what no future symbol can need -- with HALF A PERIOD of slack
        # behind the next symbol. The next clock update may snap the grid
        # backwards by up to T/2 (set_clock), and if the buffer had been cut
        # at the guard window's first sample, a slide of even a fraction of
        # a sample would put that window "outside the buffer" and the whole
        # symbol would be skipped above. Measured on synthetic 200 Bd OOK
        # through the live chain: one dropped symbol per backward snap that
        # crossed an integer sample, at eye 1.00 and full confidence.
        keep_from = int(np.floor(self._next_abs - 0.5 * T)) - self._start
        keep_from = int(np.clip(keep_from, 0, self._buf.size))
        if keep_from:
            self._buf = self._buf[keep_from:]
            self._val = self._val[keep_from:]
            self._start += keep_from

        return (np.asarray(labels, dtype=np.int8),
                np.asarray(agrees, dtype=np.float32))


def differential_bits(labels: np.ndarray, n_levels: int = 2) -> np.ndarray:
    """Symbols -> bits by CHANGE, for coding where the absolute level is not data.

    bit[k] = (labels[k] - labels[k-1]) mod n_levels. This is what makes DBPSK
    decodable at all: the squaring carrier recovery leaves a 180 degree
    ambiguity, so "was it 0 or 1" is genuinely unknowable while "did it
    change" is exact. It is also NRZI's rule.

    An undecided symbol (-1) poisons the one bit it takes part in; that bit
    comes back as -1. Returns int8, one shorter than labels -- the first
    symbol is a reference, not a bit.
    """
    a = np.asarray(labels, dtype=np.int16).reshape(-1)
    if a.size < 2:
        return np.empty(0, dtype=np.int8)
    prev, cur = a[:-1], a[1:]
    ok = (prev >= 0) & (cur >= 0)
    return np.where(ok, (cur - prev) % int(n_levels), -1).astype(np.int8)


def symbols_to_bits(labels: np.ndarray, bits_per_symbol: int = 1) -> np.ndarray:
    """Each symbol -> its bits, most significant first; an undecided symbol
    (-1) becomes that many -1 bits, so nothing after it shifts.

    A 2-level symbol is its own bit. A 4-level one (4-FSK tone index, DQPSK
    quarter-turn step) is two bits in NATURAL binary order -- Gray coding, if
    the link uses it, is the link's convention, not the signal's.
    """
    a = np.asarray(labels, dtype=np.int16).reshape(-1)
    b = int(bits_per_symbol)
    if b <= 1:
        return a.astype(np.int8)
    out = np.empty(a.size * b, dtype=np.int8)
    for i in range(b):
        shift = b - 1 - i
        out[i::b] = np.where(a >= 0, (a >> shift) & 1, -1).astype(np.int8)
    return out


def bits_to_text(bits: np.ndarray, group: int = 8) -> str:
    """Bits as a grouped 0/1 string, with '.' for undecided -- for the readout.

    Grouping is what makes a repeating preamble visible at a glance; '.'
    keeps an undecided symbol from silently shifting every bit after it.
    """
    b = np.asarray(bits, dtype=np.int16).reshape(-1)
    chars = ["." if x < 0 else str(int(x)) for x in b]
    if group <= 0:
        return "".join(chars)
    return " ".join("".join(chars[i:i + group]) for i in range(0, len(chars), group))
