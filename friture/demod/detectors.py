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

"""Per-sample detectors on a complex baseband: frequency, amplitude, tone, phase side.

Ported from ultraScan's dsp/demod.py (M22) and dsp/demod_digital.py (M23,
M26), on numpy alone. Each one answers a question for every SAMPLE and stops
there -- none of them knows what a symbol is. The step from a per-sample
trace to symbols and bits is friture.demod.symbols, one implementation shared
by every keyed mode rather than one per detector.

  FmDemodulator     "what frequency, when":  f[n] = angle(z[n]*conj(z[n-1])) * fs/2pi
  AmDemodulator     "how loud, when":        env[n] = abs(z[n])
  FskDemodulator    "which of these tones":  the frequency snapped to a tone list,
                                             or "none of these"
  DbpskDemodulator  "which side of the carrier's own line": squaring for the
                                             carrier, then a sign -- the phase-
                                             keyed mode that IS recoverable here
  estimate_fsk_tones  the tone list, guessed from the frequency histogram

All block-streaming: the only carried state is what a block boundary would
otherwise destroy, so however the capture is chopped into blocks the answer
is the same as for one long call.

Frequencies are BASEBAND-RELATIVE: 0 Hz is whatever the DDC put at DC, which
for friture.demod.ddc.ComplexDdc is the band's LOWER EDGE.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


class _BlockMovingAverage:
    """N-tap moving average whose history survives block edges.

    Direct convolution with the last N-1 inputs carried: the same samples
    scipy's lfilter(b, 1, x, zi) would give for this FIR. taps == 1 is a
    pass-through (no state, no copy).
    """

    def __init__(self, taps: int) -> None:
        self.taps = int(taps)
        self._b = np.full(self.taps, 1.0 / self.taps, dtype=np.float64)
        self.reset()

    @property
    def active(self) -> bool:
        return self.taps > 1

    def reset(self) -> None:
        self._history = np.zeros(self.taps - 1, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.active or x.size == 0:
            return x
        stream = np.concatenate((self._history, np.asarray(x, dtype=np.float64)))
        y = np.convolve(stream, self._b, mode="valid")
        self._history = stream[-(self.taps - 1):]
        return y


class FmDemodulator:
    """Phase-difference FM demodulator: complex baseband in, Hz per sample out.

    ``f[n] = angle(z[n] * conj(z[n-1])) * fs_bb / (2 pi)``

    The angle of one sample's phase advance over one sample period IS the
    instantaneous frequency; nothing more is needed.

    LIMIT -- deviations beyond +-fs_bb/2 WRAP (aliasing, by construction).
    ``np.angle`` returns (-pi, +pi], so the output is confined to
    (-fs_bb/2, +fs_bb/2]. Keep the content of interest inside that by
    choosing the band width; it cannot be recovered afterwards.

    ``np.unwrap`` is not used and must not be added: it cannot undo the
    aliasing (the information is gone) and it lets one noisy sample push every
    later sample by a multiple of fs_bb.

    Block streaming: the only carried state is the previous block's last
    sample, plus the smoother's history.
    """

    def __init__(self, fs_bb: float, amp_gate: float = 0.0,
                 smooth_taps: int = 1) -> None:
        """
        fs_bb       : complex baseband sample rate [Hz] (> 0)
        amp_gate    : samples whose |z| is below it are marked valid=False.
                      0.0 disables it (default).
        smooth_taps : moving-average length on the output, in samples. 1
                      disables it (default). State is carried across blocks.
        """
        if fs_bb <= 0:
            raise ValueError(f"fs_bb must be > 0, got {fs_bb!r}")
        if smooth_taps < 1:
            raise ValueError(f"smooth_taps must be >= 1, got {smooth_taps!r}")
        if amp_gate < 0:
            raise ValueError(f"amp_gate must be >= 0, got {amp_gate!r}")

        self.fs_bb = float(fs_bb)
        self.amp_gate = float(amp_gate)
        self.smooth_taps = int(smooth_taps)
        self._scale = self.fs_bb / _TWO_PI      # radians/sample -> Hz
        self._smoother = _BlockMovingAverage(self.smooth_taps)
        self._prev: complex | None = None       # last sample of the last block

    def reset(self) -> None:
        """Drop carried state (call after an upstream gap or a reconfigure)."""
        self._prev = None
        self._smoother.reset()

    def set_smoothing(self, taps: int) -> None:
        """Change the moving-average length. The smoother's history restarts."""
        taps = max(1, int(taps))
        if taps != self.smooth_taps:
            self.smooth_taps = taps
            self._smoother = _BlockMovingAverage(taps)

    def process(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        z : 1-D complex ndarray

        Returns ``(freq_hz, valid)``, both ``len(z)`` long:
            freq_hz : float32, baseband-relative instantaneous frequency [Hz]
            valid   : bool, True where the amplitude gate passed at BOTH ends
                      of the phase difference

        Right after ``reset()`` there is no previous sample, so ``f[0]`` is 0.0
        by construction. Where ``valid`` is False the frequency is left as
        computed, not NaN: dropping gated samples is the caller's decision, and
        a NaN would silently poison a later mean.
        """
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        if z.size == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)

        prev = z[0] if self._prev is None else self._prev
        z_prev = np.empty_like(z)
        z_prev[0] = prev
        z_prev[1:] = z[:-1]

        freq = np.angle(z * np.conj(z_prev)) * self._scale
        # Both ends must clear the gate: a phase difference against a near-zero
        # sample is a random angle, so one dead end poisons the sample.
        valid = (np.abs(z) >= self.amp_gate) & (np.abs(z_prev) >= self.amp_gate)

        self._prev = complex(z[-1])
        freq = self._smoother.process(freq)
        return freq.astype(np.float32), valid


class AmDemodulator:
    """Envelope detector: ``env[n] = abs(z[n])``, optionally smoothed."""

    def __init__(self, smooth_taps: int = 1) -> None:
        if smooth_taps < 1:
            raise ValueError(f"smooth_taps must be >= 1, got {smooth_taps!r}")
        self.smooth_taps = int(smooth_taps)
        self._smoother = _BlockMovingAverage(self.smooth_taps)

    def reset(self) -> None:
        self._smoother.reset()

    def process(self, z: np.ndarray) -> np.ndarray:
        """z : 1-D complex ndarray -> float32 envelope, ``len(z)`` long."""
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        if z.size == 0:
            return np.empty(0, dtype=np.float32)
        env = self._smoother.process(np.abs(z))
        return env.astype(np.float32)


class PmDemodulator:
    """Phase demodulator: accumulated phase [rad] of a complex baseband.

    ``dphi[n] = angle(z[n] * conj(z[n-1]))`` -- the same one-sample phase
    advance FmDemodulator measures, but summed instead of scaled to Hz:

        phi[n] = phi[n-1] + dphi[n] - 2 pi f_offset_hz / fs_bb

    A carrier that is not at baseband DC makes phi climb forever; that ramp
    IS the frequency offset. Give ``f_offset_hz`` and it is subtracted as it
    accumulates, leaving what the phase does ON TOP of a steady carrier.
    ``set_offset`` changes it mid-stream: the phase accumulated so far stays,
    the slope from here on changes.

    BPSK CANNOT be demodulated this way: a +-pi step sits on ``np.angle``'s
    branch cut and its sign is not in the data, so every such step is a coin
    flip that offsets everything after it. That is DpskDemodulator's job.
    Steps smaller than pi, and slow drift, are tracked correctly.

    ``np.unwrap`` is not used: the accumulation IS the unwrap, done causally
    so it streams. Carried state: the previous sample and the running phase.
    Note the phase grows without bound, so float32's resolution decays with
    it; the live readout detrends a window rather than accumulating forever.
    """

    def __init__(self, fs_bb: float, f_offset_hz: float = 0.0) -> None:
        if fs_bb <= 0:
            raise ValueError(f"fs_bb must be > 0, got {fs_bb!r}")
        self.fs_bb = float(fs_bb)
        self.set_offset(f_offset_hz)
        self.reset()

    def set_offset(self, f_offset_hz: float) -> None:
        """Carrier offset from baseband DC to remove [Hz]; may be negative."""
        self.f_offset_hz = float(f_offset_hz)
        self._dphi_offset = _TWO_PI * self.f_offset_hz / self.fs_bb

    def reset(self) -> None:
        """Drop carried state: the phase restarts from 0 at the next sample."""
        self._prev: complex | None = None
        self._phi = 0.0

    def process(self, z: np.ndarray) -> np.ndarray:
        """z : 1-D complex -> float32 accumulated phase [rad], ``len(z)`` long.

        Right after ``reset()`` there is no previous sample, so ``phi[0]`` is
        exactly 0.0 and the offset ramp is not charged for an interval that
        did not happen. Empty input returns empty and changes no state.
        """
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        if z.size == 0:
            return np.empty(0, dtype=np.float32)
        first = self._prev is None
        z_prev = np.empty_like(z)
        z_prev[0] = z[0] if first else self._prev
        z_prev[1:] = z[:-1]
        inc = np.angle(z * np.conj(z_prev)) - self._dphi_offset
        if first:
            inc[0] = 0.0
        phi = self._phi + np.cumsum(inc)
        self._phi = float(phi[-1])
        self._prev = complex(z[-1])
        return phi.astype(np.float32)


class FskDemodulator:
    """Assign each sample's instantaneous frequency to the nearest known tone.

    Wraps one FmDemodulator and snaps its Hz output onto ``tone_freqs_hz``. A
    sample further than ``tolerance_hz`` from EVERY tone is reported invalid
    rather than being forced into the closest one -- "none of these" is an
    answer, and it is the answer that keeps a noise-only band from producing
    a plausible bit stream.

    PER-SAMPLE ONLY: no symbol clock, no majority vote, no bits. Feed
    ``(tone_index, valid)`` to friture.demod.symbols exactly as returned.

    ``tone_freqs_hz`` is a public attribute: the live path re-estimates the
    tones from the signal itself and updates it in place.
    """

    def __init__(self, fs_bb: float, tone_freqs_hz, tolerance_hz: float,
                 amp_gate: float = 0.0, smooth_taps: int = 1) -> None:
        """
        fs_bb         : complex baseband sample rate [Hz] (> 0)
        tone_freqs_hz : two or more baseband-relative tone frequencies [Hz].
                        Order is preserved -- the returned index points into
                        this list as given. At most 127 tones (int8 output).
        tolerance_hz  : a sample further than this from the nearest tone is
                        invalid (> 0)
        amp_gate      : delegated to FmDemodulator
        smooth_taps   : delegated to FmDemodulator (the one knob that trades
                        symbol-edge sharpness for noise immunity)
        """
        tones = np.asarray(tone_freqs_hz, dtype=np.float64).reshape(-1)
        if tones.size < 2:
            raise ValueError(
                f"tone_freqs_hz needs >= 2 tones (that is what makes it keying), "
                f"got {tones.size}")
        if tones.size > 127:
            raise ValueError("at most 127 tones (the index is int8)")
        if tolerance_hz <= 0:
            raise ValueError(f"tolerance_hz must be > 0, got {tolerance_hz!r}")

        self._fm = FmDemodulator(fs_bb, amp_gate=amp_gate, smooth_taps=smooth_taps)
        self.fs_bb = self._fm.fs_bb
        self.tone_freqs_hz = tones
        self.tolerance_hz = float(tolerance_hz)

    def reset(self) -> None:
        self._fm.reset()

    @property
    def smooth_taps(self) -> int:
        return self._fm.smooth_taps

    def set_smoothing(self, taps: int) -> None:
        """Moving average on the frequency trace before classification.

        THE knob for a real channel. In a room the previous tone's
        reverberation overlaps the new tone, and the instantaneous frequency
        of two tones beating is not their mean: it swings around the stronger
        one by up to spacing * a / (1 - a) for an interferer a times as
        strong -- at 4 kHz spacing and a reflection 10 dB down, +-1.9 kHz, at
        the beat rate. Measured in this room at 12/16 kHz: for 20 ms after a
        switch only 26% of samples sat within 1 kHz of the new tone, so a
        fixed tolerance called most of every symbol "neither". Averaging over
        ONE beat period (fs_bb / spacing samples) cancels the swing exactly
        and costs half that in delay.
        """
        self._fm.set_smoothing(taps)

    def process(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns ``(tone_index, valid)``, both ``len(z)`` long:
            tone_index : int8, index into ``tone_freqs_hz``, or -1 where invalid
            valid      : bool, the FM demodulator's own validity AND within
                         ``tolerance_hz`` of some tone
        """
        freq, valid = self._fm.process(z)
        if freq.size == 0:
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=bool)

        tones = np.asarray(self.tone_freqs_hz, dtype=np.float64).reshape(-1)
        f = freq.astype(np.float64)
        dist = np.abs(f[:, None] - tones[None, :])
        idx = np.argmin(dist, axis=1)
        nearest = dist[np.arange(f.size), idx]

        ok = valid & (nearest <= self.tolerance_hz)
        return np.where(ok, idx, -1).astype(np.int8), ok


def estimate_fsk_tones(freq_hz: np.ndarray, valid: np.ndarray, n_tones: int,
                       bins: int = 256, exclude_hz: float | None = None) -> list[float]:
    """Guess the tone set from an instantaneous-frequency trace (stateless).

    An FSK signal parks on a few frequencies and hurries between them, so its
    frequency histogram is a few tall spikes on an almost empty floor. Take
    the tallest bin, blank out its neighbourhood so its own shoulder cannot be
    picked twice, take the next tallest, and so on.

    exclude_hz : how far around a picked tone is blanked before the next
        pick. None (default) uses half the range shared evenly between the
        tones, which assumes the trace's range IS the tones' range. That is
        true of a clean signal and false of a real one: reverberation swings
        the instantaneous frequency far outside the tones at every switch,
        the range explodes, and the zone then swallows the second tone (a
        room at 12/16 kHz: no pair was ever adopted). Give the classifier's
        tolerance here instead -- tones closer than that cannot be told
        apart anyway.

    Returns bin CENTRES, sorted ascending -- quantised to (range / bins).
    Fewer than ``n_tones`` entries come back if the exclusion zones consume
    every remaining occupied bin.

    Raises ValueError if fewer than ``bins`` valid samples are available: a
    histogram with less than one sample per bin is noise with a shape, and
    reading tones off it would be exactly the mistake this is meant to avoid.
    """
    if n_tones < 1:
        raise ValueError(f"n_tones must be >= 1, got {n_tones!r}")
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins!r}")

    f = np.asarray(freq_hz, dtype=np.float64).reshape(-1)
    v = np.asarray(valid, dtype=bool).reshape(-1)
    if f.size != v.size:
        raise ValueError(
            f"freq_hz and valid must be the same length, got {f.size} and {v.size}")
    f = f[v]
    if f.size < bins:
        raise ValueError(
            f"need >= {bins} valid samples for a {bins}-bin histogram, got {f.size}")

    counts, edges = np.histogram(f, bins=int(bins))
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Half a tone's worth of spacing, assuming the tones share the range
    # evenly: wide enough to swallow one peak's shoulders, narrow enough to
    # keep the neighbouring tone visible.
    if exclude_hz is None:
        exclude_hz = (edges[-1] - edges[0]) / float(n_tones) / 2.0
    exclude_hz = max(float(exclude_hz), float(edges[1] - edges[0]))

    remaining = counts.astype(np.float64)
    picked: list[float] = []
    for _ in range(int(n_tones)):
        k = int(np.argmax(remaining))
        if remaining[k] <= 0.0:
            break                              # nothing occupied left to claim
        picked.append(float(centres[k]))
        remaining[np.abs(centres - centres[k]) <= exclude_hz] = -1.0
    return sorted(picked)


class DpskDemodulator:
    """Differential M-ary PSK, M = 2 (DBPSK) or 4 (DQPSK): raise to the M-th power, then decide.

    A PSK symbol change is a phase jump of 2 pi/M; for M = 2 that is +-pi,
    which sits on ``np.angle``'s branch cut, so accumulating the phase cannot
    decode it. But the two things this class needs both survive the cut:

      WHERE THE CARRIER IS.  ``z**M`` raises the modulation away (a symbol is
        a multiplication by an M-th root of unity, and that to the M-th power
        is 1), leaving a clean tone at M times the carrier. Its instantaneous
        frequency is measured with the same FmDemodulator everything else
        uses, taken as a per-block MEDIAN so a stray sample cannot move it,
        and divided by M.

      WHICH POINT EACH SAMPLE IS NEAREST.  After de-rotating by that carrier
        the constellation is M points evenly spaced around the origin.
        Rotating it so one point sits on the positive real axis makes the
        decision a comparison: for M = 2, ``real(z) < 0``; for M = 4, the
        quadrant.

    What remains unknown is WHICH point is "0": the M-th power destroys that
    by construction. It does not need to be recovered: the output is a
    per-sample trace, and ``symbols.differential_bits`` reads the CHANGES,
    which is what differential PSK transmits in the first place.

    THE ALIASING PROBLEM, AND WHY THE SIGNAL IS DE-ROTATED BEFORE IT IS
    RAISED.  Raising multiplies the carrier by M, so a carrier above
    fs_bb/(2M) wraps, and even one below it has its noise wrap: measured
    with M = 4 on a 5 kHz carrier (4 f_c = 20 kHz against a 25 kHz Nyquist)
    at 20 dB SNR, the noisy trace crossed the wrap often enough to bias the
    median by 120 Hz, and smoothing that wrapped trace made it 760 Hz. So a
    COARSE carrier -- the median instantaneous frequency of ``z`` itself,
    which the keying jumps disturb only a little -- is removed first, by an
    accumulated ramp; the raised signal then sits near DC, its frequency is
    M times the small residual with no wrap in reach, and it can be
    smoothed freely (there is no modulation left in it to smear).

    THE LOCK INDICATOR (``last_lock``), and why this class would be dangerous
    without it: when the carrier estimate degrades, the residual rotation
    turns the decision into a periodic trace at a multiple of the error -- a
    genuine periodicity that the clock recovery locks onto with high
    confidence and the slicer reads with high agreement.
    ``|mean(d**M)| / mean(|d|**M)`` is how well-formed the constellation is
    (1.0 on the points, 0.0 for noise) and falls off fast: ultraScan measured
    1.00 / 0.52 working against 0.02 / 0.01 broken for M = 2. This is the
    number that must gate a decode; a threshold around 0.3 is easy to place.

    Block-streaming: the de-rotation ramp and the constellation angle are
    ACCUMULATED, never re-derived per block, so a new carrier estimate slides
    the reference instead of stepping it.

    THE ANGLE IS TRACKED BY A LOOP, NOT BY DIFFERENCING MEASUREMENTS. The
    per-block carrier measurement is biased by the transition samples (the
    median of a block holding a few smeared phase jumps sits about a hertz
    off), so the de-rotated constellation turns slowly at the residual. As
    first written -- ported from ultraScan -- the angle advanced by
    angle_ema times the step between the previous and current MEASUREMENTS,
    which follows a steady turn at only a fifth of its rate: the tracked
    constellation fell behind, crossed a decision boundary every half second,
    and the trace changed its labelling each time. Measured on synthetic
    200 Bd DBPSK through the DDC: agreement with the sent symbols
    0.98 / 0.02 / 0.97 / 0.02 in alternating 0.5 s stretches, at lock 1.00 and
    a 1.28 Hz carrier error -- one wrong differential bit per crossing. Now
    the step is the error between the TRACKED angle and the measurement
    (proportional), and a persistent error is folded back into the carrier
    (integral), so the ramp catches up and the lag goes to zero instead of
    accumulating. The measurement itself only steers the carrier while it is
    outside the loop's reach (acquisition); inside it, the constellation is
    the better reference.
    """

    ORDER = 2
    # Frequency feedback per block, as a fraction of the angle error
    # converted to hertz. With angle_ema 0.2 this is a second-order loop
    # with poles at |z| ~ 0.89 per block: stable, settles in ~10 blocks.
    FREQ_GAIN = 0.02
    # No frequency feedback from a block too short to measure one: a single
    # sample's angle error over a single sample's time is any frequency.
    MIN_LOOP_BLOCK = 64
    # smoothing on the coarse carrier estimate, per block
    COARSE_EMA = 0.3
    # Samples whose amplitude is under this share of the block's median are
    # not used for the carrier: a band-limited phase step passes through a
    # dip (to zero for a reversal, to ~0.7 for a quarter turn) while the
    # phase swings between the points.
    TRANSITION_GATE = 0.8
    # the circular mean of the raised phase advance runs over this many
    # samples before its angles are taken and their median found
    STEP_SMOOTH_TAPS = 16

    def __init__(self, fs_bb: float, *, amp_gate: float = 0.0,
                 offset_ema: float = 0.2, angle_ema: float = 0.2,
                 order: int | None = None) -> None:
        """
        fs_bb      : complex baseband sample rate [Hz] (> 0)
        amp_gate   : samples with |z| below this are invalid (>= 0)
        offset_ema : smoothing on the carrier estimate while acquiring, per block
        angle_ema  : smoothing on the constellation angle, per block
        order      : 2 or 4 (default: the class's ORDER)
        """
        if amp_gate < 0:
            raise ValueError(f"amp_gate must be >= 0, got {amp_gate!r}")
        for name, val in (("offset_ema", offset_ema), ("angle_ema", angle_ema)):
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {val!r}")
        self.order = int(self.ORDER if order is None else order)
        if self.order not in (2, 4):
            raise ValueError(f"order must be 2 or 4, got {order!r}")
        # The coarse detector runs UNGATED; the gate is applied to |z|
        # directly, below.
        self._fm1 = FmDemodulator(fs_bb, amp_gate=0.0, smooth_taps=1)
        self.fs_bb = self._fm1.fs_bb
        self.amp_gate = float(amp_gate)
        self.offset_ema = float(offset_ema)
        self.angle_ema = float(angle_ema)
        self.last_carrier_hz = 0.0
        self.carrier_in_range = True
        self.last_lock = 0.0
        self.reset()

    def reset(self) -> None:
        """Drop every carried quantity: ramps, angle, and the coarse detector."""
        self._fm1.reset()
        self._carrier: float | None = None   # estimated carrier [Hz]
        self._carrier_prev: float | None = None  # ...as it was at the end of the last block
        self._ramp = 0.0                     # accumulated de-rotation phase [rad]
        self._thetaM: float | None = None    # tracked constellation angle, times M
        self._coarse: float | None = None    # the coarse carrier the raise is referenced to
        self._ramp_c = 0.0                   # its accumulated phase [rad]
        self._prev_unit: complex | None = None  # last raised unit vector, for the lag-1 term
        self._step_hist = np.zeros(self.STEP_SMOOTH_TAPS - 1, dtype=np.complex128)

    def process(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns ``(sym, valid)``, both ``len(z)`` long:
            sym   : int8 in 0..M-1 -- which constellation point this sample is
                    nearest. NOT bits: the labelling is arbitrary by
                    construction; ``symbols.differential_bits`` reads the changes.
            valid : bool, the amplitude gate on |z|
        """
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        n = z.size
        if n == 0:
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=bool)
        M = self.order

        # 1. Carrier. A COARSE estimate first -- the median instantaneous
        #    frequency of z itself, smoothed across blocks -- is removed by
        #    an accumulated ramp; then the raised signal sits near DC (see
        #    the class docstring) and its residual frequency is the angle of
        #    the summed lag-1 phase advance of its UNIT vectors, M times the
        #    carrier's residual -- a mean of phase advances with no wrap in
        #    reach, which averages the noise instead of voting on it.
        #    Unit vectors, not the raw products: |z**M| is |z|**M, and at
        #    15 dB SNR that puts the estimate in the hands of a few loud
        #    samples (measured, M = 4: lock 0.88 with a median, 0.46 with
        #    the power-weighted sum). The symbol transitions, where the
        #    band-limited z dips and its phase passes between the points,
        #    are left out by a relative amplitude gate instead. (A median of
        #    the instantaneous frequency was used first: it was biased by
        #    wrap-around near Nyquist, and smoothing it smeared the
        #    transition glitches over a whole symbol at 2000 Bd.)
        f_direct = float(np.median(self._fm1.process(z)[0]))
        self._coarse = (f_direct if self._coarse is None
                        else (1.0 - self.COARSE_EMA) * self._coarse + self.COARSE_EMA * f_direct)
        ramp_c = self._ramp_c + _TWO_PI * self._coarse * np.arange(1, n + 1) / self.fs_bb
        self._ramp_c = float(ramp_c[-1] % _TWO_PI)
        zM = (z * np.exp(-1j * ramp_c)) ** M
        amp = np.abs(z)
        keep = amp >= self.TRANSITION_GATE * float(np.median(amp))
        unit = np.where(keep, zM / np.maximum(np.abs(zM), 1e-300), 0.0)
        prev = np.empty_like(unit)
        prev[0] = 0.0 if self._prev_unit is None else self._prev_unit
        prev[1:] = unit[:-1]
        self._prev_unit = complex(unit[-1])
        # the per-sample phase advance as a vector (zero where gated), a
        # short circular mean of it, and the MEDIAN of the angles: the mean
        # beats down the noise, the median throws out the clicks -- the
        # bursts where noise carries the signal round the origin -- that
        # a plain vector sum over the block lets in. Measured, M = 4 at
        # 15 dB: a whole-block vector sum left 7-18 Hz of error and the
        # lock at 0.46-0.80; this leaves the loop in charge.
        steps = unit * np.conj(prev)
        stream = np.concatenate((self._step_hist, steps))
        smooth = np.convolve(stream, np.ones(self.STEP_SMOOTH_TAPS), mode="valid")
        self._step_hist = stream[-(self.STEP_SMOOTH_TAPS - 1):]
        good = np.abs(smooth) > 0.5 * self.STEP_SMOOTH_TAPS   # over half the window unbiased
        f_res = float(np.median(np.angle(smooth[good]))) if good.any() else 0.0
        f_c = self._coarse + f_res * self.fs_bb / (_TWO_PI * M)
        # Agreement AFTER resolving: a gap still much larger than the direct
        # median's own error means neither measurement is usable here.
        self.carrier_in_range = abs(f_c - f_direct) < self.fs_bb / (4.0 * M)
        # The loop below can pull in a residual of up to a quarter turn per
        # block in the raised domain. Outside that the angle error wraps
        # and the loop has nothing to pull on, so the measurement steers
        # (acquisition); inside it the measurement is left alone.
        block_s = n / self.fs_bb
        pull_in_hz = 1.0 / (4.0 * M * block_s)
        if self._carrier is None:
            self._carrier = f_c
        elif abs(f_c - self._carrier) > pull_in_hz:
            self._carrier = ((1.0 - self.offset_ema) * self._carrier
                             + self.offset_ema * f_c)

        # 2. De-rotate by an ACCUMULATED ramp whose slope GLIDES from the
        #    previous estimate to this one across the block. A step in the
        #    slope at every block edge put a kink in the de-rotated phase
        #    122 times a second, and at low SNR -- where the per-block
        #    estimate jitters by tens of hertz -- those kinks flipped
        #    decisions on a schedule: the clock recovery then found a line
        #    near the block rate at 16-25 dB confidence with the lock still
        #    high (measured, M = 2 at 10 dB: 109 Bd, lock 0.91, 62% of the
        #    bits right). Sliding the slope leaves the phase smooth.
        start = self._carrier if self._carrier_prev is None else self._carrier_prev
        slope = np.linspace(start, self._carrier, n)
        self._carrier_prev = self._carrier
        ramp = self._ramp + np.cumsum(_TWO_PI * slope / self.fs_bb)
        self._ramp = float(ramp[-1] % _TWO_PI)
        d = z * np.exp(-1j * ramp)

        # 3. Constellation angle, tracked in the RAISED domain -- where the
        #    M points coincide, so the estimate cannot flicker between them.
        #    Accumulated (unwrapped) so a slow drift cannot step across a
        #    branch cut, which would relabel the trace and forge a
        #    transition out of nothing.
        dM = d ** M
        u = np.mean(dM)
        mag = abs(u)
        if mag > 0.0:
            u = u / mag
            if self._thetaM is None:
                self._thetaM = float(np.angle(u))
            else:
                # proportional: the error between the tracked angle and this
                # block's measurement -- see the class docstring for why not
                # the step between two measurements
                err = float(np.angle(u * np.exp(-1j * self._thetaM)))
                self._thetaM += self.angle_ema * err
                # integral: a persistent error is the constellation turning
                # at the residual carrier error, 2 pi M df block_s per block
                # in the raised domain. Fold it back into the carrier.
                if n >= self.MIN_LOOP_BLOCK:
                    df = self.FREQ_GAIN * err / (M * _TWO_PI * block_s)
                    self._carrier += float(np.clip(df, -pull_in_hz, pull_in_hz))
        self.last_carrier_hz = self._carrier
        theta = 0.0 if self._thetaM is None else self._thetaM / M

        # 4. The lock indicator -- see the class docstring.
        p = float(np.mean(np.abs(d) ** M))
        self.last_lock = 0.0 if p <= 0.0 else float(np.abs(np.mean(dM)) / p)

        # 5. A comparison, not a phase accumulation: the nearest point.
        a = d * np.exp(-1j * theta)
        if M == 2:
            sym = (a.real < 0.0).astype(np.int8)
        else:
            sym = (np.round(np.angle(a) / (_TWO_PI / M)).astype(np.int64) % M).astype(np.int8)
        valid = np.abs(z) >= self.amp_gate
        return sym, valid


class DbpskDemodulator(DpskDemodulator):
    """Differential BPSK: DpskDemodulator with M = 2."""

    ORDER = 2


class DqpskDemodulator(DpskDemodulator):
    """Differential QPSK: DpskDemodulator with M = 4, two bits per symbol.

    The differential symbol ``(sym[k] - sym[k-1]) mod 4`` is the phase step
    in quarter turns, 0..3; ``symbols.symbols_to_bits(..., 2)`` writes it as
    two bits, most significant first. Which dibit a real link assigns to
    which quarter turn (Gray or natural, and in which rotation sense) is the
    link's convention, not something the signal says.
    """

    ORDER = 4
