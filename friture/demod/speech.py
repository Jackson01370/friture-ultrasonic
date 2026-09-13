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

"""Is there a VOICE in this band? A hidden-Markov test, not a transcriber.

This does not recover speech and it does not say what was said. It answers
one question -- is something with the structure of a voice present -- and it
is built so that the answer can be disbelieved: the score means nothing on
its own and is meant to be compared with the same score measured on a
stretch known to hold no speech.

* THE POINT OF THE MARKOV CHAIN IS TIME, NOT CLASSIFICATION *

    Noise produces a convincing pitch in any single frame you care to look
    at -- pick the best lag out of two hundred and something and one of them
    always wins. What noise cannot produce is a pitch that MOVES SMOOTHLY
    over a fifth of a second, breaks for a consonant, and comes back near
    where it left. A voice does exactly that. So frames are not judged one
    at a time and counted up; a Viterbi pass over a lattice of pitch
    candidates scores whole PATHS, charging for every jump in pitch and
    every change of state. A momentary coincidence buys nothing, because it
    cannot be continued. That is the whole of the gain over per-frame tests,
    and it is why this reaches further down into the noise than measuring
    harmonicity and rhythm separately does.

* WHITENING COMES FIRST, OR NOTHING ELSE WORKS *

    Autocorrelation asks "does this repeat?", and coloured noise repeats: a
    room whose noise slopes tens of dB across the speech band is, to an
    autocorrelator, a low-pass ring with a broad peak at exactly the lags a
    voice lives at. That is the most dangerous kind of false positive
    because it is stable, and a per-frame test has no way to see through it.
    Dividing every frame by the long-term average magnitude spectrum of the
    segment flattens the room out, so what survives is periodicity the
    room shape does not already explain.

The score is a log-likelihood ratio: how much better the best voiced path
explains the segment than the hypothesis that nothing is voiced anywhere.
It is reported per second so that segments of different lengths compare.

WHAT THIS CANNOT DO, measured rather than guessed:

  It stops at about 0 dB. Buried in the real noise of a real room, real
  speech is found in 9 tries out of 10 down to the level of the noise
  itself and half the time 5 dB under it, over 20 s segments at a 5%
  false-alarm rate. Below about -10 dB it is gone entirely. A voice too
  quiet for the microphone to hear is not recovered by this or anything
  else -- there is no information left to recover.

  It cannot tell a voice from a BUZZER. A harmonic stack at a fixed pitch
  scores like speech: measured on this room, a 130 Hz square wave 25.29
  against real speech at +10 dB reading 25.36. Pitch movement was measured
  as a way to separate them and does not (speech 0.32-0.38 octaves of
  spread sits between a 130 Hz buzzer at 0.13 and a 200 Hz one at 0.68).
  Anything built on this has to say so to whoever reads its output.

  Steady tones, beating tones, mains hum and sustained chords are NOT
  confusable -- whitening removes them, and friture/test/test_speech_-
  detector.py pins each one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpeechEvidence:
    """What the detector found, and enough to argue with it."""

    score: float                 # log-likelihood ratio per second, vs "nothing voiced"
    voiced_fraction: float       # share of frames the path calls voiced
    median_f0_hz: float          # the pitch it settled on (nan if never voiced)
    longest_run_s: float         # the longest unbroken voiced stretch
    n_frames: int
    seconds: float

    def describe(self) -> str:
        if not np.isfinite(self.median_f0_hz):
            return "no voiced path (score %.2f/s)" % self.score
        return ("score %.2f/s, voiced %.0f%% of the time, pitch around %.0f Hz, "
                "longest unbroken stretch %.2f s"
                % (self.score, 100.0 * self.voiced_fraction,
                   self.median_f0_hz, self.longest_run_s))


class VoiceDetector:
    """A pitch-path detector: a lattice of pitch candidates, Viterbi, one score.

    Built for one sample rate; feed it any number of samples at that rate.
    Nothing is kept between calls -- each score is self-contained, so a
    calibration run and a measurement run cannot contaminate each other.
    """

    F0_MIN_HZ = 70.0             # below a deep male voice
    F0_MAX_HZ = 400.0            # above a high female voice
    FRAME_S = 0.032              # long enough for two periods at 70 Hz
    HOP_S = 0.010                # the usual speech frame rate

    # What a frame must reach, after whitening, before the voiced hypothesis
    # is better than the unvoiced one -- the unvoiced state constant
    # observation score, and so a threshold in exactly the likelihood sense.
    #
    # IT IS DERIVED, NOT CHOSEN. It was a flat 0.30 first, and that number
    # alone cost about 15 dB of sensitivity: picking the best of 190 lags in
    # pure noise already scores 0.15-0.17, so a voice quieter than the noise
    # never reaches 0.30 in any single frame and the whole segment scored
    # exactly zero. Measured: speech buried at -10 dB still reaches 0.34 at
    # its 90th percentile against the noise 0.23, so the information is
    # plainly there -- a threshold above both simply threw it away.
    #
    # After whitening the noise is white, so the level it reaches is a
    # property of the FRAME, not of the room: the normalised autocorrelation
    # at one lag is about N(0, 1/n_eff), and the best of n_lags of them lands
    # near sqrt(2 ln n_lags / n_eff). Measured against that prediction of
    # 0.176 -- white noise 0.153, band-limited hiss 0.151, this room 0.172,
    # this room with its machine hum running 0.151. So the theory is used and
    # the margin below is the only free number.
    # Measured against that prediction, 0.2 is where the sweep put it: the
    # useful threshold is well BELOW the level noise reaches, not above it.
    # At -10 dB the speech is a tenth of the power, so the correlation at the
    # true lag is only about 0.08 -- under the noise 0.176 -- and a frame
    # cannot see it at all. Only accumulation along a held lag can, and that
    # needs the threshold under the signal, not over the noise. Sweeping it
    # from 0.0 to 0.85 moved the reach from -5 dB to -10 dB, with everything
    # from 0.0 to 0.45 within a trial or two of each other.
    VOICED_MARGIN = 0.2
    # What a jump in pitch costs, per octave, and how far one frame may move
    # at all. Measured on the synthesised speech: a real voice moves a median
    # 0.014 octaves per 10 ms frame, 0.039 at its 90th percentile, so 0.08 is
    # generous to the voice and still cuts the lattice from 42 reachable
    # candidates per frame to 14 -- freedom noise was using to pick the best
    # of 42 fluctuations every frame. Worth stating plainly: tightening this
    # did NOT move the sensitivity (measured at 0.15/0.08/0.05/0.03 octaves
    # and 2.4/6/12 per octave, every combination within a trial of the
    # others), so it is kept because it is true of speech rather than because
    # it bought anything.
    PITCH_JUMP_COST = 2.4
    MAX_JUMP_OCTAVES = 0.08
    # What starting or stopping voicing costs, which imposes a duration: a
    # voiced stretch has to be long enough to pay its own entry fee out of
    # the periodicity it accumulates. A run is about 10 frames and earns
    # roughly 0.8 at -10 dB, against 0.9 for entering and leaving at 0.45,
    # which looked like the binding constraint. It is not: lowering it to
    # 0.2, 0.08, 0.03 and 0 made detection at -10 dB worse, not better
    # (50% -> 42% -> 33%), because a cheap state change lets NOISE flick in
    # and out of voicing wherever it happens to be favourable. Kept at 0.45.
    STATE_CHANGE_COST = 0.45

    def __init__(self, fs: float) -> None:
        if fs <= 2.0 * self.F0_MAX_HZ:
            raise ValueError("fs must be well above %g Hz, got %r" % (self.F0_MAX_HZ, fs))
        self.fs = float(fs)
        self.frame = int(round(self.FRAME_S * self.fs))
        self.hop = int(round(self.HOP_S * self.fs))
        self.lag_min = max(2, int(np.floor(self.fs / self.F0_MAX_HZ)))
        self.lag_max = int(np.ceil(self.fs / self.F0_MIN_HZ))
        if self.lag_max >= self.frame:
            raise ValueError("the frame is too short to hold a period at F0_MIN_HZ")
        self.lags = np.arange(self.lag_min, self.lag_max + 1)
        self.f0_of_lag = self.fs / self.lags
        window = np.hanning(self.frame)
        self._n_eff = window.sum() ** 2 / (window ** 2).sum()
        self._occupancy = 1.0
        self.VOICED_THRESHOLD = self.threshold_for(1.0)
        self._transition = self._build_transition()

    # -- the lattice ---------------------------------------------------------

    def _build_transition(self) -> np.ndarray:
        """Cost of moving from one pitch candidate to another, in log units.

        Minus infinity where the jump exceeds MAX_JUMP_OCTAVES, so the pass
        only ever looks at a band around the diagonal.
        """
        octaves = np.abs(np.log2(self.f0_of_lag[None, :] / self.f0_of_lag[:, None]))
        t = -self.PITCH_JUMP_COST * octaves
        t[octaves > self.MAX_JUMP_OCTAVES] = -np.inf
        return t

    # -- the observations ----------------------------------------------------

    def _frames(self, x: np.ndarray) -> np.ndarray:
        n = 1 + (x.size - self.frame) // self.hop
        if n < 1:
            return np.zeros((0, self.frame))
        idx = np.arange(self.frame)[None, :] + self.hop * np.arange(n)[:, None]
        return x[idx] * np.hanning(self.frame)[None, :]

    def threshold_for(self, occupancy: float) -> float:
        """The unvoiced score for a signal filling this fraction of the band.

        THE THRESHOLD DEPENDS ON THE BANDWIDTH, not just on the frame. A
        narrow band is genuinely more autocorrelated than a wide one -- fewer
        independent frequency bins means the autocorrelation decorrelates
        more slowly -- so noise in it reaches a higher periodicity for
        entirely innocent reasons. Measured on 20 s of this room, unvoiced
        throughout: full band scored 4.08, the same room filtered to
        80-7800 Hz scored 4.39, and filtered to a telephone band scored 8.29.
        Ranking a telephone band above a full one is backwards -- removing
        signal cannot add evidence -- and a fixed threshold does exactly
        that. Scaling the effective sample count by the occupied fraction of
        the spectrum puts them back on the same footing.
        """
        n_eff = max(self._n_eff * max(occupancy, 1e-3), 4.0)
        return self.VOICED_MARGIN * float(np.sqrt(2.0 * np.log(self.lags.size) / n_eff))

    def periodicity(self, x: np.ndarray) -> np.ndarray:
        """Normalised autocorrelation of the WHITENED signal, per frame per lag.

        Whitening is against the long-term average magnitude spectrum of this
        same segment, so what is measured is periodicity that the colour of
        the room does not already explain -- see the module docstring.
        """
        frames = self._frames(np.asarray(x, dtype=np.float64).reshape(-1))
        if frames.shape[0] == 0:
            return np.zeros((0, self.lags.size))
        nfft = 2 ** int(np.ceil(np.log2(2 * self.frame)))
        spec = np.fft.rfft(frames, nfft, axis=1)
        mag = np.abs(spec)
        average = np.mean(mag, axis=0)
        # AN EMPTY BIN IS DROPPED, NOT FLOORED. Dividing by a floor instead
        # was a real defect and it cost several dB: the band this runs on is
        # usually a BAND -- the listen band, or anything filtered -- and
        # outside it the average is essentially zero, so flooring turned bins
        # holding nothing but arithmetic into full-scale noise and fed it to
        # the autocorrelator. Measured on the same room and speech, band-
        # passed to 80-7800 Hz: the noise-only score rose from 3.16 to 3.39
        # and detection at -8 dB fell from 100% to nothing. Bins with no
        # signal contribute no evidence, so they contribute nothing.
        keep = average > 1e-9 + 1e-3 * float(average.max())
        white = np.zeros_like(spec)
        white[:, keep] = spec[:, keep] / average[None, keep]
        # what fraction of the spectrum actually holds anything: the
        # threshold is derived from it, see threshold_for
        self._occupancy = float(keep.mean())
        ac = np.fft.irfft(np.abs(white) ** 2, nfft, axis=1)[:, :self.lag_max + 1]
        zero = ac[:, :1]
        # A frame with no signal at all divides to nonsense; call it unvoiced
        # by handing back a flat zero instead.
        dead = (zero[:, 0] <= 1e-30)
        zero = np.where(zero <= 1e-30, 1.0, zero)
        # unbiased: the overlap shrinks with the lag, so short lags would
        # otherwise always look the most periodic
        taper = (self.frame - self.lags) / float(self.frame)
        r = (ac[:, self.lags] / zero) / taper[None, :]
        r[dead, :] = 0.0
        return np.clip(r, -1.0, 1.0)

    # -- the pass ------------------------------------------------------------

    def score(self, x: np.ndarray) -> SpeechEvidence:
        """Best voiced path against "nothing is voiced", as a rate per second."""
        r = self.periodicity(x)
        # the band this segment actually occupies decides the threshold
        self.VOICED_THRESHOLD = self.threshold_for(self._occupancy)
        n = r.shape[0]
        seconds = n * self.hop / self.fs
        if n < 3:
            return SpeechEvidence(0.0, 0.0, float("nan"), 0.0, n, seconds)

        n_lags = self.lags.size
        delta_v = r[0] - self.STATE_CHANGE_COST
        delta_u = np.float64(self.VOICED_THRESHOLD)
        back_v = np.zeros((n, n_lags), dtype=np.int32)
        back_v_from_u = np.zeros((n, n_lags), dtype=bool)
        back_u_voiced = np.zeros(n, dtype=bool)
        back_u_from = np.zeros(n, dtype=np.int32)

        for t in range(1, n):
            cand = delta_v[:, None] + self._transition
            best_from_v = np.max(cand, axis=0)
            arg_from_v = np.argmax(cand, axis=0)
            from_u = delta_u - self.STATE_CHANGE_COST
            take_u = from_u > best_from_v
            delta_v_next = np.where(take_u, from_u, best_from_v) + r[t]
            back_v[t] = arg_from_v
            back_v_from_u[t] = take_u
            k = int(np.argmax(delta_v))
            from_v = delta_v[k] - self.STATE_CHANGE_COST
            if from_v > delta_u:
                back_u_voiced[t] = True
                back_u_from[t] = k
                delta_u_next = from_v + self.VOICED_THRESHOLD
            else:
                delta_u_next = delta_u + self.VOICED_THRESHOLD
            delta_v, delta_u = delta_v_next, delta_u_next

        best = max(float(np.max(delta_v)), float(delta_u))
        # the null hypothesis: unvoiced at every frame, paying no transitions
        null = self.VOICED_THRESHOLD * n
        llr_per_second = (best - null) / max(seconds, 1e-12)

        path = self._backtrack(n, delta_v, delta_u, back_v, back_v_from_u,
                               back_u_voiced, back_u_from)
        voiced = path >= 0
        n_voiced = int(voiced.sum())
        median_f0 = float(np.median(self.f0_of_lag[path[voiced]])) if n_voiced else float("nan")
        return SpeechEvidence(llr_per_second, n_voiced / n, median_f0,
                              self._longest_run(voiced) * self.hop / self.fs, n, seconds)

    def _backtrack(self, n, delta_v, delta_u, back_v, back_v_from_u,
                   back_u_voiced, back_u_from) -> np.ndarray:
        """The chosen path, as a lag index per frame, -1 where unvoiced."""
        path = np.full(n, -1, dtype=np.int64)
        k = int(np.argmax(delta_v))
        voiced = float(delta_v[k]) >= float(delta_u)
        for t in range(n - 1, 0, -1):
            if voiced:
                path[t] = k
                if back_v_from_u[t, k]:
                    voiced = False
                else:
                    k = int(back_v[t, k])
            elif back_u_voiced[t]:
                voiced = True
                k = int(back_u_from[t])
        if voiced:
            path[0] = k
        return path

    @staticmethod
    def _longest_run(flags: np.ndarray) -> int:
        best = run = 0
        for f in flags:
            run = run + 1 if f else 0
            best = max(best, run)
        return best
