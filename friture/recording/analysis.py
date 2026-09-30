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

"""What was in the recording, written down as it is recorded.

Beside every WAV segment sits <stem>.analysis.jsonl, one JSON record per
line, so that later there is a list of WHEN and WHERE something happened to
jump to, instead of 28 hours of audio to scroll through. Three analyses run,
whatever docks are open -- if the log depended on the layout, two days
could not be compared:

  levels  every LEVEL_S (1 s): broadband RMS and peak, and the RMS of each
          band in LEVEL_BANDS. What got louder, and where.
  voice   every VOICE_S (5 s): the pitch-path voice detector of
          friture.demod.speech, at 16 kHz on 80-7800 Hz. Raw score, voiced
          share, pitch, longest unbroken voiced stretch -- the numbers, not a
          verdict: whether a score is unusual can only be judged against the
          room around it, which the event list does with the whole day in
          hand. Its two known limits hold here too: about 0 dB SNR at best,
          and a buzzer scores like a voice.
  lines   every LINES_S (60 s): the Band Survey over that minute -- lines
          standing out of their own neighbourhood, with steadiness across
          the minute. A line that appears, moves or goes shows up as a
          difference between two consecutive minutes.

Every record carries where its window STARTED: the segment file ("seg"),
the frame offset in it ("pos"), the capture run and index in it, and the
wall-clock time; and its exact length in frames ("n"). A window that runs
over the end of a file into the next one (a planned split, the run goes
on) is written to the file it began in.

Records are written in the order their windows COMPLETE, not the order they
started: a level window and a voice window finishing in the same piece of
audio can come out either way round depending on how the audio was cut
into pieces. The set of records is the same whatever the cutting; a reader
that wants time order sorts by "idx".

A window is never stitched across a break in the capture. At a break the
windows in progress are cut short and written if they are long enough to
mean something (MIN_FRACTION of their length), with their real duration,
and everything starts again on the other side.

This module has no threads and no files: feed() takes audio and returns
records. That is what lets the same code run live in the recorder and
offline over old recordings with identical results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from friture.demod.speech import VoiceDetector
from friture.demod.survey import SpectrumSurvey

LOG_VERSION = 1
LEVEL_S = 1.0
VOICE_S = 5.0
LINES_S = 60.0
LEVEL_BANDS = ((100, 1_000), (1_000, 4_000), (4_000, 10_000), (10_000, 20_000),
               (20_000, 30_000), (30_000, 60_000), (60_000, 125_000))
VOICE_FS = 16_000
VOICE_BAND = (80.0, 7_800.0)
LINES_TOP = 16
# A window cut short by a break is still written if it got this far.
MIN_FRACTION = 0.4


@dataclass(frozen=True)
class Where:
    """Where the first frame of a piece of audio lives."""

    seg: str            # segment file stem
    pos: int            # frame offset in that segment
    run: str            # capture run key
    idx: int            # frame index in that run
    t: float            # wall-clock time of the frame (epoch seconds)

    def advanced(self, frames: int, fs: int) -> "Where":
        return Where(self.seg, self.pos + frames, self.run, self.idx + frames, self.t + frames / fs)


def header(fs: int) -> dict:
    return {"kind": "header", "version": LOG_VERSION, "fs": fs,
            "level_s": LEVEL_S, "voice_s": VOICE_S, "lines_s": LINES_S,
            "level_bands": [list(b) for b in LEVEL_BANDS],
            "voice_fs": VOICE_FS, "voice_band": list(VOICE_BAND),
            "lines_fields": ["hz", "excess_db", "level_db", "spread_db", "width_hz"]}


class _Window:
    """Collects frames until it holds `length` of them."""

    def __init__(self, length: int) -> None:
        self.length = length
        self.parts: list[np.ndarray] = []
        self.n = 0
        self.start: Where | None = None

    def add(self, x: np.ndarray, where: Where) -> np.ndarray:
        """Take what fits; return the rest."""
        if self.start is None:
            self.start = where
        take = min(self.length - self.n, x.size)
        self.parts.append(x[:take])
        self.n += take
        return x[take:]

    @property
    def full(self) -> bool:
        return self.n >= self.length

    def take(self) -> tuple[np.ndarray, Where]:
        data = np.concatenate(self.parts) if len(self.parts) > 1 else self.parts[0]
        start = self.start
        self.parts, self.n, self.start = [], 0, None
        return data, start


def _db(x: float) -> float:
    return round(10.0 * np.log10(max(x, 1e-20)), 2)


class LiveAnalysis:
    """Audio in, log records out. Channel 0 of whatever is fed."""

    def __init__(self, fs: int) -> None:
        self.fs = int(fs)
        self._voice = VoiceDetector(float(VOICE_FS))
        self._level = _Window(int(round(LEVEL_S * self.fs)))
        self._voice_w = _Window(int(round(VOICE_S * self.fs)))
        self._survey = self._new_survey()
        self._survey_start: Where | None = None
        self._survey_n = 0
        self._expect: tuple[str, int] | None = None
        freqs = np.fft.rfftfreq(self._level.length, 1.0 / self.fs)
        self._band_masks = [(freqs >= lo) & (freqs < hi) for lo, hi in LEVEL_BANDS]

    def _new_survey(self) -> SpectrumSurvey:
        # steadiness from every 8th window keeps a minute in 57 windows
        return SpectrumSurvey(float(self.fs), history=64, keep_every=8)

    # -- feeding ---------------------------------------------------------------

    def feed(self, data: np.ndarray, where: Where) -> list[dict]:
        """Take the next frames; float in [-1, 1) or int16."""
        x = np.asarray(data)
        if x.ndim > 1:
            x = x[:, 0]
        x = (x.astype(np.float64) / 32768.0) if x.dtype == np.int16 else x.astype(np.float64)
        out: list[dict] = []
        if self._expect is not None and self._expect != (where.run, where.idx):
            out += self.flush()
        self._expect = (where.run, where.idx + x.size)
        if x.size == 0:
            return out

        # levels and voice: fixed windows, back to back
        for w, finish in ((self._level, self._finish_level), (self._voice_w, self._finish_voice)):
            rest, at = x, where
            while rest.size:
                before = rest.size
                rest = w.add(rest, at)
                at = at.advanced(before - rest.size, self.fs)
                if w.full:
                    out.append(finish(*w.take(), partial=False))

        # lines: the survey accumulates; a minute is counted in frames
        rest, at = x, where
        while rest.size:
            if self._survey_start is None:
                self._survey_start = at
            take = min(int(round(LINES_S * self.fs)) - self._survey_n, rest.size)
            self._survey.process(rest[:take])
            self._survey_n += take
            at = at.advanced(take, self.fs)
            rest = rest[take:]
            if self._survey_n >= int(round(LINES_S * self.fs)):
                out.append(self._finish_lines(partial=False))
        return out

    def flush(self) -> list[dict]:
        """A break: write the windows in progress if they are long enough, then start over."""
        out = []
        for w, finish in ((self._level, self._finish_level), (self._voice_w, self._finish_voice)):
            if w.n >= MIN_FRACTION * w.length:
                out.append(finish(*w.take(), partial=True))
            else:
                w.take() if w.n else None
        if self._survey_n >= MIN_FRACTION * LINES_S * self.fs and self._survey.ready:
            out.append(self._finish_lines(partial=True))
        else:
            self._survey = self._new_survey()
            self._survey_start, self._survey_n = None, 0
        self._expect = None
        return out

    # -- the three analyses ------------------------------------------------------

    def _base(self, kind: str, where: Where, frames: int, partial: bool) -> dict:
        # "n" is the exact length in frames; "dur" is the same, rounded, for
        # reading. Anything that places a window uses n: rebuilt from dur, a
        # 151,872-frame remnant came back as 151,875 and appeared to run 3
        # frames past the end of its run.
        rec = {"kind": kind, "t": round(where.t, 3), "seg": where.seg, "pos": where.pos,
               "run": where.run, "idx": where.idx, "n": int(frames),
               "dur": round(frames / self.fs, 4)}
        if partial:
            rec["partial"] = True
        return rec

    def _finish_level(self, x: np.ndarray, where: Where, partial: bool) -> dict:
        rec = self._base("levels", where, x.size, partial)
        ms = float(np.mean(x * x))
        rec["rms_dbfs"] = _db(ms)
        rec["peak_dbfs"] = _db(float(np.max(np.abs(x))) ** 2)
        if x.size == self._level.length:
            masks = self._band_masks
            n = x.size
        else:
            freqs = np.fft.rfftfreq(x.size, 1.0 / self.fs)
            masks = [(freqs >= lo) & (freqs < hi) for lo, hi in LEVEL_BANDS]
            n = x.size
        p = np.abs(np.fft.rfft(x)) ** 2
        # Parseval: mean square = (|X0|^2 + 2 sum |Xk|^2 ...) / n^2
        rec["bands_dbfs"] = [_db(2.0 * float(p[m].sum()) / (n * n)) for m in masks]
        return rec

    def _finish_voice(self, x: np.ndarray, where: Where, partial: bool) -> dict:
        rec = self._base("voice", where, x.size, partial)
        low = resample_poly(x, VOICE_FS, self.fs) if self.fs != VOICE_FS else x
        n = 2 ** int(np.ceil(np.log2(max(low.size, 2))))
        spec = np.fft.rfft(low, n)
        f = np.fft.rfftfreq(n, 1.0 / VOICE_FS)
        spec[(f < VOICE_BAND[0]) | (f > VOICE_BAND[1])] = 0.0
        band = np.fft.irfft(spec, n)[:low.size]
        e = self._voice.score(band)
        rec["score"] = round(float(e.score), 4)
        rec["voiced"] = round(float(e.voiced_fraction), 3)
        rec["f0_hz"] = round(float(e.median_f0_hz), 1) if np.isfinite(e.median_f0_hz) else None
        rec["longest_s"] = round(float(e.longest_run_s), 3)
        return rec

    def _finish_lines(self, partial: bool) -> dict:
        rec = self._base("lines", self._survey_start, self._survey_n, partial)
        found = self._survey.lines(100.0, self.fs / 2.0, top=LINES_TOP) if self._survey.ready else []
        rec["lines"] = [[round(l.frequency_hz, 1), round(l.excess_db, 1), round(l.level_db, 1),
                         round(l.spread_db, 1), round(l.extent_hz, 1)] for l in found]
        self._survey = self._new_survey()
        self._survey_start, self._survey_n = None, 0
        return rec
