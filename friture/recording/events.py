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

"""What happened, and when: events found in the analysis logs.

The logs say what every second, every 5 s and every minute held. This turns
them into a short list of moments worth going back to. Everything is judged
against THE ROOM AROUND IT -- a baseline taken from the same recording a
few minutes either side -- because "loud" or "voice-like" mean nothing in
absolute numbers: this microphone's floor, this room's hum and this
detector's resting score set them.

  loud       a band rose LOUD_DB over its own median of the surrounding
             LOUD_BASELINE_S. Bands that rise together are one event.
  voice      the voice score rose VOICE_MARGIN over the room's resting
             score. Measured on speech played into this room at known
             levels, 5 s windows: the room rests at 2.7-3.3; speech at
             +5 dB SNR scores 10-12, at -1 dB 8-9, at -3 dB 7-8, at -9 dB
             4-7. Speech holds it for window after window; a knock lifted
             ONE window, to 7.0. So an event of two windows or more is
             "sustained", one window alone is "brief" and may be a knock --
             and neither proves a voice: a buzzer scores like one.
  line_on    a line at least LINE_DB over its neighbourhood that was not
  line_off   there in the minutes before (or is gone in the minutes after),
             and stays that way for two minutes -- one minute alone is a
             line wobbling across the top-16 cut, not something switching.

Pure functions over records; no files, no Qt. friture.recording.event_index
reads the logs and keeps the results up to date.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from friture.recording.analysis import LEVEL_BANDS

LOUD_DB = 10.0
LOUD_BASELINE_S = 600.0          # the median is taken over +-this
LOUD_MERGE_GAP_S = 2.0
LOUD_MIN_HISTORY = 30            # seconds of levels needed before judging
VOICE_MARGIN = 2.0
VOICE_BASELINE_S = 1800.0        # the room's resting score from +-this
VOICE_BASELINE_PERCENTILE = 20.0 # low, so a long stretch of talk does not
                                 # raise the bar it is measured against
VOICE_MIN_HISTORY = 12           # windows
LINE_DB = 12.0
LINE_MAX_GAP_S = 600.0           # minutes further apart than this are not compared


@dataclass
class Event:
    kind: str                    # loud / voice / line_on / line_off
    start: float                 # epoch of the first moment
    end: float
    seg: str                     # segment the event starts in, and where
    pos: int
    strength: float              # dB over the room, or score over its rest
    detail: str
    brief: bool = False
    bands: list = field(default_factory=list)
    freq_hz: float | None = None

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def _band_name(lo: float, hi: float) -> str:
    def f(v):
        return "%g kHz" % (v / 1000) if v >= 1000 else "%g Hz" % v
    return "%s-%s" % (f(lo), f(hi))


def _rolling(times: np.ndarray, values: np.ndarray, half: float, fn, min_n: int,
             step: float = 30.0) -> np.ndarray:
    """fn over values within +-half seconds of each time; nan where too few.

    Evaluated every `step` seconds and interpolated between: a baseline over
    +-10 minutes moves slowly by construction, and computing it at every
    one of a day's 86,400 level records took 2 s per band. Measured: 32
    minutes of logs took 0.27 s the exact way, which is 14 s for the 28
    hours a 50 GB folder holds.
    """
    n = times.size
    if n == 0:
        return np.full(values.shape, np.nan)
    grid = np.arange(times[0], times[-1] + step, step)
    at = np.full(grid.shape, np.nan)
    lo_idx = np.searchsorted(times, grid - half, side="left")
    hi_idx = np.searchsorted(times, grid + half, side="right")
    for k, (lo, hi) in enumerate(zip(lo_idx, hi_idx)):
        if hi - lo >= min_n:
            at[k] = fn(values[lo:hi])
    ok = ~np.isnan(at)
    if not ok.any():
        return np.full(values.shape, np.nan)
    out = np.interp(times, grid[ok], at[ok])
    # no extrapolating a baseline into stretches that had too little around them
    near = np.interp(times, grid, ok.astype(float)) > 0.0
    out[~near] = np.nan
    return out


def _merge(spans, gap):
    """Join (start, end, payload) spans closer than gap; payloads collected."""
    spans = sorted(spans, key=lambda s: s[0])
    out = []
    for s in spans:
        if out and s[0] - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], s[1])
            out[-1][2].append(s[2])
        else:
            out.append([s[0], s[1], [s[2]]])
    return out


# -- loud ------------------------------------------------------------------------

def loud_events(levels: list[dict]) -> list[Event]:
    if not levels:
        return []
    levels = sorted(levels, key=lambda r: r["t"])
    t = np.array([r["t"] for r in levels])
    spans = []
    for b, (lo, hi) in enumerate(LEVEL_BANDS):
        v = np.array([r["bands_dbfs"][b] for r in levels], dtype=float)
        base = _rolling(t, v, LOUD_BASELINE_S, np.median, LOUD_MIN_HISTORY)
        over = v - base
        for i in np.flatnonzero(over >= LOUD_DB):
            r = levels[i]
            spans.append((r["t"], r["t"] + r["dur"], (b, float(over[i]), r)))
    events = []
    for start, end, parts in _merge(spans, LOUD_MERGE_GAP_S):
        bands = sorted({p[0] for p in parts})
        peak = max(p[1] for p in parts)
        first = min((p[2] for p in parts), key=lambda r: r["t"])
        names = [_band_name(*LEVEL_BANDS[b]) for b in bands]
        span_txt = (names[0] if len(names) == 1
                    else "%s to %s" % (_band_name(*LEVEL_BANDS[bands[0]]).split("-")[0],
                                       _band_name(*LEVEL_BANDS[bands[-1]]).split("-")[1]))
        events.append(Event("loud", start, end, first["seg"], first["pos"], round(peak, 1),
                            "%s up to %+.0f dB over the room, %.0f s" % (span_txt, peak, end - start),
                            bands=names))
    return events


# -- voice -----------------------------------------------------------------------

def voice_events(voice: list[dict]) -> list[Event]:
    if not voice:
        return []
    voice = sorted(voice, key=lambda r: r["t"])
    t = np.array([r["t"] for r in voice])
    s = np.array([r["score"] for r in voice], dtype=float)
    base = _rolling(t, s, VOICE_BASELINE_S,
                    lambda v: float(np.percentile(v, VOICE_BASELINE_PERCENTILE)), VOICE_MIN_HISTORY)
    if np.all(np.isnan(base)):
        # too little recorded to know the room: judge against what there is
        base = np.full(s.shape, np.percentile(s, VOICE_BASELINE_PERCENTILE))
    base = np.where(np.isnan(base), np.nanmedian(base), base)
    hot = s - base >= VOICE_MARGIN
    events = []
    i = 0
    while i < len(voice):
        if not hot[i]:
            i += 1
            continue
        j = i
        # consecutive windows of the same run, back to back
        while (j + 1 < len(voice) and hot[j + 1] and voice[j + 1]["run"] == voice[j]["run"]
               and voice[j + 1]["idx"] == voice[j]["idx"] + voice[j].get("n", 0)):
            j += 1
        run = voice[i:j + 1]
        peak = float(np.max(s[i:j + 1] - base[i:j + 1]))
        rest = float(np.median(base[i:j + 1]))
        f0 = [r["f0_hz"] for r in run if r.get("f0_hz")]
        brief = len(run) == 1
        detail = ("voice-like for %.0f s: score %.1f against the room's %.1f%s"
                  % (run[-1]["t"] + run[-1]["dur"] - run[0]["t"], float(np.max(s[i:j + 1])), rest,
                     ", pitch around %.0f Hz" % np.median(f0) if f0 else ""))
        if brief:
            detail += " -- one window only, may be a knock"
        events.append(Event("voice", run[0]["t"], run[-1]["t"] + run[-1]["dur"], run[0]["seg"],
                            run[0]["pos"], round(peak, 2), detail, brief=brief))
        i = j + 1
    return events


# -- lines -----------------------------------------------------------------------

def _near(f: float, fs: list[float]) -> bool:
    tol = max(15.0, 0.002 * f)
    return any(abs(f - g) <= tol for g in fs)


def line_events(lines: list[dict]) -> list[Event]:
    lines = sorted(lines, key=lambda r: r["t"])
    minutes = [(r, [l[0] for l in r["lines"]], [l[0] for l in r["lines"] if l[1] >= LINE_DB])
               for r in lines]
    events = []

    def comparable(a, b):
        return abs(a[0]["t"] - b[0]["t"]) <= LINE_MAX_GAP_S + a[0]["dur"]

    for m in range(2, len(minutes) - 1):
        prev2, prev1, cur, nxt = minutes[m - 2], minutes[m - 1], minutes[m], minutes[m + 1]
        if not (comparable(prev2, prev1) and comparable(prev1, cur) and comparable(cur, nxt)):
            continue
        r = cur[0]
        for f in cur[2]:
            # strong now and in the next minute; nowhere in the two before
            if _near(f, nxt[1]) and not _near(f, prev1[1]) and not _near(f, prev2[1]):
                ex = max(l[1] for l in r["lines"] if abs(l[0] - f) < 1e-6)
                events.append(Event("line_on", r["t"], r["t"] + r["dur"], r["seg"], r["pos"], ex,
                                    "a line appeared at %.3f kHz, %+.0f dB over its neighbourhood"
                                    % (f / 1000, ex), freq_hz=f))
        for f in prev1[2]:
            # strong in the two minutes before; nowhere now nor in the next
            if _near(f, prev2[2]) and not _near(f, cur[1]) and not _near(f, nxt[1]):
                ex = max(l[1] for l in prev1[0]["lines"] if abs(l[0] - f) < 1e-6)
                events.append(Event("line_off", r["t"], r["t"] + r["dur"], r["seg"], r["pos"], ex,
                                    "the line at %.3f kHz (%+.0f dB) went away" % (f / 1000, ex),
                                    freq_hz=f))
    return events


def detect(records: list[dict]) -> list[Event]:
    """Every event in a set of analysis records, in time order."""
    by = {"levels": [], "voice": [], "lines": []}
    for r in records:
        if r.get("kind") in by:
            by[r["kind"]].append(r)
    out = loud_events(by["levels"]) + voice_events(by["voice"]) + line_events(by["lines"])
    return sorted(out, key=lambda e: e.start)
