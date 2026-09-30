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

"""Play a folder of recordings back through the application as if it were live.

    RecordingTimeline   the segments on one wall-clock axis: where each one
                        starts, how long it is, which ones join seamlessly
                        and where the gaps are
    ReplaySource        a position on that axis that advances in real time
                        (or 2x, 4x ...) and hands out capture-sized blocks

The source knows nothing about the application. AudioBackend asks it for
blocks in place of the microphone's, so every dock -- spectrum,
spectrogram, Band Survey, Digital Decode, Listen, the level meters -- sees
the recording exactly as it saw the room. The microphone keeps running
underneath and the continuous recorder keeps writing it.

WHAT A BLOCK MAY NOT DO is span a break in the recording. Across a planned
split (one file continues the last, the run indices meet) the source reads
straight on: the two files are one stream. Across a real gap -- a stop, a
crash, a device change -- it drops the few frames that would not fill a
block, moves to the next file, and says so ("jumped"), so the caller can
start the docks afresh instead of letting them average two different
moments together.

THE FILE BEING WRITTEN is part of the timeline. Its length is taken from
its size on disk, not its sidecar (which says 0 until it closes), so
replaying near the present follows the recording as it grows, like a
video player catching up with a live stream.

Pacing is by a clock, not by how often the caller asks: blocks become due
at speed x the sample rate. After a stall no more than MAX_CATCHUP_S of
audio is handed over at once -- the rest is simply skipped over in time --
so a busy moment does not turn into a burst that the docks then choke on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from friture.recording.store import SegmentStore
from friture.recording.wav_segment import BYTES_PER_SAMPLE, HEADER_BYTES, SegmentInfo


@dataclass
class TimelineEntry:
    info: SegmentInfo
    path: Path
    n_frames: int
    growing: bool                 # still being written

    @property
    def start(self) -> float:
        return self.info.start_epoch

    @property
    def end(self) -> float:
        return self.info.start_epoch + self.n_frames / self.info.fs


class RecordingTimeline:
    """The segments of one folder, in time order, as one axis."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.entries: list[TimelineEntry] = []
        self.fs = 0
        self.refresh()

    def refresh(self) -> None:
        entries = []
        for info in SegmentStore(self.folder).segments():
            path = self.folder / info.wav
            try:
                size = path.stat().st_size
            except OSError:
                continue
            frames = max((size - HEADER_BYTES) // (info.channels * BYTES_PER_SAMPLE), 0)
            growing = info.state == "open"
            n = frames if growing else min(info.n_frames, frames)
            if n <= 0:
                continue
            entries.append(TimelineEntry(info, path, n, growing))
        self.entries = entries
        if entries:
            self.fs = entries[0].info.fs

    @property
    def empty(self) -> bool:
        return not self.entries

    @property
    def span(self) -> tuple[float, float]:
        if not self.entries:
            return 0.0, 0.0
        return self.entries[0].start, max(e.end for e in self.entries)

    def continues(self, i: int) -> bool:
        """True if entry i+1 carries on from entry i with no gap."""
        if i + 1 >= len(self.entries):
            return False
        a, b = self.entries[i].info, self.entries[i + 1].info
        return b.run_id == a.run_id and b.run_index == a.run_index + self.entries[i].n_frames

    def locate(self, epoch: float) -> tuple[int, int]:
        """(entry, frame) at a wall-clock time; a time in a gap goes to what follows it."""
        if not self.entries:
            raise ValueError("the timeline is empty")
        for i, e in enumerate(self.entries):
            if epoch < e.start:
                return i, 0
            if epoch < e.end:
                return i, int((epoch - e.start) * e.info.fs)
        last = len(self.entries) - 1
        return last, self.entries[last].n_frames

    def epoch_at(self, i: int, frame: int) -> float:
        e = self.entries[i]
        return e.start + frame / e.info.fs

    def index_of(self, wav: str) -> int | None:
        for i, e in enumerate(self.entries):
            if e.info.wav == wav:
                return i
        return None

    def ranges(self) -> list[tuple[float, float]]:
        """Wall-clock stretches that hold audio, joined where files continue each other."""
        out: list[list[float]] = []
        for i, e in enumerate(self.entries):
            if out and i > 0 and self.continues(i - 1):
                out[-1][1] = e.end
            else:
                out.append([e.start, e.end])
        return [(a, b) for a, b in out]


class ReplaySource:
    """A playing position on a RecordingTimeline, handing out blocks as they fall due."""

    MAX_CATCHUP_S = 0.5
    REFRESH_S = 2.0

    def __init__(self, timeline: RecordingTimeline, block: int, clock=time.monotonic) -> None:
        self.timeline = timeline
        self.block = int(block)
        self._clock = clock
        self.speed = 1.0
        self.playing = False
        self._i = 0
        self._frame = 0
        self._credit = 0.0
        self._last = clock()
        self._last_refresh = self._last
        self.at_end = False
        # measured, for "is it keeping up at this speed"
        self._delivered = 0
        self._played_since = self._last
        self._handle = None
        self._handle_path: Path | None = None

    # -- control -------------------------------------------------------------------

    def seek(self, epoch: float) -> None:
        self.timeline.refresh()
        if self.timeline.empty:
            return
        self._i, self._frame = self.timeline.locate(epoch)
        self._credit = 0.0
        self.at_end = False

    def play(self) -> None:
        if not self.playing:
            self.playing = True
            self._last = self._clock()
            self._credit = 0.0
            self._delivered = 0
            self._played_since = self._last

    def pause(self) -> None:
        self.playing = False

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, float(speed))
        self._delivered = 0
        self._played_since = self._clock()

    @property
    def position(self) -> float:
        if self.timeline.empty:
            return 0.0
        return self.timeline.epoch_at(self._i, self._frame)

    @property
    def current_wav(self) -> str:
        return self.timeline.entries[self._i].info.wav if not self.timeline.empty else ""

    @property
    def achieved_speed(self) -> float:
        """How fast the audio actually went out, measured, over the last stretch."""
        span = self._clock() - self._played_since
        if span < 1.0 or not self.timeline.fs:
            return self.speed
        return self._delivered / self.timeline.fs / span

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._handle_path = None

    # -- delivery ------------------------------------------------------------------

    def fetch(self) -> list[tuple[np.ndarray, bool]]:
        """The blocks due now, each with whether a gap was jumped just before it."""
        now = self._clock()
        if now - self._last_refresh >= self.REFRESH_S:
            self._last_refresh = now
            self._refresh_keeping_place()
        if not self.playing or self.timeline.empty:
            self._last = now
            return []
        fs = self.timeline.fs
        self._credit += (now - self._last) * self.speed * fs
        self._last = now
        self._credit = min(self._credit, self.MAX_CATCHUP_S * self.speed * fs + self.block)
        out = []
        jumped = False
        while self._credit >= self.block:
            got = self._read_block()
            if got is None:
                break
            block, jump = got
            jumped = jumped or jump
            out.append((block, jumped))
            jumped = False
            self._credit -= self.block
            self._delivered += self.block
        if not out:
            self._credit = min(self._credit, float(self.block))
        return out

    def _refresh_keeping_place(self) -> None:
        wav = self.current_wav
        self.timeline.refresh()
        i = self.timeline.index_of(wav) if wav else None
        if i is not None:
            self._i = i
        elif not self.timeline.empty:
            # the file we were in has gone (rotation deleted it): carry on
            # from wherever that moment now falls
            self._i = min(self._i, len(self.timeline.entries) - 1)
            self._frame = 0

    def _read_block(self):
        """One block from the current position, or None if there is nothing yet.

        When there is not a whole block yet -- the present has been reached
        in the file being written -- the position goes back to where this
        call started, which may be in the PREVIOUS file if the block had
        already crossed into this one. Rewinding by the frames read, within
        the current file, would go negative in exactly that case.
        """
        start = (self._i, self._frame)
        parts, need, jumped = [], self.block, False
        tl = self.timeline
        while need:
            e = tl.entries[self._i]
            avail = e.n_frames - self._frame
            if avail > 0:
                take = min(avail, need)
                data = self._read(e, self._frame, take)
                if data is None:                     # the file vanished under us
                    return self._skip_missing()
                parts.append(data)
                self._frame += take
                need -= take
                continue
            # this file is used up
            if self._i + 1 >= len(tl.entries):
                # following the present: the file has probably grown since the
                # last full refresh, and a stat is cheap
                if e.growing and self._grow(e):
                    continue
                # the present (a file still being written: wait for more) or
                # the true end; either way no stub block goes out
                self._i, self._frame = start
                if not e.growing:
                    self.at_end = True
                    self.playing = False
                return None
            if tl.continues(self._i):
                self._i += 1
                self._frame = 0
                continue
            # a real gap: never stitch across it
            self._i += 1
            self._frame = 0
            parts, need, jumped = [], self.block, True
        block = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return block, jumped

    @staticmethod
    def _grow(e: TimelineEntry) -> bool:
        """Re-measure a file being written; True if it holds more than we knew."""
        try:
            size = e.path.stat().st_size
        except OSError:
            return False
        frames = max((size - HEADER_BYTES) // (e.info.channels * BYTES_PER_SAMPLE), 0)
        if frames > e.n_frames:
            e.n_frames = frames
            return True
        return False

    def _read(self, e: TimelineEntry, frame: int, count: int):
        ch = e.info.channels
        try:
            if self._handle_path != e.path:
                self.close()
                self._handle = open(e.path, "rb")
                self._handle_path = e.path
            self._handle.seek(HEADER_BYTES + frame * ch * BYTES_PER_SAMPLE)
            raw = self._handle.read(count * ch * BYTES_PER_SAMPLE)
        except OSError:
            self.close()
            return None
        data = np.frombuffer(raw, dtype="<i2")
        if data.size < count * ch:
            return None
        return data.reshape(-1, ch)

    def _skip_missing(self):
        self.close()
        self._refresh_keeping_place()
        return None
