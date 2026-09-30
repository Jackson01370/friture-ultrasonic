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

"""One WAV segment on disk: written as it grows, readable if the app dies.

A WAV file states its own length in its header, and a program that is
killed never gets to write the final one. So the header is rewritten every
HEADER_REFRESH_S while the file grows, and a file found with a header that
disagrees with its size is REPAIRED from the size -- the samples are all
there, only the two length fields are stale. Everything written before a
crash is recovered except, at most, a fraction of a block.

Each WAV has a JSON sidecar beside it, written when the segment opens and
again when it closes. "state" is "open" while it is being written; a
segment still "open" that nobody is writing was interrupted, and is
repaired and marked "recovered".

The sidecar is where the time lives: when the first sample was captured,
how many frames there are, which unbroken run of the capture it belongs to
and where in that run it starts, so the segments of one run can be joined
end to end without a sample lost or doubled.
"""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

HEADER_BYTES = 44
BYTES_PER_SAMPLE = 2          # 16-bit: the UltraMic is 16-bit, and the float32
                              # the capture delivers sits exactly on its grid
HEADER_REFRESH_S = 5.0
# How long written frames may sit in this process before going to the OS.
# A killed process loses whatever it still holds, and the file buffer is
# 1 MB -- two seconds of audio at 250 kHz. Handing it over every half
# second bounds the loss to that; the OS keeps what it was given even when
# the process dies.
FLUSH_S = 0.5


def _header(fs: int, channels: int, data_bytes: int) -> bytes:
    block_align = channels * BYTES_PER_SAMPLE
    return b"".join((
        b"RIFF", struct.pack("<I", 36 + data_bytes), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, channels, fs, fs * block_align,
                             block_align, 8 * BYTES_PER_SAMPLE),
        b"data", struct.pack("<I", data_bytes),
    ))


def float_to_int16(block: np.ndarray) -> np.ndarray:
    """The capture float32 back to the 16-bit samples it came from.

    Exact, not an approximation: measured on the UltraMic, every captured
    value lies on the 1/32768 grid (largest distance from it 0), and
    multiplying a float32 by 32768 is exact. Clipping only matters for the
    +1.0 a float could hold and a 16-bit sample cannot.
    """
    return np.clip(np.rint(block * 32768.0), -32768, 32767).astype("<i2")


@dataclass
class SegmentInfo:
    """The sidecar: everything needed to place a segment in time."""

    wav: str                       # file name, beside the sidecar
    fs: int
    channels: int
    start_epoch: float             # wall-clock time of the first sample (UTC epoch)
    start_local: str               # the same, readable, local time
    run_id: str                    # which unbroken capture run this belongs to
    run_index: int                 # where in that run its first frame is
    n_frames: int = 0
    end_epoch: float | None = None
    state: str = "open"            # open / closed / recovered
    device: str = ""
    # the previous segment of the same run, if this one continues it with
    # no gap -- the join is exact because the run indices meet
    continues: str | None = None
    # what is known about a break before this segment, when there was one
    gap_before_s: float | None = None
    gap_reason: str | None = None
    protected: bool = False
    # How far the sample clock and the wall clock have drifted apart, parts
    # per million, measured from the START OF THE RUN to the end of this
    # segment -- not over the segment alone. Each end is a wall-clock
    # estimate good to a few tens of ms, which over one 10-minute file is
    # +-40 ppm of noise: measured, four files of one 32-minute recording
    # read -41, +13, -6 and +18 ppm while the whole 1130 s run read +5.4.
    # The longer the span, the smaller the noise, so the span is the run.
    clock_drift_ppm: float | None = None
    notes: list = field(default_factory=list)

    def save(self, folder: Path) -> None:
        path = folder / (Path(self.wav).stem + ".json")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "SegmentInfo":
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def segment_name(start_epoch: float) -> str:
    """2026-09-30_14-05-07.123 -- local time, sorts in time order."""
    t = datetime.fromtimestamp(start_epoch)
    return t.strftime("%Y-%m-%d_%H-%M-%S") + ".%03d" % int((start_epoch % 1) * 1000)


class WavSegmentWriter:
    """Append 16-bit frames to one WAV, keeping its header honest as it goes."""

    def __init__(self, folder: Path, info: SegmentInfo) -> None:
        self.folder = Path(folder)
        self.info = info
        self.path = self.folder / info.wav
        self._f = open(self.path, "wb", buffering=1 << 20)
        self._f.write(_header(info.fs, info.channels, 0))
        self._data_bytes = 0
        self._last_refresh = time.monotonic()
        self._last_flush = self._last_refresh
        info.save(self.folder)

    @property
    def n_frames(self) -> int:
        return self._data_bytes // (self.info.channels * BYTES_PER_SAMPLE)

    @property
    def file_bytes(self) -> int:
        return HEADER_BYTES + self._data_bytes

    def write(self, frames_int16: np.ndarray) -> None:
        data = np.ascontiguousarray(frames_int16, dtype="<i2").tobytes()
        self._f.write(data)
        self._data_bytes += len(data)
        now = time.monotonic()
        if now - self._last_refresh >= HEADER_REFRESH_S:
            self.refresh_header()
            self._last_refresh = self._last_flush = now
        elif now - self._last_flush >= FLUSH_S:
            self._f.flush()
            self._last_flush = now

    def refresh_header(self) -> None:
        """Make the file on disk say how long it is now.

        Flushes to the OS as well, which is what survives the process being
        killed; surviving a power cut would need an fsync per refresh.
        """
        self._f.flush()
        pos = self._f.tell()
        self._f.seek(0)
        self._f.write(_header(self.info.fs, self.info.channels, self._data_bytes))
        self._f.seek(pos)
        self._f.flush()

    def close(self, end_epoch: float, run_start: tuple[float, int] | None = None) -> SegmentInfo:
        """Finish the file. run_start is (wall time, run index) of the run's first frame."""
        self.refresh_header()
        os.fsync(self._f.fileno())
        self._f.close()
        info = self.info
        info.n_frames = self.n_frames
        info.end_epoch = end_epoch
        info.state = "closed"
        t0, i0 = run_start if run_start is not None else (info.start_epoch, info.run_index)
        span = end_epoch - t0
        nominal = (info.run_index + info.n_frames - i0) / info.fs
        if nominal > 300.0:
            info.clock_drift_ppm = 1e6 * (span - nominal) / nominal
        info.save(self.folder)
        return info


def read_header(path: Path) -> tuple[int, int, int]:
    """(fs, channels, data bytes the header claims) of a WAV this module wrote."""
    with open(path, "rb") as f:
        h = f.read(HEADER_BYTES)
    if len(h) < HEADER_BYTES or h[:4] != b"RIFF" or h[8:12] != b"WAVE":
        raise ValueError("%s is not a WAV this recorder wrote" % path)
    channels, fs = struct.unpack("<HI", h[22:28])
    data_bytes = struct.unpack("<I", h[40:44])[0]
    return fs, channels, data_bytes


def repair(folder: Path, wav_name: str) -> SegmentInfo | None:
    """Make an interrupted segment readable and say how much it holds.

    The frame count is taken from the FILE SIZE, not the header -- the
    header is what a crash leaves stale. A trailing partial frame is cut
    off. The sidecar, if the crash left one, keeps its start time; without
    one the start is taken from the file name.
    """
    folder = Path(folder)
    wav = folder / wav_name
    fs, channels, _ = read_header(wav)
    frame_bytes = channels * BYTES_PER_SAMPLE
    size = wav.stat().st_size
    data_bytes = ((size - HEADER_BYTES) // frame_bytes) * frame_bytes
    with open(wav, "r+b") as f:
        f.truncate(HEADER_BYTES + data_bytes)
        f.seek(0)
        f.write(_header(fs, channels, data_bytes))
    side = folder / (wav.stem + ".json")
    if side.exists():
        info = SegmentInfo.load(side)
    else:
        start = datetime.strptime(wav.stem[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
        info = SegmentInfo(wav=wav.name, fs=fs, channels=channels, start_epoch=start,
                           start_local=wav.stem, run_id="unknown", run_index=0)
    info.n_frames = data_bytes // frame_bytes
    info.end_epoch = info.start_epoch + info.n_frames / fs
    info.state = "recovered"
    info.notes = list(info.notes) + ["repaired after an interrupted write"]
    info.save(folder)
    return info
