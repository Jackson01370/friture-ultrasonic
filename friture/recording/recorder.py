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

"""The drive recorder: every captured block, to disk, on its own thread.

    capture thread --push()--> queue --writer thread--> WAV segments

push() runs on the capture thread (see AudioBackend.add_raw_sink) and only
converts the block to 16 bits and queues it; it never waits. The writer
thread owns every file. A slow disk therefore backs up the queue -- QUEUE_S
of audio, about 30 MB -- and not the capture. If even that fills, blocks
are dropped, counted, and the break is written down: the segment ends there
and the next one records how much was missing and why. A recording that
admits a gap can be trusted; one that splices across it cannot.

WHERE A SEGMENT ENDS

  - every SEGMENT_S of samples, exactly -- the next one continues the same
    run from the very next frame, and says so ("continues");
  - wherever the capture's run breaks: a stop and start, a device change,
    a record action that died, blocks the disk could not keep up with, an
    overflow the driver reported;
  - after IDLE_CLOSE_S with nothing arriving, so a paused capture leaves a
    closed, finished file rather than one held open for hours.

Times are wall-clock estimates of each segment's first sample, taken from
the capture thread (see AudioBackend._drain); the sample count within a
run is exact. Their disagreement over a closed segment is kept as
clock_drift_ppm, because over a 28-hour recording a crystal a few tens of
ppm off moves "when did it happen" by seconds.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from friture.recording.analysis import LiveAnalysis, Where
from friture.recording.analysis_log import AnalysisLog
from friture.recording.store import BYTES_PER_GB, SegmentStore
from friture.recording.wav_segment import (
    BYTES_PER_SAMPLE,
    SegmentInfo,
    WavSegmentWriter,
    float_to_int16,
    segment_name,
)

log = logging.getLogger(__name__)

# WHAT IS WRITTEN BESIDE THE AUDIO is decided in friture.recording.analysis:
# levels every second, a voice score every 5 s and the Band Survey lines
# every minute, in <stem>.analysis.jsonl next to each WAV.


@dataclass
class RecorderStatus:
    state: str = "off"            # off / waiting / recording / error
    message: str = ""
    folder: str = ""
    current_file: str = ""
    current_seconds: float = 0.0
    stored_bytes: int = 0
    cap_bytes: int = 0
    dropped_blocks: int = 0
    segments_closed: int = 0
    recovered: int = 0
    over_cap_bytes: int = 0       # protected segments alone exceed the cap
    analysis_lag_s: float = 0.0   # audio written but not yet analysed
    analysis_skipped_s: float = 0.0   # audio the analysis had to skip to keep up
    errors: int = 0               # errors the writer recovered from this session
    sidecars_deferred: int = 0    # sidecars Windows would not let it replace in time


class ContinuousRecorder:

    SEGMENT_S = 600.0             # 10 minutes, 300 MB at 250 kHz mono
    QUEUE_S = 60.0                # how far the disk may fall behind the capture
    # How far the analysis may fall behind the recording before it skips.
    # The analysis costs about 5% of a core (measured: 125 s of recording in
    # 5.9 s of CPU), so it only falls behind when the machine is starved,
    # and then it is the analysis that gives way, never the audio.
    ANALYSIS_QUEUE_S = 120.0
    IDLE_CLOSE_S = 2.0
    HOUSEKEEPING_S = 30.0         # rotation and free-space checks
    MIN_FREE_BYTES = 2 * BYTES_PER_GB

    def __init__(self, fs: int, frames_per_block: int, describe_source=lambda: "") -> None:
        self.fs = int(fs)
        self.segment_frames = int(round(self.SEGMENT_S * self.fs))
        self._describe_source = describe_source
        self._q: queue.Queue = queue.Queue(maxsize=max(16, int(self.QUEUE_S * fs / frames_per_block)))
        self._commands: queue.Queue = queue.Queue()
        self._enabled = False
        self._dropped = 0
        self._dropped_lock = threading.Lock()
        # run ids restart at 0 with every launch; the session makes them unique
        self._session = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._status = RecorderStatus()
        self._status_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # The analysis runs on a thread of its own, fed by the writer AFTER
        # each piece is on disk, so it sees exactly what was recorded and
        # knows which file and offset each piece landed at.
        self._frames_per_block = int(frames_per_block)
        self._aq: queue.Queue = queue.Queue(
            maxsize=max(16, int(self.ANALYSIS_QUEUE_S * fs / frames_per_block)))
        self._analysis_thread: threading.Thread | None = None
        self._analysis = LiveAnalysis(self.fs)
        self._alog = AnalysisLog(self.fs)
        self._skipped_frames = 0

        # writer-thread state
        self._store: SegmentStore | None = None
        self._cap_bytes = 0
        self._writer: WavSegmentWriter | None = None
        self._expect: tuple[str, int] | None = None      # (run key, next index) of the open segment
        self._last_closed: tuple[str, int, str, float] | None = None
        self._run_start: tuple[str, float, int] | None = None   # (run key, wall time, index)
        self._last_data_at = 0.0
        self._last_end_epoch = 0.0
        self._next_housekeeping = 0.0
        self._stored_bytes_at_housekeeping = 0
        self._segments_closed = 0
        self._paused_for_space = False
        self._errors = 0
        self._deferred = 0

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="friture-recorder", daemon=True)
            self._thread.start()
        if self._analysis_thread is None:
            self._analysis_thread = threading.Thread(target=self._run_analysis,
                                                     name="friture-recording-analysis", daemon=True)
            self._analysis_thread.start()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Write out what is queued, close the segment, then finish the analysis."""
        self._enabled = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self._analysis_thread is not None:
            self._aq.put(("stop",))
            # at 5% of a core even a full queue is a few seconds of work
            self._analysis_thread.join(timeout + 15.0)
            self._analysis_thread = None

    # -- called from the GUI thread ------------------------------------------------

    def configure(self, enabled: bool, folder, cap_gb: float) -> None:
        self._commands.put(("configure", bool(enabled), Path(folder), float(cap_gb)))
        self._enabled = bool(enabled)

    def status(self) -> RecorderStatus:
        with self._status_lock:
            s = self._status
            return RecorderStatus(**vars(s))

    # -- called from the CAPTURE thread -------------------------------------------

    def push(self, block, run_id: int, run_index: int, t_first: float, overflow: bool) -> None:
        if not self._enabled:
            return
        try:
            self._q.put_nowait((float_to_int16(block), run_id, run_index, t_first, overflow))
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1

    # -- the writer thread ---------------------------------------------------------

    def _run(self) -> None:
        # NOTHING MAY END THIS LOOP BUT A STOP. It used to guard only the write
        # itself, and a sidecar that Windows would not let it replace -- a
        # reader held the file open -- raised out of the idle close and killed
        # the thread: the recording ended for good, silently. Found by a test
        # that failed one run in six. Whatever goes wrong now is logged,
        # counted, shown, and the loop carries on; the next block that is
        # written clears the error state.
        while True:
            try:
                self._handle_commands()
                try:
                    item = self._q.get(timeout=0.25)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    if self._writer is not None and time.monotonic() - self._last_data_at > self.IDLE_CLOSE_S:
                        self._close("the capture stopped")
                    self._housekeeping()
                    continue
                self._write(*item)
                self._housekeeping()
            except Exception as e:
                log.exception("Recording hit an error; carrying on")
                self._errors += 1
                self._close_quietly()
                self._set(state="error", errors=self._errors,
                          message="%s: %s" % (type(e).__name__, e))
                time.sleep(0.5)
        try:
            self._close("the application closed")
        except Exception:
            log.exception("Could not close the last segment cleanly")
        self._set(state="off", message="stopped")

    def _handle_commands(self) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            _, enabled, folder, cap_gb = cmd
            self._cap_bytes = int(cap_gb * BYTES_PER_GB)
            # Only switching off or moving the folder ends the file being
            # written. A new cap is just a number for the next rotation --
            # splitting the recording because someone typed in a spin box
            # would put a break where the capture had none.
            same_place = (enabled and self._store is not None
                          and Path(self._store.folder) == Path(folder))
            if same_place:
                self._set(cap_bytes=self._cap_bytes)
                self._next_housekeeping = 0.0
                continue
            self._close("the recording settings changed")
            if not enabled:
                self._store = None
                self._set(state="off", message="continuous recording is off",
                          folder=str(folder), cap_bytes=self._cap_bytes, current_file="",
                          current_seconds=0.0)
                continue
            try:
                store = SegmentStore(folder)
                store.ensure()
                recovered = store.repair_interrupted()
                self._store = store
                self._set(state="waiting", folder=str(folder), cap_bytes=self._cap_bytes,
                          recovered=len(recovered),
                          message=("repaired %d segment(s) an earlier session left open"
                                   % len(recovered)) if recovered else "waiting for the capture")
                self._next_housekeeping = 0.0
                self._housekeeping()
            except OSError as e:
                self._store = None
                self._set(state="error", folder=str(folder), message="cannot use the folder: %s" % e)

    def _write(self, data: np.ndarray, run_id: int, run_index: int, t_first: float, overflow: bool) -> None:
        if self._store is None or self._paused_for_space:
            return
        self._last_data_at = time.monotonic()
        key = "%s-%d" % (self._session, run_id)
        frames = data.shape[0]
        channels = data.shape[1] if data.ndim > 1 else 1

        broken = (self._writer is None
                  or self._expect != (key, run_index)
                  or channels != self._writer.info.channels
                  or overflow)
        if broken:
            gap_s, reason = self._describe_break(key, run_index, t_first, overflow)
            if self._writer is not None:
                self._close(reason or "the capture broke")
            self._open(key, run_index, t_first, channels, gap_s, reason)

        offset = 0
        while offset < frames:
            room = self.segment_frames - self._writer.n_frames
            piece = data[offset:offset + room]
            pos = self._writer.n_frames
            self._writer.write(piece)
            self._to_analysis(piece, Where(Path(self._writer.info.wav).stem, pos, key,
                                           run_index + offset, t_first + offset / self.fs))
            offset += piece.shape[0]
            self._last_end_epoch = t_first + offset / self.fs
            self._expect = (key, run_index + offset)
            if self._writer.n_frames >= self.segment_frames:
                self._close(None)                   # a planned split: the run goes on
                if offset < frames:
                    self._open(key, run_index + offset, t_first + offset / self.fs, channels, None, None)

        self._set(state="recording", current_file=self._writer.info.wav if self._writer else "",
                  current_seconds=(self._writer.n_frames / self.fs) if self._writer else 0.0,
                  stored_bytes=self._stored_bytes_at_housekeeping + (self._writer.file_bytes if self._writer else 0),
                  dropped_blocks=self._dropped,
                  message="")

    def _describe_break(self, key, run_index, t_first, overflow):
        """What is known about the gap in front of a new segment."""
        last = self._last_closed if self._writer is None else (
            self._expect[0], self._expect[1], self._writer.info.wav, self._last_end_epoch)
        if last is None:
            return None, None
        last_key, last_next, _, last_end = last
        if overflow:
            return None, "the audio driver reported lost input"
        if last_key == key and run_index == last_next:
            return None, None                       # no break at all
        if last_key == key and run_index > last_next:
            with self._dropped_lock:
                dropped = self._dropped
            why = ("the disk fell %.0f s behind and blocks were dropped" % self.QUEUE_S
                   if dropped else "samples were missing from the capture")
            return (run_index - last_next) / self.fs, why
        return max(t_first - last_end, 0.0), "the capture restarted"

    def _open(self, key, run_index, t_first, channels, gap_s, reason) -> None:
        cont = None
        if (self._last_closed is not None and gap_s is None and reason is None
                and self._last_closed[0] == key and self._last_closed[1] == run_index):
            cont = self._last_closed[2]
        name = segment_name(t_first) + ".wav"
        info = SegmentInfo(
            wav=name, fs=self.fs, channels=channels, start_epoch=t_first,
            start_local=datetime.fromtimestamp(t_first).isoformat(timespec="milliseconds"),
            run_id=key, run_index=run_index, device=self._describe_source(),
            continues=cont, gap_before_s=gap_s, gap_reason=reason)
        self._writer = WavSegmentWriter(self._store.folder, info)
        self._expect = (key, run_index)
        # clock drift is measured from where the recorded run began, which
        # a continuing segment inherits (see SegmentInfo.clock_drift_ppm)
        if cont is None or self._run_start is None or self._run_start[0] != key:
            self._run_start = (key, t_first, run_index)

    def _close(self, reason) -> None:
        if self._writer is None:
            return
        w = self._writer
        self._writer = None
        anchor = self._run_start[1:] if self._run_start and self._run_start[0] == w.info.run_id else None
        info = w.close(self._last_end_epoch, anchor)
        saved = w.saved
        if reason:
            info.notes = list(info.notes) + ["ended because %s" % reason]
            saved = info.save(self._store.folder)
        if not saved:
            self._deferred += 1
            log.warning("The sidecar of %s stayed locked; its closed state waits in .json.tmp", info.wav)
            self._set(sidecars_deferred=self._deferred)
        self._last_closed = (info.run_id, info.run_index + info.n_frames, info.wav, info.end_epoch)
        if reason:
            # a real break: the analysis writes out its windows in progress
            # now rather than when the next run's audio shows the break
            try:
                self._aq.put(("flush", self._store.folder), timeout=5.0)
            except queue.Full:
                pass
        self._segments_closed += 1
        self._stored_bytes_at_housekeeping += w.file_bytes
        self._set(segments_closed=self._segments_closed, current_file="", current_seconds=0.0,
                  state="waiting" if reason else "recording")
        self._next_housekeeping = 0.0             # rotate right away

    def _to_analysis(self, piece: np.ndarray, where: Where) -> None:
        try:
            self._aq.put_nowait(("data", self._store.folder, piece, where))
        except queue.Full:
            # the audio is on disk; only its analysis is lost, and the gap in
            # the run index makes the analysis start its windows afresh
            self._skipped_frames += piece.shape[0]

    def _run_analysis(self) -> None:
        """The analysis thread: pieces in, log records out."""
        while True:
            item = self._aq.get()
            try:
                if item[0] == "stop":
                    break
                if item[0] == "flush":
                    folder = item[1]
                    records = self._analysis.flush()
                else:
                    _, folder, piece, where = item
                    records = self._analysis.feed(piece, where)
                for rec in records:
                    self._alog.write(folder, rec)
            except Exception:
                log.exception("Recording analysis failed; the audio is unaffected")
            self._set(analysis_lag_s=self._aq.qsize() * self._frames_per_block / self.fs,
                      analysis_skipped_s=self._skipped_frames / self.fs)
        try:
            self._alog.close()
        except Exception:
            pass

    def _close_quietly(self) -> None:
        try:
            self._close("a write failed")
        except Exception:
            self._writer = None

    def _housekeeping(self) -> None:
        now = time.monotonic()
        if self._store is None or now < self._next_housekeeping:
            return
        self._next_housekeeping = now + self.HOUSEKEEPING_S
        keep = self._writer.info.wav if self._writer is not None else None
        if self._writer is not None:
            self._writer.refresh_header()
        try:
            deleted, over = self._store.rotate(self._cap_bytes, keep=keep)
            stored = self._store.total_bytes()
            free = self._store.free_bytes()
        except OSError as e:
            self._set(state="error", message="cannot manage the folder: %s" % e)
            return
        current = self._writer.file_bytes if self._writer is not None else 0
        self._stored_bytes_at_housekeeping = stored - current
        if deleted:
            log.info("Rotation deleted %d old segment(s)", deleted)
        if free < self.MIN_FREE_BYTES:
            if not self._paused_for_space:
                self._close("the disk is almost full")
            self._paused_for_space = True
            self._set(state="error", stored_bytes=stored, over_cap_bytes=over,
                      message="stopped: only %.1f GB free on the disk" % (free / BYTES_PER_GB))
            return
        self._paused_for_space = False
        self._set(stored_bytes=stored, over_cap_bytes=over)

    def _set(self, **fields) -> None:
        with self._status_lock:
            for k, v in fields.items():
                setattr(self._status, k, v)
