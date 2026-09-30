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

"""friture.recording: nothing lost, nothing doubled, nothing unreadable.

The properties a drive recorder is worth anything for:

  EXACT      the samples read back from the files, joined in order, are
             bit-for-bit the samples that were captured
  SEAMLESS   a planned split loses and repeats nothing -- the next segment
             starts on the very next frame and says it continues the last
  HONEST     wherever the capture broke, the segment ends there and the
             next one records that a gap came before it, and why
  RECOVERABLE a segment a crash left half-written is readable afterwards,
             with every frame that reached the disk
  BOUNDED    the folder stays under its cap, oldest first, and what the
             user protected is never deleted
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from friture.recording import wav_segment
from friture.recording.recorder import ContinuousRecorder
from friture.recording.store import SegmentStore
from friture.recording.wav_segment import (
    SegmentInfo,
    WavSegmentWriter,
    float_to_int16,
    read_header,
    repair,
)

FS = 250_000
BLOCK = 2048


def grid_block(rng, frames=BLOCK, channels=1):
    """Float32 on the 1/32768 grid, as the UltraMic capture measured."""
    return (rng.integers(-3000, 3000, size=(frames, channels)) / 32768.0).astype(np.float32)


def wait_for(cond, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


class WavSegmentTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _info(self, name="a.wav", channels=1):
        return SegmentInfo(wav=name, fs=FS, channels=channels, start_epoch=1_700_000_000.0,
                           start_local="x", run_id="s-0", run_index=0)

    def test_int16_conversion_is_exact_on_the_capture_grid(self):
        ints = np.arange(-32768, 32768, dtype=np.int32)
        floats = (ints / 32768.0).astype(np.float32).reshape(-1, 1)
        self.assertTrue(np.array_equal(float_to_int16(floats)[:, 0], ints.astype(np.int16)))
        self.assertEqual(int(float_to_int16(np.array([[1.0]], np.float32))[0, 0]), 32767)

    def test_written_file_reads_back_exactly(self):
        rng = np.random.default_rng(1)
        blocks = [float_to_int16(grid_block(rng, channels=2)) for _ in range(20)]
        w = WavSegmentWriter(self.dir, self._info(channels=2))
        for b in blocks:
            w.write(b)
        info = w.close(1_700_000_000.0 + 20 * BLOCK / FS)
        fs, data = wavfile.read(self.dir / "a.wav")
        self.assertEqual(fs, FS)
        self.assertTrue(np.array_equal(data, np.concatenate(blocks)))
        self.assertEqual(info.n_frames, 20 * BLOCK)
        self.assertEqual(json.loads((self.dir / "a.json").read_text())["state"], "closed")

    def test_header_is_honest_while_the_file_is_still_open(self):
        rng = np.random.default_rng(2)
        w = WavSegmentWriter(self.dir, self._info())
        for _ in range(10):
            w.write(float_to_int16(grid_block(rng)))
        w.refresh_header()
        _, _, data_bytes = read_header(self.dir / "a.wav")
        self.assertEqual(data_bytes, 10 * BLOCK * 2)
        fs, data = wavfile.read(self.dir / "a.wav")      # readable mid-recording
        self.assertEqual(data.shape[0], 10 * BLOCK)
        w.close(0.0)

    def test_a_crashed_segment_is_repaired_from_its_size(self):
        """Simulate the kill: frames flushed to the OS, header never updated."""
        rng = np.random.default_rng(3)
        old = wav_segment.HEADER_REFRESH_S
        wav_segment.HEADER_REFRESH_S = 1e9
        try:
            w = WavSegmentWriter(self.dir, self._info())
            blocks = [float_to_int16(grid_block(rng)) for _ in range(7)]
            for b in blocks:
                w.write(b)
            w._f.write(b"\x01")             # half a frame, as a kill mid-write leaves
            w._f.flush()
            w._f.close()                    # no close(): the header still says 0
        finally:
            wav_segment.HEADER_REFRESH_S = old
        self.assertEqual(read_header(self.dir / "a.wav")[2], 0)
        info = repair(self.dir, "a.wav")
        self.assertEqual(info.state, "recovered")
        self.assertEqual(info.n_frames, 7 * BLOCK)
        fs, data = wavfile.read(self.dir / "a.wav")
        self.assertTrue(np.array_equal(data, np.concatenate(blocks)[:, 0]))


    def test_the_header_reaches_the_disk_the_moment_the_file_opens(self):
        """A process that ended before the first flush used to leave 0 bytes."""
        w = WavSegmentWriter(self.dir, self._info())
        self.assertEqual((self.dir / "a.wav").stat().st_size, 44)
        w.close(0.0)


class ExitWithoutShutdownTest(unittest.TestCase):
    """The process ends and nobody called shutdown() -- a script that never
    closed its window did exactly this, into the user's own folder, and left
    0-byte files that stayed "open" for ever."""

    def test_the_segment_is_closed_properly_anyway(self):
        import subprocess
        import sys
        import textwrap
        folder = Path(tempfile.mkdtemp())
        try:
            code = textwrap.dedent("""
                import sys, time
                import numpy as np
                from friture.recording.recorder import ContinuousRecorder
                rec = ContinuousRecorder(250000, 2048)
                rec.start()
                rec.configure(True, sys.argv[1], 50)
                t0 = time.monotonic()
                while rec.status().state not in ("waiting", "recording") and time.monotonic() - t0 < 5:
                    time.sleep(0.02)
                b = (np.arange(2048).reshape(-1, 1) % 100 / 32768).astype(np.float32)
                for i in range(20):
                    rec.push(b, 0, i * 2048, 1000.0 + i * 2048 / 250000, False)
                time.sleep(0.2)
            """)
            root = Path(__file__).resolve().parents[2]
            subprocess.run([sys.executable, "-c", code, str(folder)], cwd=root, check=True, timeout=60)
            segs = SegmentStore(folder).segments()
            self.assertEqual(len(segs), 1)
            self.assertEqual(segs[0].state, "closed")
            self.assertEqual(segs[0].n_frames, 20 * 2048)
            self.assertEqual((folder / segs[0].wav).stat().st_size, 44 + 20 * 2048 * 2)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


class StoreTest(unittest.TestCase):

    def test_a_header_less_file_is_cleared_not_left_open(self):
        (self.dir / "2026-01-01_00-00-00.000.wav").write_bytes(b"")
        SegmentInfo(wav="2026-01-01_00-00-00.000.wav", fs=FS, channels=1, start_epoch=0.0,
                    start_local="x", run_id="r", run_index=0).save(self.dir)   # state "open"
        store = SegmentStore(self.dir)
        self.assertEqual(store.repair_interrupted(), [])
        self.assertEqual(store.removed_empty, 1)
        self.assertEqual(list(self.dir.iterdir()), [])

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.store = SegmentStore(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _fake(self, stem, size, protected=False):
        (self.dir / (stem + ".wav")).write_bytes(b"\0" * size)
        SegmentInfo(wav=stem + ".wav", fs=FS, channels=1, start_epoch=0.0, start_local=stem,
                    run_id="r", run_index=0, state="closed", protected=protected).save(self.dir)

    def test_rotation_deletes_oldest_first_and_spares_the_protected(self):
        self._fake("2026-01-01_00-00-00.000", 1000)
        self._fake("2026-01-01_00-10-00.000", 1000, protected=True)
        self._fake("2026-01-01_00-20-00.000", 1000)
        self._fake("2026-01-01_00-30-00.000", 1000)
        side = len((self.dir / "2026-01-01_00-00-00.000.json").read_bytes())
        deleted, over = self.store.rotate(cap_bytes=2 * (1000 + side) + 10,
                                          keep="2026-01-01_00-30-00.000.wav")
        left = sorted(p.name for p in self.dir.glob("*.wav"))
        self.assertEqual(deleted, 2)
        self.assertEqual(over, 0)
        self.assertEqual(left, ["2026-01-01_00-10-00.000.wav", "2026-01-01_00-30-00.000.wav"])

    def test_the_protected_are_kept_even_over_the_cap(self):
        self._fake("2026-01-01_00-00-00.000", 5000, protected=True)
        deleted, over = self.store.rotate(cap_bytes=100)
        self.assertEqual(deleted, 0)
        self.assertGreater(over, 0)
        self.assertTrue((self.dir / "2026-01-01_00-00-00.000.wav").exists())


class RecorderTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.rec = ContinuousRecorder(FS, BLOCK, describe_source=lambda: "test source")
        self.rec.segment_frames = 5 * BLOCK + 700      # splits land mid-block on purpose
        self.rec.IDLE_CLOSE_S = 0.3

    def tearDown(self):
        self.rec.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _start(self):
        self.rec.start()
        self.rec.configure(True, self.dir, 50)
        self.assertTrue(wait_for(lambda: self.rec.status().state in ("waiting", "recording")))

    def _closed(self):
        return [s for s in SegmentStore(self.dir).segments() if s.state == "closed"]

    def _read_all(self):
        return [(s, wavfile.read(self.dir / s.wav)[1]) for s in SegmentStore(self.dir).segments()]

    def test_split_segments_join_back_bit_for_bit(self):
        self._start()
        rng = np.random.default_rng(4)
        blocks = [grid_block(rng) for _ in range(23)]
        t0 = 1_700_000_000.0
        for i, b in enumerate(blocks):
            self.rec.push(b, 0, i * BLOCK, t0 + i * BLOCK / FS, False)
        self.assertTrue(wait_for(lambda: len(self._closed()) >= 5))
        segs = self._read_all()
        joined = np.concatenate([d for _, d in segs])
        self.assertTrue(np.array_equal(joined, float_to_int16(np.concatenate(blocks))[:, 0]),
                        "the files do not add up to what was captured")
        infos = [s for s, _ in segs]
        for prev, nxt in zip(infos, infos[1:]):
            self.assertEqual(nxt.continues, prev.wav, "a planned split must say it continues")
            self.assertEqual(nxt.run_index, prev.run_index + prev.n_frames)
            self.assertIsNone(nxt.gap_before_s)
        self.assertTrue(all(s.n_frames == self.rec.segment_frames for s in infos[:-1]))
        self.assertEqual(infos[0].device, "test source")

    def test_a_restart_ends_the_segment_and_is_written_down(self):
        self._start()
        rng = np.random.default_rng(5)
        for i in range(3):
            self.rec.push(grid_block(rng), 0, i * BLOCK, 100.0 + i * BLOCK / FS, False)
        for i in range(3):
            self.rec.push(grid_block(rng), 1, i * BLOCK, 200.0 + i * BLOCK / FS, False)
        self.assertTrue(wait_for(lambda: len(self._closed()) >= 2))
        a, b = self._closed()[:2]
        self.assertEqual(b.gap_reason, "the capture restarted")
        self.assertIsNone(b.continues)
        self.assertAlmostEqual(b.gap_before_s, 200.0 - (100.0 + 3 * BLOCK / FS), places=3)

    def test_missing_samples_split_the_segment_with_the_gap_measured(self):
        self._start()
        rng = np.random.default_rng(6)
        self.rec.push(grid_block(rng), 0, 0, 10.0, False)
        self.rec.push(grid_block(rng), 0, 5 * BLOCK, 10.0 + 5 * BLOCK / FS, False)   # 4 blocks missing
        self.assertTrue(wait_for(lambda: len(self._closed()) >= 2))
        b = self._closed()[1]
        self.assertAlmostEqual(b.gap_before_s, 4 * BLOCK / FS, places=6)
        self.assertIn("missing", b.gap_reason)

    def test_a_driver_overflow_splits_the_segment(self):
        self._start()
        rng = np.random.default_rng(7)
        self.rec.push(grid_block(rng), 0, 0, 10.0, False)
        self.rec.push(grid_block(rng), 0, BLOCK, 10.0 + BLOCK / FS, True)
        self.assertTrue(wait_for(lambda: len(self._closed()) >= 2))
        self.assertEqual(self._closed()[1].gap_reason, "the audio driver reported lost input")

    def test_a_full_queue_drops_and_counts_instead_of_blocking(self):
        rec = ContinuousRecorder(FS, BLOCK)
        rec._q = __import__("queue").Queue(maxsize=3)
        rec._enabled = True                       # thread NOT started: nothing drains
        rng = np.random.default_rng(8)
        t = time.monotonic()
        for i in range(10):
            rec.push(grid_block(rng), 0, i * BLOCK, 0.0, False)
        self.assertLess(time.monotonic() - t, 0.5, "push must never wait")
        self.assertEqual(rec._dropped, 7)

    def test_disabled_writes_nothing(self):
        self.rec.start()
        self.rec.configure(False, self.dir, 50)
        rng = np.random.default_rng(9)
        for i in range(5):
            self.rec.push(grid_block(rng), 0, i * BLOCK, 0.0, False)
        time.sleep(0.5)
        self.assertEqual(list(self.dir.glob("*.wav")), [])
        self.assertEqual(self.rec.status().state, "off")

    def test_a_pause_closes_the_file(self):
        self._start()
        self.rec.push(grid_block(np.random.default_rng(10)), 0, 0, 5.0, False)
        self.assertTrue(wait_for(lambda: len(self._closed()) == 1, timeout=3.0))
        self.assertIn("the capture stopped", " ".join(self._closed()[0].notes))

    def test_a_sidecar_held_open_by_someone_else_does_not_kill_the_recorder(self):
        """Found by a flaky test: on Windows a file that anyone has open cannot be
        replaced, os.replace raised PermissionError, and the writer thread died
        -- the recording stopped for good, silently. Antivirus, the search
        indexer and a program reading the sidecars all hold files open like
        this. The recorder must ride it out and keep recording.
        """
        self._start()
        rng = np.random.default_rng(13)
        self.rec.push(grid_block(rng), 0, 0, 5.0, False)
        self.assertTrue(wait_for(lambda: list(self.dir.glob("*.json")), timeout=3.0))
        side = sorted(self.dir.glob("*.json"))[0]
        holder = open(side, "rb")                     # someone reading it, and not letting go
        try:
            time.sleep(1.0)                           # the idle close runs while it is held
        finally:
            holder.close()
        self.assertTrue(self.rec._thread.is_alive(), "the writer thread died")
        self.assertTrue(wait_for(lambda: len(self._closed()) == 1, timeout=5.0),
                        "the held segment never got closed")
        # and it is still recording afterwards
        self.rec.push(grid_block(rng), 1, 0, 50.0, False)
        self.assertTrue(wait_for(lambda: len(self._closed()) == 2, timeout=5.0))

    def test_a_sidecar_held_past_every_retry_is_adopted_on_the_next_start(self):
        old = SegmentInfo.REPLACE_TRIES
        SegmentInfo.REPLACE_TRIES = 2                 # give up after 0.1 s
        try:
            self._start()
            self.rec.push(grid_block(np.random.default_rng(14)), 0, 0, 5.0, False)
            self.assertTrue(wait_for(lambda: list(self.dir.glob("*.json")), timeout=3.0))
            side = sorted(self.dir.glob("*.json"))[0]
            with open(side, "rb"):
                self.assertTrue(wait_for(lambda: self.rec.status().sidecars_deferred == 1, timeout=3.0))
            self.assertTrue(side.with_suffix(".json.tmp").exists())
            self.assertEqual(SegmentInfo.load(side).state, "open", "the old state is still in place")
        finally:
            SegmentInfo.REPLACE_TRIES = old
        SegmentStore(self.dir).repair_interrupted()
        self.assertEqual(SegmentInfo.load(side).state, "closed", "the closed state was adopted")
        self.assertFalse(side.with_suffix(".json.tmp").exists())

    def test_an_unexpected_error_does_not_end_the_recording(self):
        self._start()
        real = self.rec._housekeeping
        calls = {"n": 0}

        def breaks_once():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("something nobody planned for")
            real()

        self.rec._housekeeping = breaks_once
        rng = np.random.default_rng(15)
        self.rec.push(grid_block(rng), 0, 0, 5.0, False)
        self.assertTrue(wait_for(lambda: self.rec.status().errors == 1, timeout=3.0))
        self.assertTrue(self.rec._thread.is_alive(), "the writer thread died")
        self.rec.push(grid_block(rng), 1, 0, 50.0, False)
        self.assertTrue(wait_for(lambda: len(self._closed()) >= 2, timeout=5.0),
                        "nothing was recorded after the error")

    def test_changing_only_the_cap_does_not_split_the_recording(self):
        self._start()
        rng = np.random.default_rng(12)
        self.rec.push(grid_block(rng), 0, 0, 10.0, False)
        time.sleep(0.2)
        self.rec.configure(True, self.dir, 20)           # a new cap, same folder
        time.sleep(0.2)
        self.rec.push(grid_block(rng), 0, BLOCK, 10.0 + BLOCK / FS, False)
        self.assertTrue(wait_for(lambda: len(self._closed()) >= 1, timeout=3.0))
        time.sleep(0.5)
        segs = SegmentStore(self.dir).segments()
        self.assertEqual(len(segs), 1, "a cap change split the file")
        self.assertEqual(segs[0].n_frames, 2 * BLOCK)
        self.assertEqual(self.rec.status().cap_bytes, 20 * 1_000_000_000)

    def test_a_segment_left_open_by_a_crash_is_repaired_on_start(self):
        old = wav_segment.HEADER_REFRESH_S
        wav_segment.HEADER_REFRESH_S = 1e9
        try:
            w = WavSegmentWriter(self.dir, SegmentInfo(
                wav="2026-01-01_00-00-00.000.wav", fs=FS, channels=1, start_epoch=0.0,
                start_local="x", run_id="old-0", run_index=0))
            w.write(float_to_int16(grid_block(np.random.default_rng(11))))
            w._f.flush()
            w._f.close()
        finally:
            wav_segment.HEADER_REFRESH_S = old
        self._start()
        self.assertEqual(self.rec.status().recovered, 1)
        info = SegmentInfo.load(self.dir / "2026-01-01_00-00-00.000.json")
        self.assertEqual(info.state, "recovered")
        self.assertEqual(info.n_frames, BLOCK)


if __name__ == "__main__":
    unittest.main()
