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

"""friture.recording.analysis: the log beside the audio says what was in it.

  THE SAME AUDIO, THE SAME LOG   however it is chopped into blocks -- which
                                 is what lets old recordings be analysed
                                 offline with the live code
  CORRECT NUMBERS                a sine of known amplitude reads its level
                                 in its own band and not in the others; a
                                 tone shows up as a line at its frequency;
                                 a voice outscores the room it is in
  NO WINDOW ACROSS A BREAK       at a break the windows in progress are cut,
                                 marked partial, and never joined to audio
                                 from the other side
  PLACED EXACTLY                 each record names the file and offset its
                                 window started at
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from friture.recording.analysis import (
    LEVEL_BANDS,
    LINES_S,
    MIN_FRACTION,
    LiveAnalysis,
    Where,
)
from friture.recording.analysis_log import AnalysisLog, log_path, read_log
from friture.recording.recorder import ContinuousRecorder
from friture.recording.store import SegmentStore
from friture.test.test_speech_detector import synthetic_voice

FS = 250_000


def noise(seconds, seed=0, amp=0.003):
    return amp * np.random.default_rng(seed).normal(size=int(seconds * FS))


def run(analysis, x, block, where=None):
    where = where or Where("seg", 0, "r", 0, 1000.0)
    out = []
    for i in range(0, x.size, block):
        out += analysis.feed(x[i:i + block], where.advanced(i, FS))
    return out


class AnalysisTest(unittest.TestCase):

    def test_block_size_does_not_change_the_log(self):
        x = noise(12.0, seed=1) + 0.01 * np.sin(2 * np.pi * 30_000 * np.arange(int(12 * FS)) / FS)
        a = run(LiveAnalysis(FS), x, 2048) + LiveAnalysis(FS).flush()
        b = run(LiveAnalysis(FS), x, 77_777)
        # the same SET of records; the order is the order windows complete,
        # which legitimately depends on the cutting (see the analysis docstring)
        strip = lambda recs: sorted((r for r in recs if r["kind"] != "lines"),
                                    key=lambda r: (r["idx"], r["kind"]))
        self.assertEqual(json.dumps(strip(a)), json.dumps(strip(b)))
        self.assertEqual(sum(r["kind"] == "levels" for r in a), 12)
        self.assertEqual(sum(r["kind"] == "voice" for r in a), 2)

    def test_a_sine_reads_its_level_in_its_own_band(self):
        amp = 0.1
        x = amp * np.sin(2 * np.pi * 15_000 * np.arange(FS) / FS)
        rec = next(r for r in run(LiveAnalysis(FS), x, 4096) if r["kind"] == "levels")
        expected = 10 * np.log10(amp * amp / 2)            # -23.0 dBFS
        band = [i for i, (lo, hi) in enumerate(LEVEL_BANDS) if lo <= 15_000 < hi][0]
        self.assertAlmostEqual(rec["bands_dbfs"][band], expected, delta=0.1)
        self.assertAlmostEqual(rec["rms_dbfs"], expected, delta=0.1)
        self.assertAlmostEqual(rec["peak_dbfs"], 20 * np.log10(amp), delta=0.1)
        others = [v for i, v in enumerate(rec["bands_dbfs"]) if i != band]
        self.assertLess(max(others), expected - 60)

    def test_a_steady_tone_is_a_line_at_its_frequency(self):
        t = np.arange(int(LINES_S * FS)) / FS
        x = noise(LINES_S, seed=2) + 0.003 * np.sin(2 * np.pi * 25_000 * t)
        recs = [r for r in run(LiveAnalysis(FS), x, 65_536) if r["kind"] == "lines"]
        self.assertEqual(len(recs), 1)
        top = max(recs[0]["lines"], key=lambda l: l[1])
        self.assertAlmostEqual(top[0], 25_000, delta=5)
        self.assertGreater(top[1], 10.0)
        self.assertLess(top[3], 6.0, "a steady tone must read as steady")

    def test_a_voice_outscores_the_room(self):
        room = noise(10.0, seed=3)
        voice16 = synthetic_voice()                       # 12 s at 16 kHz
        voice = np.interp(np.arange(room.size) / FS, np.arange(voice16.size) / 16_000, voice16)
        loud = room + voice * (room.std() * 3 / voice.std())
        s_room = [r["score"] for r in run(LiveAnalysis(FS), room, 8192) if r["kind"] == "voice"]
        s_voice = [r["score"] for r in run(LiveAnalysis(FS), loud, 8192) if r["kind"] == "voice"]
        self.assertGreater(min(s_voice), 2 * max(s_room),
                           "voice %s vs room %s" % (s_voice, s_room))

    def test_records_say_where_their_window_started(self):
        start = Where("2026-01-01_00-00-00.000", 1234, "sess-7", 50_000, 2000.0)
        recs = run(LiveAnalysis(FS), noise(6.0, seed=4), 3000, start)
        levels = [r for r in recs if r["kind"] == "levels"]
        for k, r in enumerate(levels):
            self.assertEqual(r["seg"], start.seg)
            self.assertEqual(r["pos"], 1234 + k * FS)
            self.assertEqual(r["n"], FS)
            self.assertEqual(r["idx"], 50_000 + k * FS)
            self.assertAlmostEqual(r["t"], 2000.0 + k, places=3)

    def test_no_window_crosses_a_break(self):
        a = LiveAnalysis(FS)
        first = run(a, noise(2.6, seed=5), 4096, Where("s", 0, "run-A", 0, 100.0))
        # the run changes: 0.6 s of an unfinished level window and 2.6 s of
        # a voice window are in progress
        second = a.feed(noise(0.1, seed=6), Where("s", int(2.6 * FS), "run-B", 0, 200.0))
        cut = [r for r in second if r.get("partial")]
        kinds = sorted(r["kind"] for r in cut)
        self.assertEqual(kinds, ["levels", "voice"])          # both >= MIN_FRACTION
        lv = next(r for r in cut if r["kind"] == "levels")
        self.assertEqual(lv["run"], "run-A")
        self.assertAlmostEqual(lv["dur"], 0.6, places=2)
        self.assertEqual(lv["n"], int(2.6 * FS) - 2 * FS)
        # nothing from run B was folded into a run A record
        for r in first + second:
            if r["run"] == "run-A":
                self.assertLessEqual(r["idx"] + r["dur"] * FS, 2.6 * FS + 1)

    def test_too_short_a_remnant_is_dropped_not_padded(self):
        a = LiveAnalysis(FS)
        run(a, noise(1.2, seed=7), 4096)                       # 0.2 s of a level window
        cut = a.flush()
        self.assertFalse(any(r["kind"] == "levels" for r in cut),
                         "0.2 s is under MIN_FRACTION (%.1f) of a level window" % MIN_FRACTION)


class AnalysisLogTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_header_once_and_a_torn_last_line_is_skipped(self):
        log = AnalysisLog(FS)
        for k in range(3):
            log.write(self.dir, {"kind": "levels", "seg": "a", "pos": k})
        log.close()
        log.write(self.dir, {"kind": "levels", "seg": "a", "pos": 3})   # reopened: no 2nd header
        log.close()
        with open(log_path(self.dir, "a"), "a", encoding="utf-8") as f:
            f.write('{"kind": "levels", "seg": "a", "po')                 # a crash mid-line
        head, recs, bad = read_log(log_path(self.dir, "a"))
        self.assertEqual(head["kind"], "header")
        self.assertEqual([r["pos"] for r in recs], [0, 1, 2, 3])
        self.assertEqual(bad, 1)
        text = log_path(self.dir, "a").read_text(encoding="utf-8")
        self.assertEqual(text.count('"header"'), 1)


class RecorderWithAnalysisTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.rec = ContinuousRecorder(FS, 2048)
        self.rec.segment_frames = int(3.5 * FS)     # splits inside level windows
        self.rec.IDLE_CLOSE_S = 0.5

    def tearDown(self):
        self.rec.shutdown()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_log_sits_beside_each_file_and_points_into_it(self):
        self.rec.start()
        self.rec.configure(True, self.dir, 50)
        t0 = time.monotonic()
        while self.rec.status().state not in ("waiting", "recording") and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        x = (noise(8.0, seed=8) * 32768).astype(np.float32) / 32768
        blocks = x.reshape(-1, 1)
        for i in range(0, blocks.shape[0], 2048):
            self.rec.push(np.rint(blocks[i:i + 2048] * 32768) / 32768, 0, i, 500.0 + i / FS, False)
        self.rec.shutdown()                            # flushes the analysis too
        segs = {s.wav[:-4]: s for s in SegmentStore(self.dir).segments()}
        self.assertGreaterEqual(len(segs), 3)
        all_recs = []
        for stem in segs:
            head, recs, bad = read_log(log_path(self.dir, stem))
            self.assertIsNotNone(head, "no log beside %s" % stem)
            self.assertEqual(bad, 0)
            all_recs += recs
            for r in recs:
                self.assertEqual(r["seg"], stem, "a record filed under the wrong segment")
                self.assertLess(r["pos"], segs[stem].n_frames)
        levels = [r for r in all_recs if r["kind"] == "levels"]
        self.assertEqual(len(levels), 8)                # one per second, across the splits
        self.assertEqual(sorted(r["idx"] for r in levels), [k * FS for k in range(8)])
        voice = [r for r in all_recs if r["kind"] == "voice"]
        self.assertEqual(len(voice), 2)                 # 5 s + a 3 s remnant, >= MIN_FRACTION
        self.assertTrue(voice[-1].get("partial"))
        self.assertEqual(self.rec.status().analysis_skipped_s, 0.0)

    def test_rotation_takes_the_log_with_the_audio(self):
        stem = "2026-01-01_00-00-00.000"
        (self.dir / (stem + ".wav")).write_bytes(b"\0" * 5000)
        (self.dir / (stem + ".json")).write_text(json.dumps(
            {"wav": stem + ".wav", "fs": FS, "channels": 1, "start_epoch": 0.0, "start_local": "x",
             "run_id": "r", "run_index": 0, "state": "closed"}), encoding="utf-8")
        log_path(self.dir, stem).write_text("x" * 3000, encoding="utf-8")
        store = SegmentStore(self.dir)
        self.assertGreater(store.total_bytes(), 8000, "the log must count against the cap")
        store.rotate(cap_bytes=10)
        self.assertEqual(list(self.dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
