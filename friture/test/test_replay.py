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

"""friture.recording.replay: the docks must get exactly what was recorded.

  EXACT       the blocks handed out, joined, are the samples in the files,
              straight through a planned split
  NO STITCH   across a real gap no block holds audio from both sides, and
              the first block after it says so
  PACED       at speed s, s seconds of audio go out per second of clock; a
              stall does not turn into an unbounded burst
  FOLLOWS     replaying up to the file being written waits for it to grow
              instead of stopping, and carries on as it does
  SURVIVES    a file deleted under it (rotation) is not an exception
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from friture.recording.replay import RecordingTimeline, ReplaySource
from friture.recording.wav_segment import SegmentInfo, WavSegmentWriter

FS = 250_000
BLOCK = 2048


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(folder, name, start, run, index, data, close=True):
    w = WavSegmentWriter(folder, SegmentInfo(wav=name, fs=FS, channels=1, start_epoch=start,
                                             start_local=name, run_id=run, run_index=index))
    w.write(data.reshape(-1, 1))
    if close:
        w.close(start + data.size / FS)
    else:
        w.refresh_header()
    return w


class ReplayTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(1)
        self.a = rng.integers(-2000, 2000, 3 * BLOCK + 500).astype(np.int16)
        self.b = rng.integers(-2000, 2000, 2 * BLOCK + 100).astype(np.int16)
        self.c = rng.integers(-2000, 2000, 2 * BLOCK).astype(np.int16)
        make(self.dir, "2026-01-01_00-00-00.000.wav", 1000.0, "r", 0, self.a)
        # b continues a exactly: same run, the indices meet
        make(self.dir, "2026-01-01_00-00-00.500.wav", 1000.0 + self.a.size / FS, "r", self.a.size, self.b)
        # c after a 60 s gap, another run
        make(self.dir, "2026-01-01_00-01-00.000.wav", 1060.0, "s", 0, self.c)
        self.clock = FakeClock()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def source(self):
        src = ReplaySource(RecordingTimeline(self.dir), BLOCK, clock=self.clock)
        src.seek(0.0)
        src.play()
        return src

    def drain(self, src, seconds=1.0, steps=50):
        out = []
        for _ in range(steps):
            self.clock.t += seconds / steps
            out += src.fetch()
        return out

    def test_the_timeline_sees_the_join_and_the_gap(self):
        tl = RecordingTimeline(self.dir)
        self.assertEqual(len(tl.entries), 3)
        self.assertTrue(tl.continues(0))
        self.assertFalse(tl.continues(1))
        ranges = tl.ranges()
        self.assertEqual(len(ranges), 2, "a and b are one stretch, c another")
        self.assertAlmostEqual(ranges[0][1], 1000.0 + (self.a.size + self.b.size) / FS)
        # a time inside the gap goes to what follows it
        self.assertEqual(tl.locate(1030.0), (2, 0))
        self.assertEqual(tl.locate(0.0), (0, 0))

    def test_blocks_are_the_recorded_samples_straight_through_a_split(self):
        src = self.source()
        got = self.drain(src, seconds=1.0)
        before_gap = [b for b, _ in got][:(self.a.size + self.b.size) // BLOCK]
        joined = np.concatenate(before_gap)[:, 0]
        expected = np.concatenate((self.a, self.b))[:joined.size]
        self.assertTrue(np.array_equal(joined, expected), "the split was not seamless")
        self.assertFalse(any(j for _, j in got[:len(before_gap)]))

    def test_a_gap_is_jumped_not_stitched(self):
        src = self.source()
        got = self.drain(src, seconds=1.0)
        n_before = (self.a.size + self.b.size) // BLOCK       # the 100-frame... remainder is dropped
        after = got[n_before:]
        self.assertTrue(after, "nothing came after the gap")
        self.assertTrue(after[0][1], "the first block after the gap must say it jumped")
        self.assertTrue(np.array_equal(after[0][0][:, 0], self.c[:BLOCK]),
                        "the block after the gap must be all from the far side")
        self.assertTrue(src.at_end or not src.playing)

    def test_pacing_follows_the_speed(self):
        for speed in (1.0, 4.0):
            src = self.source()
            src.set_speed(speed)
            self.clock.t += 0.01                          # 10 ms
            n = sum(b.shape[0] for b, _ in src.fetch())
            self.assertAlmostEqual(n, speed * 0.01 * FS, delta=BLOCK,
                                   msg="at %gx, 10 ms should yield about %d frames" % (speed, speed * 2500))

    def test_a_stall_is_not_an_unbounded_burst(self):
        src = self.source()
        self.clock.t += 30.0                              # the GUI froze for 30 s
        n = sum(b.shape[0] for b, _ in src.fetch())
        self.assertLessEqual(n, ReplaySource.MAX_CATCHUP_S * FS + 2 * BLOCK)

    def test_pause_freezes_and_seek_lands(self):
        src = self.source()
        self.drain(src, seconds=0.01, steps=2)
        src.pause()
        pos = src.position
        self.clock.t += 5.0
        self.assertEqual(src.fetch(), [])
        self.assertEqual(src.position, pos)
        src.seek(1060.0 + BLOCK / FS)
        self.assertAlmostEqual(src.position, 1060.0 + BLOCK / FS, places=4)

    def test_following_a_file_that_is_still_being_written(self):
        rng = np.random.default_rng(2)
        first = rng.integers(-2000, 2000, BLOCK + 300).astype(np.int16)
        w = make(self.dir, "2026-01-01_00-02-00.000.wav", 1120.0, "t", 0, first, close=False)
        w._f.flush()
        src = ReplaySource(RecordingTimeline(self.dir), BLOCK, clock=self.clock)
        src.seek(1120.0)
        src.play()
        got = self.drain(src, seconds=0.1)
        self.assertEqual(sum(b.shape[0] for b, _ in got), BLOCK, "only the whole block that exists")
        self.assertTrue(src.playing, "reaching the present must wait, not stop")
        more = rng.integers(-2000, 2000, 2 * BLOCK).astype(np.int16)
        w.write(more.reshape(-1, 1))
        w._f.flush()
        got2 = self.drain(src, seconds=0.1)
        joined = np.concatenate([b for b, _ in got + got2])[:, 0]
        self.assertTrue(np.array_equal(joined, np.concatenate((first, more))[:joined.size]))
        self.assertEqual(joined.size, (first.size + more.size) // BLOCK * BLOCK)
        w.close(1130.0)

    def test_a_file_deleted_under_it_is_not_an_exception(self):
        src = self.source()
        self.drain(src, seconds=0.005, steps=1)
        src.close()
        (self.dir / "2026-01-01_00-00-00.000.wav").unlink()
        (self.dir / "2026-01-01_00-00-00.000.json").unlink()
        self.clock.t += ReplaySource.REFRESH_S + 0.1
        got = src.fetch() + self.drain(src, seconds=0.5)
        self.assertTrue(got, "it should carry on with what is left")


if __name__ == "__main__":
    unittest.main()
