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

"""Keeping recordings, and the list of events that says which to keep.

  KEPT MEANS KEPT      a protected file survives rotation, including one
                       protected while it was still being written -- the
                       writer saves its own copy of the sidecar when it
                       closes, and must not undo the protection with it
  PADDED               protecting an event keeps the files either side of
                       it that hold the half-minute around it
  CHEAP TO REFRESH     the event index re-reads only the log that grew, and
                       forgets the events of audio that rotation deleted
"""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from friture.recording.analysis_log import log_path
from friture.recording.event_index import EventIndex
from friture.recording.recorder import ContinuousRecorder
from friture.recording.session import PROTECT_PAD_S, RecordingSession
from friture.recording.store import SegmentStore
from friture.recording.wav_segment import SegmentInfo, WavSegmentWriter

FS = 250_000


def segment(folder, start, seconds, run="r", index=0, protected=False):
    name = "%s.wav" % time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start))
    info = SegmentInfo(wav=name, fs=FS, channels=1, start_epoch=start, start_local=name,
                       run_id=run, run_index=index, protected=protected)
    w = WavSegmentWriter(folder, info)
    w.write(np.zeros((int(seconds * FS), 1), dtype=np.int16))
    w.close(start + seconds)
    return name


class ProtectTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_protected_files_survive_rotation(self):
        a = segment(self.dir, 1_000_000.0, 1.0)
        b = segment(self.dir, 1_000_010.0, 1.0)
        c = segment(self.dir, 1_000_020.0, 1.0)
        store = SegmentStore(self.dir)
        self.assertEqual(store.set_protected([a], True), [a])
        store.rotate(cap_bytes=10)
        left = sorted(p.name for p in self.dir.glob("*.wav"))
        self.assertEqual(left, [a], "only the protected file may remain")
        n, size = store.protected_bytes()
        self.assertEqual(n, 1)
        self.assertGreater(size, FS)

    def test_protecting_the_file_being_written_is_not_undone_when_it_closes(self):
        rec = ContinuousRecorder(FS, 2048)
        rec.IDLE_CLOSE_S = 0.3
        rec.start()
        rec.configure(True, self.dir, 50)
        t0 = time.monotonic()
        while rec.status().state not in ("waiting", "recording") and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        block = np.zeros((2048, 1), dtype=np.float32)
        rec.push(block, 0, 0, 2_000_000.0, False)
        while not list(self.dir.glob("*.json")) and time.monotonic() - t0 < 5:
            time.sleep(0.02)
        wav = SegmentStore(self.dir).segments()[0].wav
        # the user presses "keep" while the file is open
        SegmentStore(self.dir).set_protected([wav], True)
        rec.set_protected(wav, True)
        rec.push(block, 0, 2048, 2_000_000.0 + 2048 / FS, False)
        rec.shutdown()                                   # closes the file
        info = SegmentInfo.load(self.dir / (wav[:-4] + ".json"))
        self.assertEqual(info.state, "closed")
        self.assertTrue(info.protected, "closing the file undid the protection")

    def test_a_protection_on_disk_alone_is_kept_too(self):
        """Someone wrote the sidecar without telling the recorder."""
        info = SegmentInfo(wav="x.wav", fs=FS, channels=1, start_epoch=0.0, start_local="x",
                           run_id="r", run_index=0)
        w = WavSegmentWriter(self.dir, info)
        w.write(np.zeros((100, 1), dtype=np.int16))
        on_disk = SegmentInfo.load(self.dir / "x.json")
        on_disk.protected = True
        on_disk.save(self.dir)
        w.close(1.0)
        self.assertTrue(SegmentInfo.load(self.dir / "x.json").protected)

    def test_protecting_an_event_keeps_the_files_around_it(self):
        a = segment(self.dir, 1_000_000.0, 60.0)
        b = segment(self.dir, 1_000_060.0, 60.0, index=60 * FS)
        c = segment(self.dir, 1_000_300.0, 60.0, run="s")
        session = RecordingSession()
        session.get_folder = lambda: str(self.dir)
        # an event 10 s into b: the padding reaches back into a, not to c
        changed = session.protect(1_000_070.0, 1_000_072.0, True)
        self.assertEqual(sorted(changed), sorted([a, b]))
        self.assertTrue(session.is_protected(1_000_070.0, 1_000_072.0))
        self.assertFalse(SegmentInfo.load(self.dir / (c[:-4] + ".json")).protected)
        session.protect(1_000_070.0, 1_000_072.0, False)
        self.assertFalse(session.is_protected(1_000_070.0, 1_000_072.0))
        self.assertGreater(PROTECT_PAD_S, 5.0)


class EventIndexTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _log(self, stem, records):
        (self.dir / (stem + ".wav")).write_bytes(b"\0" * 44)
        with open(log_path(self.dir, stem), "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_only_the_log_that_changed_is_read_again(self):
        self._log("a", [{"kind": "levels", "t": 1.0, "seg": "a"}])
        self._log("b", [{"kind": "levels", "t": 2.0, "seg": "b"}])
        idx = EventIndex()
        self.assertEqual(len(idx.records(self.dir)), 2)
        cached_a = idx._cache[log_path(self.dir, "a")][1]
        self._log("b", [{"kind": "levels", "t": 3.0, "seg": "b"}])
        self.assertEqual(len(idx.records(self.dir)), 3)
        self.assertIs(idx._cache[log_path(self.dir, "a")][1], cached_a, "a was read again")

    def test_a_log_whose_audio_was_deleted_drops_out(self):
        self._log("a", [{"kind": "levels", "t": 1.0, "seg": "a"}])
        idx = EventIndex()
        self.assertEqual(len(idx.records(self.dir)), 1)
        (self.dir / "a.wav").unlink()
        self.assertEqual(idx.records(self.dir), [])

    def test_the_background_refresh_hands_back_a_result(self):
        self._log("a", [{"kind": "levels", "t": 1.0, "seg": "a"}])
        idx = EventIndex()
        self.assertTrue(idx.refresh(self.dir))
        t0 = time.monotonic()
        while idx.busy and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        events, when, took, folder = idx.result()
        self.assertEqual(events, [])
        self.assertGreater(when, 0)
        self.assertEqual(folder, self.dir)


if __name__ == "__main__":
    unittest.main()
