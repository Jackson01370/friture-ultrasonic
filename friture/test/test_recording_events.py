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

"""friture.recording.events: the list says what happened, and only that.

  QUIET ROOM, EMPTY LIST   a day of a steady room produces no events
  JUDGED AGAINST THE ROOM  the same level is an event in a quiet room and
                           nothing in a loud one
  SUSTAINED VS BRIEF       speech holds the voice score up window after
                           window; a knock lifts one. The first is an event,
                           the second is an event marked brief
  SWITCHING, NOT WOBBLE    a line that appears and stays is an event; one
                           that blinks for a minute is not
"""

import unittest

import numpy as np

from friture.recording.events import (
    LINE_DB,
    LOUD_DB,
    VOICE_MARGIN,
    detect,
    line_events,
    loud_events,
    voice_events,
)

FS = 250_000


def levels(seconds, base=-60.0, bumps=(), seed=0):
    """One level record per second; bumps are (second, band, dB) additions."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(seconds):
        b = list(base + rng.normal(0, 0.8, 7))
        for s, band, db in bumps:
            if s == k:
                b[band] += db
        out.append({"kind": "levels", "t": 1000.0 + k, "seg": "s%d" % (k // 600), "pos": (k % 600) * FS,
                    "run": "r", "idx": k * FS, "n": FS, "dur": 1.0, "bands_dbfs": b})
    return out


def voice(scores, run="r"):
    return [{"kind": "voice", "t": 1000.0 + 5 * k, "seg": "s", "pos": 5 * k * FS, "run": run,
             "idx": 5 * k * FS, "n": 5 * FS, "dur": 5.0, "score": s, "f0_hz": 180.0}
            for k, s in enumerate(scores)]


def minutes(line_lists):
    return [{"kind": "lines", "t": 1000.0 + 60 * k, "seg": "s", "pos": 60 * k * FS, "run": "r",
             "idx": 60 * k * FS, "n": 60 * FS, "dur": 60.0, "lines": ls}
            for k, ls in enumerate(line_lists)]


class LoudTest(unittest.TestCase):

    def test_a_steady_room_has_no_loud_events(self):
        self.assertEqual(loud_events(levels(3600)), [])

    def test_a_bang_is_one_event_with_its_bands(self):
        recs = levels(1200, bumps=[(700, 0, 20.0), (700, 1, 18.0), (701, 1, 15.0)])
        ev = loud_events(recs)
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertAlmostEqual(e.start, 1700.0)
        self.assertAlmostEqual(e.duration, 2.0)
        self.assertGreaterEqual(e.strength, 18.0)
        self.assertEqual(len(e.bands), 2)
        self.assertEqual((e.seg, e.pos), ("s1", 100 * FS))

    def test_loud_is_relative_to_the_room(self):
        quiet = loud_events(levels(1200, base=-70.0, bumps=[(600, 2, LOUD_DB + 3)]))
        self.assertEqual(len(quiet), 1)
        # the same +13 dB step in a room that is itself 13 dB louder all
        # the time is the room, not an event
        loud_room = [dict(r, bands_dbfs=[v + (LOUD_DB + 3 if i == 2 else 0) for i, v in enumerate(r["bands_dbfs"])])
                     for r in levels(1200, base=-70.0)]
        self.assertEqual(loud_events(loud_room), [])


class VoiceTest(unittest.TestCase):

    def test_resting_room_scores_are_not_voices(self):
        rng = np.random.default_rng(1)
        self.assertEqual(voice_events(voice(list(3.0 + rng.normal(0, 0.15, 400)))), [])

    def test_sustained_speech_is_one_event_a_knock_is_brief(self):
        rng = np.random.default_rng(2)
        s = list(3.0 + rng.normal(0, 0.15, 120))
        for k in range(40, 45):            # 25 s of speech at about -3 dB SNR
            s[k] = 7.5
        s[90] = 7.0                        # a knock: the measured single window
        ev = voice_events(voice(s))
        self.assertEqual(len(ev), 2)
        speech, knock = ev
        self.assertFalse(speech.brief)
        self.assertAlmostEqual(speech.duration, 25.0)
        self.assertTrue(knock.brief)
        self.assertIn("knock", knock.detail)

    def test_quieter_than_the_margin_is_not_an_event(self):
        s = [3.0] * 60
        for k in range(20, 30):
            s[k] = 3.0 + VOICE_MARGIN - 0.3
        self.assertEqual(voice_events(voice(s)), [])

    def test_windows_of_different_runs_are_not_joined(self):
        a = voice([3.0] * 20 + [8.0] * 2, run="a")
        b = voice([8.0] * 2, run="b")
        for r in b:
            r["t"] += 200.0
        ev = voice_events(a + b)
        self.assertEqual(len(ev), 2)


class LineTest(unittest.TestCase):

    steady = [[25000.0, 20.0, 5.0, 1.0, 8.0], [39001.0, 17.0, 1.0, 1.0, 8.0]]

    def test_steady_lines_are_not_events(self):
        self.assertEqual(line_events(minutes([self.steady] * 10)), [])

    def test_a_line_that_switches_on_and_stays_is_an_event(self):
        new = [47000.0, LINE_DB + 5, 0.0, 1.0, 8.0]
        ev = line_events(minutes([self.steady] * 4 + [self.steady + [new]] * 4))
        self.assertEqual([e.kind for e in ev], ["line_on"])
        self.assertAlmostEqual(ev[0].freq_hz, 47000.0)
        self.assertAlmostEqual(ev[0].start, 1000.0 + 4 * 60)

    def test_a_line_that_goes_away_is_an_event(self):
        ev = line_events(minutes([self.steady] * 4 + [self.steady[1:]] * 4))
        self.assertEqual([e.kind for e in ev], ["line_off"])
        self.assertAlmostEqual(ev[0].freq_hz, 25000.0)

    def test_a_one_minute_blink_is_not_switching(self):
        blink = [47000.0, LINE_DB + 5, 0.0, 1.0, 8.0]
        lists = [self.steady] * 4 + [self.steady + [blink]] + [self.steady] * 4
        self.assertEqual(line_events(minutes(lists)), [])

    def test_a_line_jittering_by_a_bin_is_the_same_line(self):
        lists = [[[25000.0 + (3.8 if k % 2 else 0), 20.0, 5.0, 1.0, 8.0]] for k in range(10)]
        self.assertEqual(line_events(minutes(lists)), [])


class DetectTest(unittest.TestCase):

    def test_everything_together_in_time_order(self):
        recs = levels(1200, bumps=[(900, 0, 20.0)]) + voice([3.0] * 30 + [8.0] * 3 + [3.0] * 30)
        ev = detect(recs)
        self.assertEqual([e.kind for e in ev], ["voice", "loud"])
        self.assertTrue(all(a.start <= b.start for a, b in zip(ev, ev[1:])))


if __name__ == "__main__":
    unittest.main()
