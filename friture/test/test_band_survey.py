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

"""friture.demod.survey: does the survey find what is there, and only that?

The cases that matter are the ones that caught this project out by hand:

  a line on a SLOPE      must be found -- the microphone's own floor rises
                         15 dB towards 25 kHz, and a line sitting on that
                         rise is still a line
  a HUMP with no line    must not be reported -- a broad rise is the shape
                         of the noise, not a signal, and reporting it is
                         what sent this project looking at 5.8 kHz
  a line that COMES AND  must be told from one that is simply on
  GOES
"""

import unittest

import numpy as np

from friture.demod.survey import SpectrumSurvey, SurveyLine

FS = 250_000.0
BLOCK = 2048


def feed(survey, x):
    for i in range(0, x.size, BLOCK):
        survey.process(x[i:i + BLOCK])


def noise(n, seed=0, amp=0.01):
    return amp * np.random.default_rng(seed).normal(size=n)


def sloped_noise(n, seed=0):
    """Noise with a broad hump around 25 kHz, like this microphone's floor."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    # a smooth rise to 25 kHz and a fall beyond: no lines anywhere in it
    shape = 1.0 + 6.0 * np.exp(-0.5 * ((f - 25_000.0) / 9_000.0) ** 2)
    return 0.01 * np.fft.irfft(X * shape, n)


class SurveyTest(unittest.TestCase):

    def test_finds_a_line_and_places_it(self):
        n = int(3.0 * FS)
        t = np.arange(n) / FS
        x = noise(n) + 0.02 * np.cos(2 * np.pi * 25_000.0 * t)
        s = SpectrumSurvey(FS)
        feed(s, x)
        self.assertTrue(s.ready)
        lines = s.lines(100.0, 120_000.0)
        self.assertTrue(lines)
        self.assertAlmostEqual(lines[0].frequency_hz, 25_000.0, delta=10.0)
        self.assertGreater(lines[0].excess_db, 15.0)
        self.assertTrue(lines[0].steady)
        self.assertIn("25.00", lines[0].describe())      # bins are 3.8 Hz apart

    def test_a_line_on_a_slope_is_still_found(self):
        """The line is 20 dB up a hump; against the band's own floor it would
        be indistinguishable from the hump's own top."""
        n = int(3.0 * FS)
        t = np.arange(n) / FS
        x = sloped_noise(n) + 0.02 * np.cos(2 * np.pi * 25_000.0 * t)
        s = SpectrumSurvey(FS)
        feed(s, x)
        lines = s.lines(100.0, 120_000.0)
        self.assertTrue(lines)
        self.assertAlmostEqual(lines[0].frequency_hz, 25_000.0, delta=10.0)
        self.assertGreater(lines[0].excess_db, 12.0)

    def test_a_hump_alone_reports_nothing(self):
        """THE test this dock exists for. A broad rise with no line in it must
        come back empty, however tall it is."""
        s = SpectrumSurvey(FS)
        feed(s, sloped_noise(int(3.0 * FS), seed=3))
        lines = s.lines(100.0, 120_000.0, min_excess_db=6.0)
        self.assertEqual(lines, [], "a hump was reported as lines: %s"
                         % [line.describe() for line in lines])
        # and the shape readout still shows the hump, which is where a rise
        # belongs: as the shape of the noise, not as a finding
        shape = s.band_shape([10_000, 20_000, 26_000, 40_000, 80_000])
        medians = [row[2] for row in shape]
        self.assertGreater(max(medians), min(medians) + 6.0)

    def test_steady_and_intermittent_are_told_apart(self):
        n = int(4.0 * FS)
        t = np.arange(n) / FS
        gate = ((t % 1.0) < 0.35).astype(float)      # on for a third of each second
        x = (noise(n, seed=5)
             + 0.02 * np.cos(2 * np.pi * 20_000.0 * t)
             + 0.02 * np.cos(2 * np.pi * 40_000.0 * t) * gate)
        s = SpectrumSurvey(FS)
        feed(s, x)
        found = {round(line.frequency_hz / 1000): line for line in s.lines(100.0, 120_000.0)}
        self.assertIn(20, found)
        self.assertIn(40, found)
        self.assertTrue(found[20].steady, found[20].describe())
        self.assertFalse(found[40].steady, found[40].describe())
        self.assertGreater(found[40].spread_db, found[20].spread_db + 5.0)

    def test_lines_are_separated_and_ranked(self):
        n = int(3.0 * FS)
        t = np.arange(n) / FS
        x = noise(n, seed=7)
        for f, a in ((15_000.0, 0.004), (31_000.0, 0.02), (62_000.0, 0.01)):
            x = x + a * np.cos(2 * np.pi * f * t)
        s = SpectrumSurvey(FS)
        feed(s, x)
        lines = s.lines(100.0, 120_000.0, top=5)
        got = [round(line.frequency_hz / 1000) for line in lines]
        self.assertEqual(got[:3], [31, 62, 15], "not ranked by excess: %s" % got)
        # nothing reported twice
        self.assertEqual(len(got), len(set(got)))

    def test_range_and_top_are_honoured(self):
        n = int(3.0 * FS)
        t = np.arange(n) / FS
        x = noise(n, seed=9) + 0.02 * np.cos(2 * np.pi * 5_000.0 * t) \
            + 0.02 * np.cos(2 * np.pi * 60_000.0 * t)
        s = SpectrumSurvey(FS)
        feed(s, x)
        self.assertEqual([round(line.frequency_hz / 1000) for line in s.lines(1_000.0, 20_000.0)], [5])
        self.assertEqual(len(s.lines(100.0, 120_000.0, top=1)), 1)

    def test_nothing_before_two_windows(self):
        s = SpectrumSurvey(FS)
        self.assertFalse(s.ready)
        self.assertEqual(s.lines(), [])
        s.process(noise(BLOCK))
        self.assertFalse(s.ready)
        self.assertEqual(s.lines(), [])

    def test_history_is_bounded_and_reset_clears(self):
        s = SpectrumSurvey(FS, history=4)
        feed(s, noise(int(2.0 * FS), seed=11))
        self.assertLessEqual(len(s._windows), 4)
        self.assertGreater(s.n_windows_total, 4)
        s.reset()
        self.assertFalse(s.ready)
        self.assertEqual(s.n_windows_total, 0)

    def test_block_size_does_not_matter(self):
        """However the capture is chopped, the same windows come out."""
        x = noise(int(2.0 * FS), seed=13) + 0.01 * np.cos(
            2 * np.pi * 30_000.0 * np.arange(int(2.0 * FS)) / FS)
        one = SpectrumSurvey(FS)
        one.process(x)
        rag = SpectrumSurvey(FS)
        i = 0
        for blk in (2048, 0, 1, 4097, 777, 65536, 12345):
            rag.process(x[i:i + blk])
            i += blk
        rag.process(x[i:])
        self.assertEqual(one.n_windows_total, rag.n_windows_total)
        self.assertTrue(np.allclose(one.spectrum_db(), rag.spectrum_db(), atol=1e-6))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            SpectrumSurvey(0.0)
        with self.assertRaises(ValueError):
            SpectrumSurvey(FS, nfft=1000)
        with self.assertRaises(ValueError):
            SpectrumSurvey(FS, history=1)

    def test_survey_line_wording(self):
        line = SurveyLine(25_000.0, 21.3, 5.0, 1.0, 32)
        self.assertTrue(line.steady)
        self.assertIn("steady", line.describe())
        self.assertFalse(SurveyLine(25_000.0, 21.3, 5.0, 20.0, 32).steady)
        self.assertIn("varies", SurveyLine(25_000.0, 21.3, 5.0, 20.0, 32).describe())


if __name__ == "__main__":
    unittest.main()
