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

import unittest

import numpy as np

from friture.listen.agc import (
    DEFAULT_MAX_GAIN,
    DEFAULT_TARGET_RMS,
    Agc,
)
from friture.listen.audio_fifo import AudioFifo
from friture.listen.denoise import SpectralDenoiser
from friture.listen.gate import NoiseGate
from friture.listen.limiter import SoftLimiter
from friture.listen.band_dsp import (
    BANDPASS,
    HETERODYNE,
    N_TAPS_MAX,
    N_TAPS_MIN,
    BandpassFilter,
    HeterodyneFilter,
    band_error,
    group_delay_seconds,
    make_filter,
    min_bandwidth,
    taps_for_bandwidth,
    transition_width,
)
from friture.audiobackend import SAMPLING_RATE
from friture.listen.listen_band_view_model import (
    MIN_WIDTH_HZ,
    NYQUIST_HZ,
    ListenBandViewModel,
)

# The rate this build actually captures at. The DSP itself is rate-agnostic --
# every function takes fs -- but testing it anywhere other than where it runs
# would leave the interesting cases untested: at 250 kHz the filters are five
# times longer for the same bandwidth in hertz, and the tap ceiling bites.
FS = float(SAMPLING_RATE)
BLOCK = 2048


def tone(frequency, n, fs=FS, amplitude=1.0):
    return amplitude * np.cos(2.0 * np.pi * frequency * np.arange(n) / fs)


def amplitude_at(signal, frequency, fs=FS):
    """Amplitude of `signal` at `frequency`, by projection onto that tone.

    A plain DFT bin would smear a tone that does not land on a bin centre;
    projecting onto the exact frequency does not care where the bins are.
    """
    n = signal.size
    k = np.arange(n)
    reference = np.exp(-2j * np.pi * frequency * k / fs)
    return 2.0 * np.abs(np.vdot(reference.conj(), signal)) / n


def run_blocks(band_filter, signal, block=BLOCK):
    """Push `signal` through in blocks, as the audio path really does."""
    return np.concatenate(
        [band_filter.process(signal[i:i + block]) for i in range(0, signal.size, block)])


class TestFilterLength(unittest.TestCase):
    """The tap count follows the band, and the delay follows the tap count."""

    def test_the_longest_filter_sets_the_floor(self):
        self.assertAlmostEqual(min_bandwidth(FS), transition_width(FS, N_TAPS_MAX))
        self.assertIsNone(band_error(1000.0, min_bandwidth(FS), FS))
        self.assertIsNotNone(band_error(1000.0, min_bandwidth(FS) - 1.0, FS))

    def test_a_narrower_band_gets_a_longer_filter(self):
        lengths = [taps_for_bandwidth(width, FS) for width in (8000, 2000, 500, 100, 30)]
        self.assertEqual(lengths, sorted(lengths), "taps must not shrink as the band narrows")
        self.assertLess(lengths[0], lengths[-1])

    def test_a_wide_band_still_gets_a_usable_filter(self):
        # short enough to stay responsive, never shorter than the minimum
        self.assertEqual(taps_for_bandwidth(20000.0, FS), N_TAPS_MIN)

    def test_the_filter_length_is_capped(self):
        self.assertEqual(taps_for_bandwidth(1e-6, FS), N_TAPS_MAX)

    def test_every_allowed_band_gets_a_roll_off_that_fits_inside_it(self):
        for width in (MIN_WIDTH_HZ, 100, 500, 2000, 12000):
            taps = taps_for_bandwidth(width, FS)
            self.assertLessEqual(
                transition_width(FS, taps), width,
                "a %d Hz band would be nothing but skirt at %d taps" % (width, taps))

    def test_delay_is_reported_and_grows_with_the_filter(self):
        wide = BandpassFilter()
        wide.configure(8000.0, 4000.0, FS)
        narrow = BandpassFilter()
        narrow.configure(8000.0, float(MIN_WIDTH_HZ), FS)

        self.assertLess(wide.group_delay_s, narrow.group_delay_s)
        self.assertAlmostEqual(narrow.group_delay_s,
                               group_delay_seconds(narrow.n_taps, FS))
        # the floor must stay in a range a person would tolerate live
        self.assertLess(narrow.group_delay_s, 0.10)


class TestBandValidation(unittest.TestCase):

    def test_band_must_fit_under_nyquist(self):
        just_fits = FS / 2.0 - 4000.0
        self.assertIsNone(band_error(just_fits, 4000.0, FS))
        self.assertIsNotNone(band_error(just_fits, 4001.0, FS))

    def test_band_cannot_start_below_zero(self):
        self.assertIsNotNone(band_error(-1.0, 2000.0, FS))

    def test_configure_refuses_an_impossible_band(self):
        past_nyquist = FS / 2.0 - 1000.0
        with self.assertRaises(ValueError):
            BandpassFilter().configure(past_nyquist, 4000.0, FS)
        with self.assertRaises(ValueError):
            HeterodyneFilter().configure(past_nyquist, 4000.0, FS)


class TestBandpass(unittest.TestCase):

    def setUp(self):
        self.filter = BandpassFilter()
        self.filter.configure(8000.0, 2000.0, FS)  # 8-10 kHz

    def test_tone_inside_the_band_survives(self):
        out = run_blocks(self.filter, tone(9000.0, 8 * BLOCK))
        self.assertGreater(amplitude_at(out, 9000.0), 0.9)

    def test_tone_outside_the_band_is_rejected(self):
        out = run_blocks(self.filter, tone(4000.0, 8 * BLOCK))
        self.assertLess(amplitude_at(out, 4000.0), 0.01)  # better than -40 dB

    def test_pitch_is_unchanged(self):
        out = run_blocks(self.filter, tone(9000.0, 8 * BLOCK))
        # the tone comes out where it went in, not moved to baseband
        self.assertGreater(amplitude_at(out, 9000.0), 10.0 * amplitude_at(out, 1000.0))

    def test_a_band_starting_at_dc_keeps_unity_gain(self):
        # the two lowpass copies meet at DC, each at -6 dB; they must add
        # back to unity rather than double or cancel
        band_filter = BandpassFilter()
        band_filter.configure(0.0, 4000.0, FS)
        out = run_blocks(band_filter, tone(500.0, 8 * BLOCK))
        self.assertAlmostEqual(amplitude_at(out, 500.0), 1.0, delta=0.05)


class TestHeterodyne(unittest.TestCase):

    def setUp(self):
        self.filter = HeterodyneFilter()
        self.filter.configure(18000.0, 4000.0, FS)  # 18-22 kHz -> 0-4 kHz

    def test_tone_lands_at_its_offset_from_the_lower_edge(self):
        out = run_blocks(self.filter, tone(20000.0, 8 * BLOCK))
        # 18000 + 2000 -> 2000
        self.assertGreater(amplitude_at(out, 2000.0), 0.9)

    def test_the_original_frequency_is_gone(self):
        out = run_blocks(self.filter, tone(20000.0, 8 * BLOCK))
        self.assertLess(amplitude_at(out, 20000.0), 0.01)

    def test_the_mirror_image_is_rejected(self):
        """A tone BELOW the lower edge must not fold back into the band.

        This is what the one-sided complex filter buys: with real taps a
        tone at f_lo - d would come out at +d, indistinguishable from the
        wanted f_lo + d.
        """
        out = run_blocks(self.filter, tone(16000.0, 8 * BLOCK))  # f_lo - 2000
        self.assertLess(amplitude_at(out, 2000.0), 0.01)

    def test_content_above_the_band_is_rejected(self):
        out = run_blocks(self.filter, tone(23000.0, 8 * BLOCK))  # f_lo + 5000
        self.assertLess(amplitude_at(out, 5000.0), 0.01)


class TestStreamingContinuity(unittest.TestCase):
    """Filter state has to survive block boundaries, or every edge clicks."""

    def _assert_block_size_does_not_matter(self, mode):
        signal = tone(9000.0, 8 * BLOCK) + 0.5 * tone(3000.0, 8 * BLOCK)

        blocked = make_filter(mode)
        blocked.configure(8000.0, 2000.0, FS)
        one_shot = make_filter(mode)
        one_shot.configure(8000.0, 2000.0, FS)

        # not exact: overlap-save runs a different transform size for a
        # 333-sample block than for the whole signal at once, and at 250 kHz
        # those transforms are 65536 points, so there is more rounding to
        # accumulate. 1e-8 is about -160 dBFS.
        np.testing.assert_allclose(
            run_blocks(blocked, signal, block=333),
            one_shot.process(signal),
            rtol=1e-8, atol=1e-8)

    def test_bandpass_is_block_size_independent(self):
        self._assert_block_size_does_not_matter(BANDPASS)

    def test_heterodyne_is_block_size_independent(self):
        self._assert_block_size_does_not_matter(HETERODYNE)

    def test_no_step_at_a_block_boundary(self):
        band_filter = BandpassFilter()
        band_filter.configure(8000.0, 2000.0, FS)
        out = run_blocks(band_filter, tone(9000.0, 4 * BLOCK), block=BLOCK)
        # a lost tail would put a visible step exactly at the boundary
        steps = np.abs(np.diff(out))
        self.assertLess(steps[BLOCK - 1], 3.0 * np.median(steps))

    def test_empty_block_is_harmless(self):
        band_filter = BandpassFilter()
        band_filter.configure(8000.0, 2000.0, FS)
        self.assertEqual(band_filter.process(np.zeros(0)).size, 0)

    def test_output_length_always_matches_input(self):
        for mode in (BANDPASS, HETERODYNE):
            band_filter = make_filter(mode)
            band_filter.configure(8000.0, 2000.0, FS)
            for size in (1, 7, 255, 256, 1024):
                self.assertEqual(band_filter.process(np.zeros(size)).size, size)


class TestNarrowBands(unittest.TestCase):
    """The point of the long filters: picking one tone out of a close pair."""

    def test_a_100_hz_band_separates_tones_50_hz_apart(self):
        signal = tone(440.0, 32 * BLOCK) + tone(520.0, 32 * BLOCK)

        band_filter = BandpassFilter()
        band_filter.configure(390.0, 100.0, FS)  # 390-490 Hz, around 440
        out = run_blocks(band_filter, signal)

        # settle past the filter's own delay before measuring
        settled = out[8 * BLOCK:]
        self.assertGreater(amplitude_at(settled, 440.0), 0.8)
        self.assertLess(amplitude_at(settled, 520.0), 0.05)

    def test_the_narrowest_band_still_passes_its_own_tone(self):
        width = float(MIN_WIDTH_HZ)
        signal = tone(1000.0, 64 * BLOCK)

        band_filter = BandpassFilter()
        band_filter.configure(1000.0 - width / 2.0, width, FS)
        out = run_blocks(band_filter, signal)

        self.assertGreater(amplitude_at(out[16 * BLOCK:], 1000.0), 0.5)

    def test_the_narrowest_band_rejects_a_tone_just_outside_it(self):
        width = float(MIN_WIDTH_HZ)
        # a tone two widths above the centre is comfortably outside
        signal = tone(1000.0 + 2.0 * width, 64 * BLOCK)

        band_filter = BandpassFilter()
        band_filter.configure(1000.0 - width / 2.0, width, FS)
        out = run_blocks(band_filter, signal)

        self.assertLess(amplitude_at(out[16 * BLOCK:], 1000.0 + 2.0 * width), 0.1)

    def test_a_narrow_heterodyne_band_still_lands_where_expected(self):
        signal = tone(20050.0, 32 * BLOCK)

        band_filter = HeterodyneFilter()
        band_filter.configure(20000.0, 100.0, FS)  # 20.0-20.1 kHz -> 0-100 Hz
        out = run_blocks(band_filter, signal)

        self.assertGreater(amplitude_at(out[8 * BLOCK:], 50.0), 0.5)


class TestBandClamping(unittest.TestCase):

    def setUp(self):
        self.band = ListenBandViewModel()

    def test_a_click_centres_the_band(self):
        self.band.width_hz = 2000
        self.band.click_center(9000.0)
        self.assertEqual(self.band.centre_hz, 9000)
        self.assertAlmostEqual(self.band.f_lo, 8000.0)
        self.assertAlmostEqual(self.band.f_hi, 10000.0)

    def test_a_click_near_nyquist_keeps_the_width(self):
        self.band.width_hz = 4000
        self.band.click_center(NYQUIST_HZ)
        self.assertAlmostEqual(self.band.f_hi, NYQUIST_HZ)
        self.assertAlmostEqual(self.band.f_hi - self.band.f_lo, 4000.0)

    def test_a_click_near_dc_keeps_the_width(self):
        self.band.width_hz = 4000
        self.band.click_center(0.0)
        self.assertAlmostEqual(self.band.f_lo, 0.0)
        self.assertAlmostEqual(self.band.f_hi - self.band.f_lo, 4000.0)

    def test_resizing_keeps_the_centre_and_gives_up_width_instead(self):
        # the opposite priority to a move: you aimed here first, so a width
        # that will not fit is capped rather than sliding the target away
        self.band.click_center(5000.0)
        self.band.width_hz = 999999
        self.assertEqual(self.band.centre_hz, 5000)
        self.assertAlmostEqual(self.band.f_lo, 0.0)
        self.assertAlmostEqual(self.band.f_hi, 10000.0)

    def test_the_widest_possible_band_is_reachable(self):
        self.band.click_center(NYQUIST_HZ / 2.0)
        self.band.width_hz = 999999
        self.assertAlmostEqual(self.band.f_lo, 0.0)
        self.assertAlmostEqual(self.band.f_hi, NYQUIST_HZ)

    def test_the_width_floor_is_the_filter_transition(self):
        self.band.width_hz = 1
        self.assertEqual(self.band.width_hz, MIN_WIDTH_HZ)

    def test_every_reachable_band_is_filterable(self):
        """Clamping must guarantee what the filters demand."""
        for width in (MIN_WIDTH_HZ, 2000, 12000, 999999):
            for centre in (-5000.0, 0.0, 300.0, 12000.0, NYQUIST_HZ, 99999.0):
                self.band.width_hz = width
                self.band.click_center(centre)
                self.assertIsNone(
                    band_error(self.band.f_lo, self.band.f_hi - self.band.f_lo, 2 * NYQUIST_HZ),
                    "centre=%s width=%s" % (centre, width))

    def test_a_click_emits_one_band_change(self):
        changes = []
        self.band.band_changed.connect(lambda: changes.append(None))
        self.band.click_center(9000.0)
        self.assertEqual(len(changes), 1)
        # clicking the same place again is not a change
        self.band.click_center(9000.0)
        self.assertEqual(len(changes), 1)

    def test_turning_listening_off_stops_the_monitor(self):
        self.band.monitoring = True
        self.assertTrue(self.band.enabled)  # monitoring implies listening
        self.band.enabled = False
        self.assertFalse(self.band.monitoring)

    def test_status_text_shows_the_shift_only_for_heterodyne(self):
        self.band.width_hz = 4000
        self.band.click_center(20000.0)
        self.band.mode = BANDPASS
        self.assertNotIn("→", self.band.status_text)
        self.band.mode = HETERODYNE
        self.assertIn("→", self.band.status_text)


class TestEdgeDrag(unittest.TestCase):
    """Pulling an edge resizes around the centre."""

    def setUp(self):
        self.band = ListenBandViewModel()
        self.band.click_center(9000.0)
        self.band.width_hz = 2000

    def test_pulling_the_top_edge_out_widens_symmetrically(self):
        self.band.drag_edge(11000.0)
        self.assertEqual(self.band.centre_hz, 9000)
        self.assertAlmostEqual(self.band.f_lo, 7000.0)
        self.assertAlmostEqual(self.band.f_hi, 11000.0)

    def test_pulling_the_bottom_edge_works_the_same_way(self):
        self.band.drag_edge(7000.0)
        self.assertEqual(self.band.centre_hz, 9000)
        self.assertAlmostEqual(self.band.f_hi - self.band.f_lo, 4000.0)

    def test_pulling_an_edge_past_the_centre_does_not_invert_the_band(self):
        self.band.drag_edge(8500.0)
        self.assertAlmostEqual(self.band.f_hi - self.band.f_lo, 1000.0)
        self.assertGreater(self.band.f_hi, self.band.f_lo)

    def test_pulling_an_edge_onto_the_centre_stops_at_the_floor(self):
        self.band.drag_edge(9000.0)
        self.assertEqual(self.band.width_hz, MIN_WIDTH_HZ)
        self.assertEqual(self.band.centre_hz, 9000)

    def test_a_drag_near_an_edge_of_the_spectrum_caps_the_width(self):
        self.band.click_center(1000.0)
        self.band.drag_edge(20000.0)
        self.assertEqual(self.band.centre_hz, 1000)
        self.assertAlmostEqual(self.band.f_lo, 0.0)
        self.assertAlmostEqual(self.band.f_hi, 2000.0)


class TestRetuningWhileRunning(unittest.TestCase):
    """The point of the whole DDC restructure: dragging must not break audio.

    A click is a step in the waveform, so that is what these look for. The
    measure needs care: near Nyquist a perfectly good tone already jumps by
    more than its own amplitude between samples, and a click would hide
    behind that -- which is why every test here uses a low tone and states
    the highest frequency the output can reach.
    """

    @staticmethod
    def discontinuity(signal, highest_output_hz):
        """Largest sample-to-sample jump, against what the tone itself makes.

        A unit sinusoid at f changes by at most 2*pi*f/fs between samples, so
        a result near 1 means the output is a clean tone and anything well
        above it is a step. See test_the_measure_would_catch_a_rebuild for
        what this reads when the audio really does break.
        """
        natural = 2.0 * np.pi * highest_output_hz / FS
        return float(np.max(np.abs(np.diff(signal)))) / natural

    def _sweep(self, mode, tone_hz, f_los, widths, rebuild=False):
        """Feed a steady tone while walking the band, block by block."""
        blocks = len(f_los)
        signal = tone(tone_hz, blocks * BLOCK)

        band_filter = make_filter(mode)
        band_filter.configure(f_los[0], widths[0], FS)

        out = []
        for i in range(blocks):
            if rebuild:
                # what a naive implementation does on every band change:
                # throw the stream away and start again
                band_filter = make_filter(mode)
                band_filter.configure(f_los[i], widths[i], FS)
            else:
                band_filter.retune(f_los[i], widths[i])
            out.append(band_filter.process(signal[i * BLOCK:(i + 1) * BLOCK]))
        return np.concatenate(out)

    # The band walks up past a 400 Hz tone while keeping it comfortably
    # inside, so the output is a steady 400 Hz for band-pass and a glide down
    # from 400 Hz for heterodyne.
    MOVING = np.linspace(0.0, 300.0, 32)
    MOVING_WIDTHS = np.full(32, 2000.0)

    # The band tightens around a 1 kHz tone, crossing several filter lengths
    # on the way -- each one a new set of taps that has to be faded in.
    NARROWING_WIDTHS = np.linspace(2000.0, 600.0, 32)
    NARROWING = 1200.0 - NARROWING_WIDTHS / 2.0

    def test_moving_the_band_makes_no_step(self):
        for mode in (BANDPASS, HETERODYNE):
            out = self._sweep(mode, 400.0, self.MOVING, self.MOVING_WIDTHS)
            self.assertLess(self.discontinuity(out, 400.0), 3.0,
                            "mode %d stepped: a retune broke the waveform" % mode)

    def test_changing_the_width_makes_no_step(self):
        for mode in (BANDPASS, HETERODYNE):
            out = self._sweep(mode, 1000.0, self.NARROWING, self.NARROWING_WIDTHS)
            self.assertLess(self.discontinuity(out, 1000.0), 3.0,
                            "mode %d stepped: a width change broke the waveform" % mode)

    def test_the_measure_would_catch_a_rebuild(self):
        """Guard against the tests above passing because they measure nothing.

        Rebuilding the filter each block is exactly the click this feature
        exists to avoid; if that does not register, neither would a
        regression.
        """
        for mode in (BANDPASS, HETERODYNE):
            broken = self._sweep(mode, 400.0, self.MOVING, self.MOVING_WIDTHS, rebuild=True)
            self.assertGreater(self.discontinuity(broken, 400.0), 8.0,
                               "mode %d: a rebuilt stream should read as a click" % mode)

    def test_a_retune_that_does_not_move_changes_nothing(self):
        band_filter = BandpassFilter()
        band_filter.configure(8000.0, 2000.0, FS)
        signal = tone(9000.0, 8 * BLOCK)

        retuned = np.concatenate([
            (band_filter.retune(8000.0, 2000.0), band_filter.process(signal[i:i + BLOCK]))[1]
            for i in range(0, signal.size, BLOCK)])

        untouched = BandpassFilter()
        untouched.configure(8000.0, 2000.0, FS)
        np.testing.assert_allclose(retuned, untouched.process(signal), rtol=1e-9, atol=1e-9)

    def test_the_band_really_follows_the_retune(self):
        # two tones; retune from one to the other and check what comes out
        signal = tone(5000.0, 64 * BLOCK) + tone(15000.0, 64 * BLOCK)

        band_filter = BandpassFilter()
        band_filter.configure(4000.0, 2000.0, FS)  # on the 5 kHz tone
        first = np.concatenate([band_filter.process(signal[i:i + BLOCK])
                                for i in range(0, 32 * BLOCK, BLOCK)])

        band_filter.retune(14000.0, 2000.0)        # now on the 15 kHz one
        second = np.concatenate([band_filter.process(signal[i:i + BLOCK])
                                 for i in range(32 * BLOCK, 64 * BLOCK, BLOCK)])

        self.assertGreater(amplitude_at(first[8 * BLOCK:], 5000.0), 0.8)
        self.assertLess(amplitude_at(first[8 * BLOCK:], 15000.0), 0.05)
        self.assertGreater(amplitude_at(second[8 * BLOCK:], 15000.0), 0.8)
        self.assertLess(amplitude_at(second[8 * BLOCK:], 5000.0), 0.05)

    def test_retune_refuses_a_band_that_runs_past_nyquist(self):
        band_filter = BandpassFilter()
        band_filter.configure(8000.0, 2000.0, FS)
        with self.assertRaises(ValueError):
            band_filter.retune(FS / 2.0 - 1000.0, 4000.0)
        # and the refusal leaves the running filter alone
        self.assertGreater(
            amplitude_at(band_filter.process(tone(9000.0, 8 * BLOCK)), 9000.0), 0.8)

    def test_retune_before_configure_is_an_error(self):
        with self.assertRaises(RuntimeError):
            BandpassFilter().retune(8000.0, 2000.0)


class TestDragSoundsSmooth(unittest.TestCase):
    """End to end: drag the band the way the mouse does, listen for breaks.

    Above the filter level, so this covers the part that decides whether to
    retune or rebuild -- the piece that would quietly undo all of it by
    throwing the stream away on every mouse move.
    """

    def setUp(self):
        from friture.listen.processor import BandProcessor

        self.band = ListenBandViewModel()
        self.band.enabled = True
        # AGC and limiter off: these weigh amplitudes and sample-to-sample
        # steps, and either one moving its own gain would sit on top of both.
        # Both have their own tests.
        self.band.agc_enabled = False
        self.band.limiter_enabled = False
        self.processor = BandProcessor(None, self.band)

    def _play_while(self, tone_hz, adjust, blocks=32):
        """Feed a steady tone, moving the band between blocks as a drag does."""
        signal = tone(tone_hz, blocks * BLOCK)
        out = []
        for i in range(blocks):
            adjust(i / float(blocks - 1))
            out.append(self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK]))
        return np.concatenate(out)

    def test_dragging_the_band_across_a_tone_stays_smooth(self):
        for mode in (BANDPASS, HETERODYNE):
            self.band.mode = mode
            self.band.width_hz = 2000
            out = self._play_while(400.0, lambda t: self.band.click_center(1000.0 + 300.0 * t))
            self.assertLess(
                TestRetuningWhileRunning.discontinuity(out, 400.0), 3.0,
                "mode %d: moving the band broke the audio" % mode)

    def test_dragging_an_edge_stays_smooth(self):
        for mode in (BANDPASS, HETERODYNE):
            self.band.mode = mode
            self.band.click_center(1200.0)
            out = self._play_while(
                1000.0, lambda t: self.band.drag_edge(1200.0 + 1000.0 - 700.0 * t))
            self.assertLess(
                TestRetuningWhileRunning.discontinuity(out, 1000.0), 3.0,
                "mode %d: resizing the band broke the audio" % mode)

    def test_a_still_band_is_not_disturbed_by_being_re_read(self):
        self.band.mode = BANDPASS  # the tone has to come back out where it went in
        self.band.width_hz = 2000
        self.band.click_center(9000.0)
        held = self._play_while(9000.0, lambda t: None)
        self.assertGreater(amplitude_at(held[8 * BLOCK:], 9000.0), 0.8)

    def test_the_audio_ends_up_where_the_drag_left_the_band(self):
        signal = tone(5000.0, 96 * BLOCK) + tone(15000.0, 96 * BLOCK)
        self.band.mode = BANDPASS
        self.band.width_hz = 2000
        self.band.click_center(5000.0)

        for i in range(32):  # settle on the 5 kHz tone
            self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK])
        for i in range(32, 64):  # drag up to the 15 kHz one
            self.band.click_center(5000.0 + 10000.0 * (i - 32) / 31.0)
            self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK])

        # measured after the drag, not during it: mid-sweep the band is
        # somewhere between the two tones and passing neither
        landed = np.concatenate([self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK])
                                 for i in range(64, 96)])[8 * BLOCK:]

        self.assertGreater(amplitude_at(landed, 15000.0), 0.8)
        self.assertLess(amplitude_at(landed, 5000.0), 0.05)


class TestAgc(unittest.TestCase):
    """Bring a quiet band up, hold a loud one down, and never step.

    Signal lengths here are set by the 0.3 s release: at 250 kHz that is 73
    blocks per time constant, so anything measuring a settled level needs
    seconds of audio, not a handful of blocks.
    """

    # ~2 s, several release time constants
    SETTLE_BLOCKS = 256

    # An amplitude the ceiling can actually lift to the target: 12x of its
    # 0.035 RMS is 0.42, comfortably past 0.2. Quieter than 0.024 amplitude
    # and no amount of AGC reaches the target -- see the ceiling test.
    LIFTABLE = 0.05

    def setUp(self):
        self.agc = Agc(FS)

    def _run(self, signal, block=BLOCK):
        return np.concatenate(
            [self.agc.process(signal[i:i + block]) for i in range(0, signal.size, block)])

    @staticmethod
    def settled_rms(signal, fraction=0.75):
        """RMS of the tail, past whatever the AGC needed to converge."""
        return float(np.sqrt(np.mean(signal[int(signal.size * fraction):] ** 2)))

    def test_a_quiet_signal_is_lifted_toward_the_target(self):
        quiet = tone(9000.0, self.SETTLE_BLOCKS * BLOCK, amplitude=self.LIFTABLE)
        out = self._run(quiet)
        self.assertAlmostEqual(self.settled_rms(out), DEFAULT_TARGET_RMS, delta=0.02)

    def test_a_loud_signal_is_pulled_down_to_the_target(self):
        loud = tone(9000.0, self.SETTLE_BLOCKS * BLOCK, amplitude=0.9)
        out = self._run(loud)
        self.assertAlmostEqual(self.settled_rms(out), DEFAULT_TARGET_RMS, delta=0.02)

    def test_the_lift_stops_at_the_ceiling(self):
        """Below a certain level the AGC gives up rather than roaring.

        This is the ceiling ultraScan had to add after +40 dB turned a quiet
        band's noise floor into something that buried speech. The cost is
        real and is the point: a signal quieter than target/max_gain simply
        does not reach the target.
        """
        far_too_quiet = tone(9000.0, self.SETTLE_BLOCKS * BLOCK, amplitude=1e-4)
        out = self._run(far_too_quiet)
        self.assertLessEqual(self.agc.gain, DEFAULT_MAX_GAIN + 1e-9)
        self.assertLess(self.settled_rms(out), 0.1 * DEFAULT_TARGET_RMS)

    def test_it_comes_down_much_faster_than_it_goes_up(self):
        """Attack must beat release, or every loud arrival is a blast."""
        def blocks_to_settle(from_gain, signal_amplitude):
            self.agc.reset()
            self.agc._gain = from_gain
            target_gain = DEFAULT_TARGET_RMS / (signal_amplitude / np.sqrt(2.0))
            signal = tone(9000.0, BLOCK, amplitude=signal_amplitude)
            for count in range(1, 2000):
                self.agc.process(signal)
                if abs(self.agc.gain - target_gain) < 0.1 * abs(from_gain - target_gain):
                    return count
            return 2000

        falling = blocks_to_settle(from_gain=10.0, signal_amplitude=0.9)
        rising = blocks_to_settle(from_gain=1.0, signal_amplitude=self.LIFTABLE)

        self.assertLess(falling, rising / 5.0,
                        "attack (%d blocks) should be far quicker than release (%d)"
                        % (falling, rising))

    def test_it_does_not_step_at_block_boundaries(self):
        """The gain ramps within each block, so nothing jumps between them.

        The input's own level slides smoothly, so the AGC is working the whole
        time and any step in the output has to be its doing rather than the
        signal's.
        """
        n = 64 * BLOCK
        envelope = 0.05 + 0.45 * (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n))
        out = self._run(tone(400.0, n) * envelope)

        steps = np.abs(np.diff(out))
        boundaries = steps[BLOCK - 1::BLOCK]
        self.assertLess(float(np.max(boundaries)),
                        3.0 * float(np.percentile(steps, 99.9)))

    def test_the_measure_would_catch_a_per_block_gain(self):
        """Guard: the check above has to fail for an AGC that does not ramp."""
        n = 64 * BLOCK
        envelope = 0.05 + 0.45 * (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n))
        signal = tone(400.0, n) * envelope

        stepped = Agc(FS)
        out = []
        for i in range(0, n, BLOCK):
            block = signal[i:i + BLOCK]
            stepped.process(block)            # advance the gain as usual...
            out.append(block * stepped.gain)  # ...but apply it flat
        out = np.concatenate(out)

        steps = np.abs(np.diff(out))
        boundaries = steps[BLOCK - 1::BLOCK]
        self.assertGreater(float(np.max(boundaries)),
                           3.0 * float(np.percentile(steps, 99.9)))

    def test_block_size_does_not_change_the_outcome(self):
        """The time constants use the real block length, so they must not."""
        signal = tone(9000.0, self.SETTLE_BLOCKS * BLOCK, amplitude=self.LIFTABLE)

        big, small = Agc(FS), Agc(FS)
        big_out = np.concatenate([big.process(signal[i:i + BLOCK])
                                  for i in range(0, signal.size, BLOCK)])
        small_out = np.concatenate([small.process(signal[i:i + 512])
                                    for i in range(0, signal.size, 512)])

        self.assertAlmostEqual(self.settled_rms(big_out), self.settled_rms(small_out), delta=0.01)

    def test_an_empty_block_is_harmless(self):
        self.assertEqual(self.agc.process(np.zeros(0)).size, 0)

    def test_reset_returns_to_unity(self):
        self._run(tone(9000.0, 64 * BLOCK, amplitude=self.LIFTABLE))
        self.assertGreater(self.agc.gain, 1.5)
        self.agc.reset()
        self.assertEqual(self.agc.gain, 1.0)

    def test_silence_does_not_divide_by_zero(self):
        out = self._run(np.zeros(8 * BLOCK))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_bad_settings_are_refused(self):
        for kwargs in ({"target_rms": 0.0}, {"max_gain": 0.0}, {"attack_s": -1.0}):
            with self.assertRaises(ValueError):
                Agc(FS, **kwargs)
        with self.assertRaises(ValueError):
            Agc(0.0)


class TestAgcInTheListeningChain(unittest.TestCase):
    """The AGC has to be reachable from the switch in the bar."""

    QUIET = 0.05

    def setUp(self):
        from friture.listen.processor import BandProcessor

        self.band = ListenBandViewModel()
        self.band.enabled = True
        self.band.mode = BANDPASS
        self.band.width_hz = 2000
        self.band.click_center(9000.0)
        self.processor = BandProcessor(None, self.band)

    def _play(self, amplitude, blocks=256):
        # from a clean start each time: the AGC carries its gain across calls,
        # so a second measurement would begin already converged
        self.processor.reset()
        signal = tone(9000.0, blocks * BLOCK, amplitude=amplitude)
        out = np.concatenate([self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK])
                              for i in range(blocks)])
        return float(np.sqrt(np.mean(out[int(out.size * 0.75):] ** 2)))

    def test_a_quiet_band_comes_out_usable_with_agc_on(self):
        self.band.agc_enabled = True
        self.assertAlmostEqual(self._play(self.QUIET), DEFAULT_TARGET_RMS, delta=0.03)

    def test_the_same_band_stays_quiet_with_agc_off(self):
        self.band.agc_enabled = False
        self.assertAlmostEqual(self._play(self.QUIET), self.QUIET / np.sqrt(2.0), delta=0.01)

    def test_the_manual_gain_still_trims_on_top_of_the_agc(self):
        self.band.agc_enabled = True
        at_unity = self._play(self.QUIET)
        self.band.gain_db = 6
        self.assertAlmostEqual(self._play(self.QUIET) / at_unity, 2.0, delta=0.15)

    def test_the_readout_reports_what_the_agc_is_adding(self):
        self.band.agc_enabled = True
        self._play(self.QUIET)
        # 0.035 RMS to a 0.2 target is about +15 dB, under the +21.6 ceiling
        self.assertAlmostEqual(self.processor.agc_gain_db, 15.1, delta=1.5)


class TestSpectralDenoiser(unittest.TestCase):
    """Remove what is always there, keep what is not."""

    def _run(self, signal, denoiser=None, block=BLOCK):
        denoiser = denoiser or SpectralDenoiser(FS)
        return np.concatenate(
            [denoiser.process(signal[i:i + block]) for i in range(0, signal.size, block)])

    def test_it_returns_as_many_samples_as_it_is_given(self):
        denoiser = SpectralDenoiser(FS)
        for size in (1, 100, BLOCK, 3 * BLOCK):
            self.assertEqual(denoiser.process(np.zeros(size)).size, size)

    def test_it_rebuilds_the_signal_exactly_when_nothing_is_removed(self):
        """Overlap-add has to be transparent before it can be useful."""
        pass_everything = SpectralDenoiser(
            FS, open_db=-200.0, knee_db=1.0, floor_db=0.0, smooth_bins=0,
            gain_attack_s=0.0, gain_release_s=0.0)
        signal = tone(3000.0, 32 * BLOCK)
        out = self._run(signal, pass_everything)

        lag = pass_everything.frame - pass_everything.hop
        settled = 4 * pass_everything.frame
        np.testing.assert_allclose(out[settled:], signal[settled - lag:-lag],
                                   rtol=1e-9, atol=1e-9)

    def test_the_delay_does_not_depend_on_the_block_size(self):
        signal = tone(3000.0, 32 * BLOCK)
        reference = None
        for block in (512, BLOCK, 4096):
            passthrough = SpectralDenoiser(
                FS, open_db=-200.0, knee_db=1.0, floor_db=0.0, smooth_bins=0,
                gain_attack_s=0.0, gain_release_s=0.0)
            out = self._run(signal, passthrough, block=block)
            if reference is None:
                reference = out
            else:
                np.testing.assert_allclose(out, reference, rtol=1e-9, atol=1e-9)

    def test_a_steady_tone_is_learned_and_removed(self):
        """The measured problem: a 25 kHz interferer that never stops."""
        interference = tone(25000.0, 128 * BLOCK, amplitude=0.3)
        out = self._run(interference)

        before = amplitude_at(out[:8 * BLOCK], 25000.0)
        after = amplitude_at(out[-16 * BLOCK:], 25000.0)
        self.assertGreater(before, 0.1, "should pass before it has been learned")
        # 10 dB, not 'gone': the gain floor is -18 dB on purpose, because
        # attenuating all the way to silence is what makes the survivors
        # audible as musical noise. Measured in this room the interferer
        # stands 12 dB above the floor, so 10 dB puts it under.
        attenuation_db = 20.0 * np.log10(before / max(after, 1e-12))
        self.assertGreater(attenuation_db, 10.0,
                           "a constant tone should sink into the background")

    def test_a_new_sound_survives_the_background_it_arrives_over(self):
        """A call has to get through interference that was there first."""
        n = 128 * BLOCK
        interference = tone(25000.0, n, amplitude=0.3)
        call = np.zeros(n)
        burst = slice(96 * BLOCK, 100 * BLOCK)
        call[burst] = tone(60000.0, 4 * BLOCK, amplitude=0.3)

        out = self._run(interference + call)
        self.assertGreater(amplitude_at(out[burst], 60000.0), 0.1,
                           "the new sound was subtracted along with the old")

    def test_a_long_sound_is_not_gradually_absorbed(self):
        """The guard that slows the estimate while a bin is loud."""
        n = 256 * BLOCK
        signal = tone(60000.0, n, amplitude=0.3)
        out = self._run(signal)

        early = amplitude_at(out[16 * BLOCK:32 * BLOCK], 60000.0)
        late = amplitude_at(out[-16 * BLOCK:], 60000.0)
        self.assertGreater(late, 0.5 * early,
                           "a sound lasting seconds was adopted as background")

    def test_silence_stays_finite(self):
        out = self._run(np.zeros(16 * BLOCK))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_a_frame_that_is_not_a_multiple_of_the_overlap_is_refused(self):
        with self.assertRaises(ValueError):
            SpectralDenoiser(FS, frame=1001)
        with self.assertRaises(ValueError):
            SpectralDenoiser(0.0)


class TestNoiseGate(unittest.TestCase):
    """Silence between the calls, without clipping them."""

    def _run(self, signal, gate=None, block=BLOCK):
        gate = gate or NoiseGate(FS)
        return np.concatenate(
            [gate.process(signal[i:i + block]) for i in range(0, signal.size, block)])

    @staticmethod
    def rms(x):
        return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    def test_steady_quiet_noise_is_ducked(self):
        rng = np.random.default_rng(1)
        hiss = rng.standard_normal(128 * BLOCK) * 0.01
        out = self._run(hiss)
        self.assertLess(self.rms(out[-16 * BLOCK:]), 0.2 * self.rms(hiss))

    def test_a_burst_over_that_noise_gets_through(self):
        rng = np.random.default_rng(1)
        n = 128 * BLOCK
        signal = rng.standard_normal(n) * 0.01
        burst = slice(96 * BLOCK, 100 * BLOCK)
        signal[burst] += tone(60000.0, 4 * BLOCK, amplitude=0.5)

        out = self._run(signal)
        # measured past the attack, which is 2 ms
        opened = out[96 * BLOCK + BLOCK:100 * BLOCK]
        self.assertGreater(self.rms(opened), 0.5 * 0.5 / np.sqrt(2.0))

    def test_it_holds_open_through_a_dip(self):
        """A call's tail is quieter than its peak and must not be chopped.

        Ambient noise comes first, as it does in use: the floor is learned
        from the quiet, not from whatever happened to be playing when the
        gate was switched on.
        """
        rng = np.random.default_rng(1)
        n = 160 * BLOCK
        signal = rng.standard_normal(n) * 0.005

        call = slice(128 * BLOCK, 136 * BLOCK)
        signal[call] += tone(60000.0, 8 * BLOCK, amplitude=0.5)
        # then a tail that decays to near the noise, which a gate without a
        # hold would cut off part way down
        tail = slice(136 * BLOCK, 144 * BLOCK)
        decay = np.linspace(1.0, 0.02, 8 * BLOCK)
        signal[tail] += tone(60000.0, 8 * BLOCK, amplitude=0.5) * decay

        out = self._run(signal)

        quiet_before = self.rms(out[120 * BLOCK:128 * BLOCK])
        late_tail = self.rms(out[140 * BLOCK:142 * BLOCK])
        self.assertGreater(late_tail, 5.0 * max(quiet_before, 1e-9),
                           "the gate shut on the tail instead of holding")

    def test_it_does_not_step_at_block_boundaries(self):
        rng = np.random.default_rng(1)
        n = 64 * BLOCK
        signal = rng.standard_normal(n) * 0.01
        signal[32 * BLOCK:36 * BLOCK] += tone(60000.0, 4 * BLOCK, amplitude=0.5)

        out = self._run(signal)
        steps = np.abs(np.diff(out))
        boundaries = steps[BLOCK - 1::BLOCK]
        self.assertLess(float(np.max(boundaries)),
                        3.0 * float(np.percentile(steps, 99.9)))

    def test_it_reports_whether_it_is_open(self):
        gate = NoiseGate(FS)
        rng = np.random.default_rng(1)
        for _ in range(64):
            gate.process(rng.standard_normal(BLOCK) * 0.01)
        self.assertFalse(gate.is_open)

        gate.process(tone(60000.0, BLOCK, amplitude=0.5))
        self.assertTrue(gate.is_open)

    def test_a_close_threshold_above_the_open_one_is_refused(self):
        with self.assertRaises(ValueError):
            NoiseGate(FS, open_db=6.0, close_db=12.0)


class TestNoiseReductionInTheListeningChain(unittest.TestCase):
    """The switches in the bar have to reach the audio."""

    def setUp(self):
        from friture.listen.processor import BandProcessor

        self.band = ListenBandViewModel()
        self.band.enabled = True
        self.band.mode = BANDPASS
        self.band.agc_enabled = False
        self.band.width_hz = 4000
        self.band.click_center(25000.0)
        self.processor = BandProcessor(None, self.band)

    def _play(self, signal):
        return np.concatenate([self.processor.process(signal[i:i + BLOCK])
                               for i in range(0, signal.size, BLOCK)])

    def test_denoise_off_leaves_the_interference_alone(self):
        self.band.denoise_enabled = False
        out = self._play(tone(25000.0, 128 * BLOCK, amplitude=0.3))
        self.assertGreater(amplitude_at(out[-16 * BLOCK:], 25000.0), 0.2)

    def test_denoise_on_learns_it_away(self):
        self.band.denoise_enabled = True
        out = self._play(tone(25000.0, 128 * BLOCK, amplitude=0.3))
        self.assertLess(amplitude_at(out[-16 * BLOCK:], 25000.0), 0.1)

    def test_the_gate_silences_the_gaps(self):
        rng = np.random.default_rng(1)
        self.band.gate_enabled = True
        hiss = rng.standard_normal(128 * BLOCK) * 0.02
        out = self._play(hiss)
        quiet = float(np.sqrt(np.mean(out[-16 * BLOCK:] ** 2)))

        self.band.gate_enabled = False
        self.processor.reset()
        loud = float(np.sqrt(np.mean(self._play(hiss)[-16 * BLOCK:] ** 2)))

        self.assertLess(quiet, 0.3 * loud)

    def test_the_agc_holds_its_gain_while_the_gate_is_shut(self):
        """Otherwise it winds up on silence and blasts the next call."""
        from friture.listen.agc import Agc

        agc = Agc(FS)
        quiet = tone(9000.0, BLOCK, amplitude=0.001)
        for _ in range(200):
            agc.process(quiet, adapt=False)
        self.assertAlmostEqual(agc.gain, 1.0, places=6)

        for _ in range(200):
            agc.process(quiet, adapt=True)
        self.assertGreater(agc.gain, 5.0)


class TestGainRangeAndClipping(unittest.TestCase):
    """The gain reaches far enough to distort, so it must say when it does."""

    QUIET = 0.0002   # about -77 dBFS, the level a quiet ultrasonic band sits at

    def setUp(self):
        from friture.listen.processor import BandProcessor

        self.band = ListenBandViewModel()
        self.band.enabled = True
        self.band.mode = BANDPASS
        self.band.agc_enabled = False
        # The clip warning is what happens when nothing is holding the level
        # back, so these run with the limiter out of the way. That the
        # limiter prevents all of this is the last test in the class.
        self.band.limiter_enabled = False
        self.band.width_hz = 2000
        self.band.click_center(9000.0)
        self.processor = BandProcessor(None, self.band)

    def _peak_at(self, gain_db, blocks=32):
        self.band.gain_db = gain_db
        self.processor.reset()
        self.processor.take_peak()
        signal = tone(9000.0, blocks * BLOCK, amplitude=self.QUIET)
        for i in range(blocks):
            self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK])
        return self.processor.take_peak()

    def test_the_gain_reaches_the_top_of_the_slider(self):
        self.band.gain_db = 150
        self.assertEqual(self.band.gain_db, 150)

    def test_more_gain_really_is_more_output(self):
        modest = self._peak_at(40)
        more = self._peak_at(60)
        self.assertAlmostEqual(more / modest, 10.0, delta=0.5)

    def test_a_quiet_band_needs_about_60_db_to_be_comfortable(self):
        """Which is why 40 was not enough -- the reason for the change."""
        self.assertLess(self._peak_at(40), 0.1)
        self.assertGreater(self._peak_at(60), 0.1)
        self.assertLess(self._peak_at(60), 1.0)

    def test_clipping_is_detected_once_the_output_passes_full_scale(self):
        self.assertLess(self._peak_at(60), 1.0)
        self.assertGreater(self._peak_at(80), 1.0)
        self.assertGreater(self._peak_at(150), 1.0)

    def test_the_peak_meter_resets_when_read(self):
        self._peak_at(150)
        self.assertEqual(self.processor.take_peak(), 0.0)

    def test_the_trim_is_exact(self):
        """20 dB should be ten times, not approximately ten times."""
        self.assertAlmostEqual(self._peak_at(20) / self._peak_at(0), 10.0, delta=0.1)

    def test_the_limiter_takes_all_of_this_away(self):
        self.band.limiter_enabled = True
        self.assertLessEqual(self._peak_at(150), 1.0)


class TestSoftLimiter(unittest.TestCase):
    """Hold the ceiling by turning down, and leave the waveform alone."""

    def _run(self, signal, limiter=None, block=BLOCK):
        limiter = limiter or SoftLimiter(FS)
        out = np.concatenate(
            [limiter.process(signal[i:i + block]) for i in range(0, signal.size, block)])
        return out, limiter

    def test_it_returns_as_many_samples_as_it_is_given(self):
        limiter = SoftLimiter(FS)
        for size in (1, 100, BLOCK, 3 * BLOCK):
            self.assertEqual(limiter.process(np.zeros(size)).size, size)

    def test_a_signal_already_under_the_ceiling_is_untouched(self):
        quiet = tone(3000.0, 32 * BLOCK, amplitude=0.5)
        out, limiter = self._run(quiet)
        lag = limiter.latency_samples
        np.testing.assert_allclose(out[8 * BLOCK:], quiet[8 * BLOCK - lag:-lag],
                                   rtol=1e-12, atol=1e-12)

    def test_nothing_gets_past_the_ceiling(self):
        """Including the cases a limiter without look-ahead would miss."""
        base = tone(3000.0, 64 * BLOCK, amplitude=0.01)
        cases = {
            "a long burst": lambda s: s.__setitem__(
                slice(32 * BLOCK, 34 * BLOCK), s[32 * BLOCK:34 * BLOCK] + 400.0),
            "a burst shorter than the look-ahead": lambda s: s.__setitem__(
                slice(32 * BLOCK, 32 * BLOCK + 100), s[32 * BLOCK:32 * BLOCK + 100] + 400.0),
            "a burst ending mid-segment": lambda s: s.__setitem__(
                slice(32 * BLOCK, 32 * BLOCK + 1013), s[32 * BLOCK:32 * BLOCK + 1013] + 400.0),
        }
        for name, damage in cases.items():
            signal = base.copy()
            damage(signal)
            out, limiter = self._run(signal)
            self.assertLessEqual(float(np.max(np.abs(out))), limiter.ceiling * 1.001,
                                 "%s got past the ceiling" % name)

    @staticmethod
    def harmonic_distortion(signal, fundamental_hz):
        """Total harmonic distortion, as a percentage.

        Windowed on purpose: projecting onto exact frequencies lets a strong
        fundamental leak into every harmonic bin, which reads as distortion
        that is not there -- it measured 0.06% on a signal with none.
        """
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
        freqs = np.fft.rfftfreq(signal.size, 1.0 / FS)

        def energy(centre, width=200.0):
            return float(np.sum(spectrum[(freqs >= centre - width) & (freqs <= centre + width)]))

        harmonics = sum(energy(k * fundamental_hz)
                        for k in range(2, 40) if k * fundamental_hz < FS / 2)
        return 100.0 * np.sqrt(harmonics / energy(fundamental_hz))

    def test_a_steady_overload_costs_no_distortion_at_all(self):
        """The point of turning down instead of cutting off.

        The same signal through a hard clip measures about 47% -- a square
        wave -- so this is not a marginal improvement.
        """
        over = tone(3000.0, 64 * BLOCK, amplitude=100.0)
        out, _ = self._run(over)
        self.assertLess(self.harmonic_distortion(out[16 * BLOCK:], 3000.0), 0.1)

    def test_the_measure_would_catch_a_hard_clip(self):
        """Guard: the check above has to fail for the thing it replaced."""
        over = tone(3000.0, 64 * BLOCK, amplitude=100.0)
        clipped = np.clip(over, -1.0, 1.0)
        self.assertGreater(self.harmonic_distortion(clipped[16 * BLOCK:], 3000.0), 30.0)

    def test_it_recovers_after_the_peak_has_gone(self):
        """Transparent again once the peak is past, not stuck turned down."""
        signal = tone(3000.0, 96 * BLOCK, amplitude=0.1)
        signal[16 * BLOCK:18 * BLOCK] *= 400.0
        out, limiter = self._run(signal)

        held = float(np.sqrt(np.mean(out[17 * BLOCK:18 * BLOCK] ** 2)))
        self.assertAlmostEqual(held, limiter.ceiling / np.sqrt(2.0), delta=0.05,
                               msg="the peak should have been held at the ceiling")

        # long afterwards the quiet signal should be passing at its own level
        untouched = 0.1 / np.sqrt(2.0)
        recovered = float(np.sqrt(np.mean(out[80 * BLOCK:88 * BLOCK] ** 2)))
        self.assertAlmostEqual(recovered, untouched, delta=0.005,
                               msg="the gain stayed down long after the peak")

    def test_it_reports_how_far_it_is_turning_down(self):
        over = tone(3000.0, 32 * BLOCK, amplitude=95.0)  # 100x the ceiling
        _, limiter = self._run(over)
        self.assertAlmostEqual(limiter.reduction_db, -40.0, delta=1.0)

    def test_bad_settings_are_refused(self):
        for kwargs in ({"ceiling": 0.0}, {"ceiling": 1.5}, {"segment": 0}):
            with self.assertRaises(ValueError):
                SoftLimiter(FS, **kwargs)
        with self.assertRaises(ValueError):
            SoftLimiter(0.0)


class TestLimiterInTheListeningChain(unittest.TestCase):

    def setUp(self):
        from friture.listen.processor import BandProcessor

        self.band = ListenBandViewModel()
        self.band.enabled = True
        self.band.mode = BANDPASS
        self.band.agc_enabled = False
        self.band.width_hz = 2000
        self.band.click_center(9000.0)
        self.band.gain_db = 100          # far past what the signal needs
        self.processor = BandProcessor(None, self.band)

    def _play(self, blocks=32):
        signal = tone(9000.0, blocks * BLOCK, amplitude=0.0002)
        out = np.concatenate([self.processor.process(signal[i * BLOCK:(i + 1) * BLOCK])
                              for i in range(blocks)])
        return out

    def test_the_limiter_keeps_the_output_in_range(self):
        self.band.limiter_enabled = True
        self.assertLessEqual(float(np.max(np.abs(self._play()))), 1.0)

    def test_without_it_the_output_runs_far_past_full_scale(self):
        self.band.limiter_enabled = False
        self.assertGreater(float(np.max(np.abs(self._play()))), 1.0)

    def test_the_reduction_meter_reports_something_to_look_at(self):
        self.band.limiter_enabled = True
        self.processor.take_reduction_db()
        self._play()
        self.assertLess(self.processor.take_reduction_db(), -10.0)

    def test_the_meter_resets_when_read(self):
        self.band.limiter_enabled = True
        self._play()
        self.processor.take_reduction_db()
        self.assertEqual(self.processor.take_reduction_db(), 0.0)


class TestAudioFifo(unittest.TestCase):

    def test_it_primes_before_delivering_anything(self):
        fifo = AudioFifo(capacity=100, prebuffer=50)
        fifo.push(np.ones(10, dtype=np.float32))
        out = np.empty(10, dtype=np.float32)
        self.assertEqual(fifo.pop_into(out), 0)
        np.testing.assert_array_equal(out, np.zeros(10))

        fifo.push(np.ones(50, dtype=np.float32))
        self.assertEqual(fifo.pop_into(out), 10)
        np.testing.assert_array_equal(out, np.ones(10))

    def test_it_keeps_fifo_order_across_the_wrap(self):
        fifo = AudioFifo(capacity=16, prebuffer=0)
        expected = np.arange(40, dtype=np.float32)
        out = np.empty(8, dtype=np.float32)
        popped = []
        for i in range(0, 40, 8):
            fifo.push(expected[i:i + 8])
            fifo.pop_into(out)
            popped.append(out.copy())
        np.testing.assert_array_equal(np.concatenate(popped), expected)

    def test_overflow_drops_and_counts(self):
        fifo = AudioFifo(capacity=16, prebuffer=0)
        self.assertEqual(fifo.push(np.ones(20, dtype=np.float32)), 16)
        self.assertEqual(fifo.n_dropped, 4)

    def test_underrun_zero_fills_counts_and_re_primes(self):
        fifo = AudioFifo(capacity=100, prebuffer=10)
        fifo.push(np.ones(20, dtype=np.float32))
        out = np.empty(30, dtype=np.float32)
        self.assertEqual(fifo.pop_into(out), 20)
        np.testing.assert_array_equal(out[20:], np.zeros(10))
        self.assertEqual(fifo.n_underruns, 1)

        # back to priming: a short push must not be delivered yet
        fifo.push(np.ones(5, dtype=np.float32))
        self.assertEqual(fifo.pop_into(out), 0)

    def test_clear_drops_everything_and_re_primes(self):
        fifo = AudioFifo(capacity=100, prebuffer=10)
        fifo.push(np.ones(50, dtype=np.float32))
        fifo.clear()
        self.assertEqual(fifo.occupancy, 0)
        out = np.empty(10, dtype=np.float32)
        self.assertEqual(fifo.pop_into(out), 0)


if __name__ == '__main__':
    unittest.main()
