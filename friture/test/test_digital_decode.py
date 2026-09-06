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

"""friture.demod: clock recovery, slicing, the detectors, and the live chain.

All synthetic. The core tests are ultraScan's test_symbols.py and
test_demod_digital.py carried over; the BandDecoder tests at the end are new
and run the whole chain at the capture rate, 250 kHz, in 2048-sample blocks
as the audio backend delivers them.

The tests that matter most are the ones that pin what must NOT happen:

  noise_reports_no_clock             noise must produce "no clock", not a baud
  lowest_line_not_the_tallest        a pattern whose 2nd harmonic dominates
                                     must still report the fundamental
  dbpsk_lock_catches_a_carrier_error the case where clock confidence AND
                                     per-symbol agreement are both high and
                                     the bits are still garbage
  noise_never_shows_bits             the same, through the live chain
"""

import unittest

import numpy as np

from friture.demod.burst import BurstDetector
from friture.demod.ddc import ComplexDdc
from friture.demod.decoder import (
    AM, ANALOG_MODES, DBPSK, DIGITAL_MODES, DQPSK, FM, FSK, FSK4, MODES, OOK, PM, BandDecoder,
)
from friture.demod.detectors import (
    AmDemodulator,
    DbpskDemodulator,
    DpskDemodulator,
    DqpskDemodulator,
    FmDemodulator,
    FskDemodulator,
    PmDemodulator,
    estimate_fsk_tones,
)
from friture.demod.symbols import (
    MIN_TRANSITIONS,
    SymbolClockTracker,
    SymbolSlicer,
    bits_to_text,
    differential_bits,
    estimate_symbol_clock,
    symbols_to_bits,
    transition_train,
)

FS = 50_000.0
# Ragged block sizes; the 0 slips an EMPTY read into the middle of the stream.
SIZES = [777, 0, 1, 4095, 1234, None]


def make_trace(baud, n_sym, *, phase=0.0, seed=0, bits=None, fs=FS):
    """A per-sample symbol trace at a known baud/phase, plus the bits behind it."""
    if bits is None:
        bits = np.random.default_rng(seed).integers(0, 2, n_sym)
    bits = np.asarray(bits, dtype=np.int16)
    T = fs / baud
    n = int(len(bits) * T)
    idx = np.clip(((np.arange(n) - phase) / T).astype(int), 0, len(bits) - 1)
    return bits[idx], bits


def make_bpsk(fc, baud, n_sym, *, fs=FS, phase0=0.7, seed=0, snr_db=None):
    """Differentially-encoded BPSK on a complex carrier; returns (z, the sent bits)."""
    r = np.random.default_rng(seed)
    bits = r.integers(0, 2, n_sym)
    sym = np.cumsum(bits) % 2                      # differential encoding
    T = fs / baud
    n = int(n_sym * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, n_sym - 1)
    z = np.exp(1j * (2 * np.pi * fc * np.arange(n) / fs + phase0)) * (1 - 2 * sym[idx])
    if snr_db is not None:
        amp = 10.0 ** (-snr_db / 20.0)
        z = z + amp * (r.normal(size=n) + 1j * r.normal(size=n)) / np.sqrt(2)
    return z, bits


def best_accuracy(got, truth, span=3):
    """Agreement allowing a +-span SYMBOL-NUMBERING offset.

    Which symbol the slicer calls number 0 depends on where the trace
    happened to start relative to the recovered grid; it is bookkeeping, not
    correctness, so the tests pin the bits and not the numbering.
    """
    best = 0.0
    for o in range(-span, span + 1):
        a, b = got[max(0, -o):], truth[max(0, o):]
        m = min(a.size, b.size)
        if m > 10:
            best = max(best, float(np.mean(a[:m] == b[:m])))
    return best


def split(dm, z, sizes=SIZES):
    out, i = [], 0
    for blk in sizes[:-1]:
        out.append(dm.process(z[i:i + blk]))
        i += blk
    out.append(dm.process(z[i:]))
    return out


def tone(f0, n, fs=FS):
    return np.exp(2j * np.pi * f0 * np.arange(n) / fs)


def fsk_signal(f_a, f_b, sym_len, n_sym, fs=FS):
    """Continuous-phase 2-FSK: alternating tones, no phase discontinuity."""
    per_sym = np.where(np.arange(n_sym) % 2 == 0, f_a, f_b).astype(np.float64)
    f_inst = np.repeat(per_sym, sym_len)
    return np.exp(2j * np.pi * np.cumsum(f_inst) / fs), f_inst


# -- the clock estimate ---------------------------------------------------

class ClockEstimateTest(unittest.TestCase):

    def test_rate_recovered_without_being_told(self):
        for baud in (200.0, 1000.0, 4800.0):
            with self.subTest(baud=baud):
                sym, _ = make_trace(baud, int(0.4 * baud) + 40, phase=13.7, seed=1)
                est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
                self.assertAlmostEqual(est.rate_hz, baud, delta=baud * 1e-3)
                self.assertGreater(est.confidence_db, 15.0)
                self.assertAlmostEqual(est.period_samples, FS / baud, delta=FS / baud * 1e-3)

    def test_timing_phase_recovered(self):
        """Within one sample: the train marks the first sample OF the new
        symbol, so a sub-sample bias of up to +1 is expected and harmless."""
        for baud in (200.0, 1000.0, 4800.0):
            with self.subTest(baud=baud):
                T = FS / baud
                sym, _ = make_trace(baud, int(0.4 * baud) + 40, phase=13.7, seed=1)
                est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
                err = ((est.phase_samples - 13.7) % T + T / 2) % T - T / 2
                self.assertLess(abs(err), 1.0, "phase off by %s samples of %s" % (err, T))

    def test_noise_reports_no_clock(self):
        """The anti-fooling test. A trace with no grid must not get a baud."""
        rng = np.random.default_rng(0)
        white = (rng.random(20_000) < 0.5).astype(np.int16)
        walk = np.cumsum(rng.normal(size=20_000))
        walk = (walk > np.median(walk)).astype(np.int16)

        real, _ = make_trace(1000.0, 400, phase=13.7, seed=1)
        real_conf = estimate_symbol_clock(
            real, None, FS, baud_min=50.0, baud_max=10_000.0).confidence_db

        for name, trace in (("white", white), ("walk", walk)):
            with self.subTest(noise=name):
                est = estimate_symbol_clock(trace, None, FS, baud_min=50.0, baud_max=10_000.0)
                self.assertLess(est.confidence_db, 15.0, "%s noise scored %s dB" % (name, est.confidence_db))
                self.assertLess(est.confidence_db, real_conf - 8.0)

    def test_lowest_line_not_the_tallest(self):
        """An alternating pattern has a 2nd harmonic as tall as its fundamental.
        Picking the maximum would report 2x the baud."""
        n_sym = 400
        bits = (np.arange(n_sym) % 2).astype(np.int16)      # 0101...: every boundary
        sym, _ = make_trace(1000.0, n_sym, phase=13.7, bits=bits)
        est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
        self.assertAlmostEqual(est.rate_hz, 1000.0, delta=1.0)

    def test_too_few_transitions_is_no_clock(self):
        sym = np.zeros(20_000, dtype=np.int16)
        sym[::2000] = 1                                     # 10 impulses, 20 edges
        est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
        self.assertLess(est.n_transitions, MIN_TRANSITIONS)
        self.assertEqual(est.rate_hz, 0.0)
        self.assertEqual(est.confidence_db, 0.0)

    def test_gated_edges_do_not_become_a_clock(self):
        """An edge into or out of an invalid region is the GATE's rhythm, not data."""
        sym = np.zeros(20_000, dtype=np.int16)
        valid = (np.arange(20_000) % 500) < 250
        self.assertEqual(transition_train(sym, valid).sum(), 0)
        est = estimate_symbol_clock(sym, valid, FS, baud_min=50.0, baud_max=10_000.0)
        self.assertEqual(est.rate_hz, 0.0)

    def test_invalid_params_raise(self):
        sym = np.zeros(1000, dtype=np.int16)
        with self.assertRaises(ValueError):
            estimate_symbol_clock(sym, None, 0.0, baud_min=50.0, baud_max=1000.0)
        with self.assertRaises(ValueError):
            estimate_symbol_clock(sym, None, FS, baud_min=1000.0, baud_max=50.0)
        with self.assertRaises(ValueError):
            estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=FS)
        with self.assertRaises(ValueError):
            transition_train(sym, np.zeros(999, dtype=bool))


# -- slicing -----------------------------------------------------------------

class SlicerTest(unittest.TestCase):

    def test_slicer_recovers_every_symbol(self):
        for baud in (200.0, 1000.0, 4800.0, 8000.0):
            with self.subTest(baud=baud):
                sym, truth = make_trace(baud, int(0.4 * baud) + 40, phase=13.7, seed=1)
                est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
                sl = SymbolSlicer(n_levels=2, guard=0.25)
                sl.set_clock(est.period_samples, est.phase_samples)
                labels, agree = sl.process(sym)
                self.assertGreater(labels.size, 0.9 * truth.size)
                self.assertEqual(best_accuracy(labels, truth), 1.0)
                self.assertEqual(agree.min(), 1.0)      # noiseless: every sample agrees

    def test_slicer_block_continuity(self):
        """Ragged reads, empty reads included, must decide the same symbols."""
        sym, _ = make_trace(1000.0, 400, phase=13.7, seed=4)
        est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)

        one = SymbolSlicer()
        one.set_clock(est.period_samples, est.phase_samples)
        l1, a1 = one.process(sym)

        rag = SymbolSlicer()
        rag.set_clock(est.period_samples, est.phase_samples)
        parts = split(rag, sym)
        self.assertTrue(np.array_equal(l1, np.concatenate([p[0] for p in parts])))
        self.assertTrue(np.allclose(a1, np.concatenate([p[1] for p in parts])))

    def test_agreement_is_the_eye(self):
        """A trace corrupted inside the symbols reads lower agreement, same bits."""
        sym, truth = make_trace(1000.0, 400, phase=13.7, seed=5)
        rng = np.random.default_rng(6)
        dirty = np.where(rng.random(sym.size) < 0.2, 1 - sym, sym).astype(np.int16)

        est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
        out = []
        for trace in (sym, dirty):
            sl = SymbolSlicer()
            sl.set_clock(est.period_samples, est.phase_samples)
            lab, ag = sl.process(trace)
            out.append((best_accuracy(lab, truth), float(ag.mean())))
        (clean_acc, clean_eye), (dirty_acc, dirty_eye) = out
        self.assertEqual(clean_eye, 1.0)
        self.assertTrue(0.6 < dirty_eye < 0.95)     # visibly narrower, still decodable
        self.assertEqual(dirty_acc, 1.0)            # 20% per-sample errors, 0 symbol errors

    def test_undecided_symbols_are_marked_not_guessed(self):
        sym, _ = make_trace(1000.0, 200, phase=0.0, seed=7)
        valid = np.zeros(sym.size, dtype=bool)
        valid[:sym.size // 2] = True
        sl = SymbolSlicer(min_samples=3)
        sl.set_clock(50.0, 0.0)
        lab, ag = sl.process(sym, valid)
        self.assertTrue((lab == -1).any())
        self.assertTrue(np.all(ag[lab == -1] == 0.0))
        self.assertEqual(sl.n_undecided, int((lab == -1).sum()))

    def test_clock_updates_keep_the_symbol_count(self):
        """A republished clock must slide the grid, not renumber it.

        The phase is published modulo T, so with the true phase near the wrap
        a +-0.6 sample jitter flips it between ~T and ~0; a 0.1% rate wobble
        moves every boundary far downstream by more than a period. Neither may
        duplicate or drop a symbol -- the live chain did exactly that before
        set_clock snapped by time (see SymbolSlicer.set_clock).
        """
        sym, truth = make_trace(1000.0, 600, phase=49.6, seed=12)     # T = 50
        est = estimate_symbol_clock(sym, None, FS, baud_min=50.0, baud_max=10_000.0)
        T = est.period_samples
        rng = np.random.default_rng(13)
        sl = SymbolSlicer()
        labels = []
        for i in range(0, sym.size, 512):
            # As the tracker publishes it: a slightly wrong period, and a
            # phase taken from the boundary nearest the CURRENT position and
            # reduced modulo that period -- so the grid is right here and
            # now, and only drifts with distance, as a real estimate does.
            period = T * (1.0 + 1e-3 * np.sin(i / 700.0))
            boundary_now = est.phase_samples + np.ceil((i - est.phase_samples) / T) * T
            phase = (boundary_now + rng.uniform(-0.6, 0.6)) % period
            sl.set_clock(period, phase)
            labels.append(sl.process(sym[i:i + 512])[0])
        got = np.concatenate(labels)
        self.assertLessEqual(abs(got.size - truth.size), 2)
        self.assertFalse((got < 0).any())
        # got[0] may be the stretch before the first true boundary, which
        # make_trace fills with bits[0]; everything after must be one run
        self.assertIn("".join(str(int(b)) for b in got[1:]),
                      "".join(str(int(b)) for b in truth))

    def test_live_grid_has_no_insertions(self):
        """Tracker + slicer as the live chain runs them: the clock republished
        every block, the slicer fed even while unlocked. The decided symbols
        must be a contiguous run of the sent ones -- no duplicate, no gap."""
        sym, truth = make_trace(1000.0, 2000, phase=49.3, seed=14)
        tr = SymbolClockTracker(FS, baud_min=50.0, baud_max=10_000.0,
                                history=25_000, interval=6_250)
        sl = SymbolSlicer()
        labels = []
        for i in range(0, sym.size, 410):
            blk = sym[i:i + 410]
            tr.process(blk)
            sl.set_clock(tr.period_samples if tr.locked else 0.0, tr.phase_abs)
            labels.append(sl.process(blk)[0])
        got = np.concatenate(labels)
        self.assertGreater(got.size, 1500)
        self.assertFalse((got < 0).any())
        self.assertIn("".join(str(int(b)) for b in got),
                      "".join(str(int(b)) for b in truth))

    def test_slicer_invalid_params(self):
        with self.assertRaises(ValueError):
            SymbolSlicer(n_levels=1)
        with self.assertRaises(ValueError):
            SymbolSlicer(guard=0.5)
        with self.assertRaises(ValueError):
            SymbolSlicer(min_samples=0)


# -- the live tracker --------------------------------------------------------

class TrackerTest(unittest.TestCase):

    def test_tracker_locks_and_then_releases(self):
        sym, _ = make_trace(1000.0, 600, phase=13.7, seed=8)
        tr = SymbolClockTracker(FS, baud_min=50.0, baud_max=10_000.0,
                                history=8192, interval=2048)
        for i in range(0, sym.size, 512):
            tr.process(sym[i:i + 512])
        self.assertTrue(tr.locked)
        self.assertAlmostEqual(tr.rate_hz, 1000.0, delta=10.0)
        self.assertGreater(tr.confidence_db, 15.0)

        # Now feed noise: the lock must not survive it (but may take `hold` tries).
        rng = np.random.default_rng(9)
        noise = (rng.random(60_000) < 0.5).astype(np.int16)
        for i in range(0, noise.size, 512):
            tr.process(noise[i:i + 512])
        self.assertTrue(not tr.locked or tr.confidence_db < 15.0)

    def test_tracker_phase_is_absolute(self):
        """The published boundary must stay on the same grid as the buffer slides."""
        sym, _ = make_trace(1000.0, 800, phase=13.7, seed=10)
        tr = SymbolClockTracker(FS, baud_min=50.0, baud_max=10_000.0,
                                history=8192, interval=2048)
        seen = []
        for i in range(0, sym.size, 512):
            if tr.process(sym[i:i + 512]) and tr.locked:
                seen.append(tr.phase_abs % tr.period_samples)
        self.assertGreaterEqual(len(seen), 3)
        T = FS / 1000.0
        ref = seen[0]
        for p in seen[1:]:
            err = ((p - ref) % T + T / 2) % T - T / 2
            self.assertLess(abs(err), 2.0, "grid moved by %s samples between estimates" % err)

    def test_tracker_invalid_params(self):
        with self.assertRaises(ValueError):
            SymbolClockTracker(FS, baud_min=50.0, baud_max=1000.0, history=32)
        with self.assertRaises(ValueError):
            SymbolClockTracker(FS, baud_min=50.0, baud_max=1000.0, interval=0)
        with self.assertRaises(ValueError):
            SymbolClockTracker(FS, baud_min=50.0, baud_max=1000.0, rate_ema=0.0)
        with self.assertRaises(ValueError):
            SymbolClockTracker(FS, baud_min=50.0, baud_max=1000.0, hold=0)


# -- bits ----------------------------------------------------------------------

class BitsTest(unittest.TestCase):

    def test_differential_bits_and_text(self):
        labels = np.array([0, 0, 1, 1, 0, 1, -1, 1, 0], dtype=np.int8)
        bits = differential_bits(labels)
        self.assertEqual(bits.tolist(), [0, 1, 0, 1, 1, -1, -1, 1])
        self.assertEqual(bits_to_text(bits, group=4), "0101 1..1")
        self.assertEqual(bits_to_text(bits, group=0), "01011..1")
        self.assertEqual(differential_bits(np.array([1], dtype=np.int8)).size, 0)

    def test_differential_decoding_is_polarity_blind(self):
        """Inverting every symbol must not change one bit -- that IS the point."""
        labels = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
        self.assertTrue(np.array_equal(differential_bits(labels),
                                       differential_bits(1 - labels)))


# -- DBPSK, end to end at baseband -------------------------------------------

class DbpskTest(unittest.TestCase):

    @staticmethod
    def decode(z, dm, block=None):
        """block=None feeds the whole array at once, which is the harsher
        case: the carrier is then estimated once for the entire call and its
        tracking loop never iterates, so a small error rotates the
        constellation across the run and the lock metric averages down.
        Pass a block size to exercise the path the live chain uses."""
        if block is None:
            sym, valid = dm.process(z)
        else:
            parts = [dm.process(z[i:i + block]) for i in range(0, z.size, block)]
            sym = np.concatenate([p[0] for p in parts])
            valid = np.concatenate([p[1] for p in parts])
        est = estimate_symbol_clock(sym, valid, FS, baud_min=100.0, baud_max=10_000.0)
        sl = SymbolSlicer()
        sl.set_clock(est.period_samples, est.phase_samples)
        labels, agree = sl.process(sym, valid)
        return est, labels, agree

    def test_dbpsk_recovers_the_bits(self):
        for fc, baud in ((5000.0, 1000.0), (2000.0, 500.0), (8000.0, 2000.0)):
            with self.subTest(fc=fc, baud=baud):
                z, truth = make_bpsk(fc, baud, 400, seed=0)
                dm = DbpskDemodulator(FS)
                est, labels, _ = self.decode(z, dm)
                self.assertAlmostEqual(dm.last_carrier_hz, fc, delta=0.01 * fc)
                self.assertGreater(dm.last_lock, 0.9)
                self.assertAlmostEqual(est.rate_hz, baud, delta=0.01 * baud)
                self.assertEqual(best_accuracy(differential_bits(labels), truth), 1.0)

    def test_dbpsk_survives_20db_snr(self):
        z, truth = make_bpsk(5000.0, 1000.0, 400, seed=2, snr_db=20.0)
        dm = DbpskDemodulator(FS)
        _, labels, _ = self.decode(z, dm, block=410)
        self.assertGreater(dm.last_lock, BandDecoder.PSK_LOCK_MIN[2])
        self.assertEqual(best_accuracy(differential_bits(labels), truth), 1.0)

    def test_no_confident_wrong_answer_at_low_snr(self):
        """The invariant behind the lock gate: at no SNR may the decoder
        show a trusted clock (confidence over the threshold), a well-formed
        constellation (lock over the gate) AND wrong bits at the same time.

        As first ported, a degraded carrier estimate at 6 dB turned the
        sign trace into a square wave at twice the residual -- a GENUINE
        periodicity, 33 dB of clock confidence, 0.94 agreement, 54% of the
        bits right -- and only the lock metric (0.01) caught it. The carrier
        estimation has since improved to the point where 6 dB decodes
        cleanly and lower SNRs simply produce no clock, but the invariant
        is what this test pins, whichever gate happens to do the work.
        """
        for snr in (20.0, 10.0, 6.0, 3.0, 0.0):
            with self.subTest(snr=snr):
                z, truth = make_bpsk(5000.0, 1000.0, 400, seed=2, snr_db=snr)
                dm = DbpskDemodulator(FS)
                est, labels, agree = self.decode(z, dm, block=410)
                acc = best_accuracy(differential_bits(labels), truth) if labels.size else 0.0
                trusted = est.confidence_db >= 15.0 and dm.last_lock >= BandDecoder.PSK_LOCK_MIN[2]
                if snr >= 20.0:
                    self.assertTrue(trusted)
                    self.assertEqual(acc, 1.0)
                if trusted:
                    self.assertGreater(acc, 0.95, "trusted but wrong at %s dB: lock %.2f conf %.1f acc %.2f"
                                       % (snr, dm.last_lock, est.confidence_db, acc))

    def test_dbpsk_block_continuity(self):
        """Ragged reads must give the same trace as one call."""
        z, _ = make_bpsk(5000.0, 1000.0, 300, seed=3)
        one = DbpskDemodulator(FS)
        s1, v1 = one.process(z)

        parts = split(DbpskDemodulator(FS), z)
        s2 = np.concatenate([p[0] for p in parts])
        v2 = np.concatenate([p[1] for p in parts])

        self.assertTrue(np.array_equal(v1, v2))
        # The carrier/angle estimates are per BLOCK, so a different chopping
        # moves them slightly; what must hold is that the DECISIONS agree. A
        # handful of samples next to a symbol edge may differ, and the
        # slicer's guard band exists for exactly that.
        self.assertGreater(float(np.mean(s1 == s2)), 0.99)

    def test_dbpsk_empty_and_invalid(self):
        dm = DbpskDemodulator(FS)
        s, v = dm.process(np.empty(0, dtype=np.complex128))
        self.assertEqual(s.size, 0)
        self.assertEqual(v.size, 0)
        with self.assertRaises(ValueError):
            DbpskDemodulator(FS, amp_gate=-1.0)
        with self.assertRaises(ValueError):
            DbpskDemodulator(FS, offset_ema=0.0)
        with self.assertRaises(ValueError):
            DbpskDemodulator(FS, angle_ema=1.5)

    def test_dbpsk_resolves_the_half_rate_ambiguity(self):
        """Above fs/4 the squared carrier wraps -- and is put back."""
        for fc in (5_000.0, 12_000.0, 14_500.0, 20_000.0):
            with self.subTest(fc=fc):
                z, truth = make_bpsk(fc, 1000.0, 300, seed=4)
                dm = DbpskDemodulator(FS)
                _, labels, _ = self.decode(z, dm)
                self.assertAlmostEqual(dm.last_carrier_hz, fc, delta=0.01 * fc)
                self.assertTrue(dm.carrier_in_range)
                self.assertEqual(best_accuracy(differential_bits(labels), truth), 1.0)

    def test_dbpsk_holds_polarity_against_a_drifting_carrier(self):
        """The angle loop must follow a slow turn of the constellation.

        A carrier that drifts 20 Hz over the run, fed in 410-sample blocks as
        the live chain does: the per-block measurement lags and is biased, so
        the de-rotated line turns. Differencing measurements (the original
        port) followed only a fifth of the turn and flipped the polarity every
        time the lag crossed the decision axis -- one wrong bit per flip.
        """
        baud, n_sym = 200.0, 500
        r = np.random.default_rng(15)
        truth = r.integers(0, 2, n_sym)
        sym = np.cumsum(truth) % 2
        T = FS / baud
        n = int(n_sym * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, n_sym - 1)
        t = np.arange(n) / FS
        fc = 5000.0 + 20.0 * t / t[-1]
        z = np.exp(1j * (2 * np.pi * np.cumsum(fc) / FS)) * (1 - 2 * sym[idx])

        dm = DbpskDemodulator(FS)
        parts = [dm.process(z[i:i + 410]) for i in range(0, n, 410)]
        got = np.concatenate([p[0] for p in parts])
        valid = np.concatenate([p[1] for p in parts])
        # polarity per 0.1 s window against the sent symbols: one constant
        # polarity throughout, whichever it is
        win = int(0.1 * FS)
        agreement = np.array([float(np.mean(got[k:k + win] == sym[idx][k:k + win]))
                              for k in range(win, n - win, win)])
        self.assertTrue(np.all(agreement > 0.95) or np.all(agreement < 0.05),
                        "polarity flipped: %s" % np.round(agreement, 2))
        self.assertAlmostEqual(dm.last_carrier_hz, 5020.0, delta=2.0)

        est = estimate_symbol_clock(got, valid, FS, baud_min=100.0, baud_max=10_000.0)
        sl = SymbolSlicer()
        sl.set_clock(est.period_samples, est.phase_samples)
        labels, _ = sl.process(got, valid)
        self.assertEqual(best_accuracy(differential_bits(labels), truth), 1.0)

    def test_dbpsk_noise_has_no_lock(self):
        rng = np.random.default_rng(11)
        noise = (rng.normal(size=20_000) + 1j * rng.normal(size=20_000)) / np.sqrt(2)
        dm = DbpskDemodulator(FS)
        dm.process(noise)
        self.assertLess(dm.last_lock, 0.1)


# -- FM / AM / FSK -----------------------------------------------------------

class DetectorTest(unittest.TestCase):

    def test_fm_constant_tone(self):
        freq, valid = FmDemodulator(FS).process(tone(5_000.0, 4000))
        self.assertEqual(freq.dtype, np.float32)
        self.assertEqual(freq[0], 0.0)                     # nothing elapsed yet
        self.assertTrue(np.allclose(freq[1:], 5_000.0, atol=1e-2))
        self.assertTrue(valid.all())

    def test_fm_block_continuity(self):
        z = tone(7_000.0, 10_000)
        one, _ = FmDemodulator(FS).process(z)
        parts = split(FmDemodulator(FS), z)
        streamed = np.concatenate([p[0] for p in parts])
        self.assertTrue(np.allclose(one, streamed, atol=1e-3))

    def test_fm_smoother_block_continuity(self):
        z = tone(7_000.0, 10_000) * (1.0 + 0.1 * np.cos(np.arange(10_000)))
        one, _ = FmDemodulator(FS, smooth_taps=9).process(z)
        parts = split(FmDemodulator(FS, smooth_taps=9), z)
        self.assertTrue(np.allclose(one, np.concatenate([p[0] for p in parts]), atol=1e-3))

    def test_fm_amp_gate(self):
        z = tone(5_000.0, 1000)
        z[400:600] *= 0.01
        _, valid = FmDemodulator(FS, amp_gate=0.5).process(z)
        self.assertFalse(valid[400:600].any())
        self.assertTrue(valid[:399].all())

    def test_am_envelope(self):
        z = tone(5_000.0, 1000) * np.linspace(0.0, 1.0, 1000)
        env = AmDemodulator().process(z)
        self.assertTrue(np.allclose(env, np.linspace(0.0, 1.0, 1000), atol=1e-6))

    def test_fsk_two_tone(self):
        """The middle 60% of every symbol must land on the right tone."""
        sym, n_sym = 50, 100
        z, _ = fsk_signal(5_000.0, 15_000.0, sym, n_sym)
        idx, valid = FskDemodulator(FS, [5_000.0, 15_000.0], 2_000.0).process(z)
        self.assertEqual(idx.dtype, np.int8)
        self.assertEqual(idx.size, z.size)
        want, got = [], []
        for k in range(n_sym):
            lo = k * sym + int(0.2 * sym)
            hi = k * sym + int(0.8 * sym)
            want.append(np.full(hi - lo, k % 2, dtype=np.int8))
            got.append(idx[lo:hi])
        self.assertGreater(float(np.mean(np.concatenate(got) == np.concatenate(want))), 0.99)

    def test_fsk_out_of_tolerance(self):
        """A 25 kHz tone against [5k, 15k]: "none of these" is the right answer."""
        idx, valid = FskDemodulator(FS, [5_000.0, 15_000.0], 2_000.0).process(tone(25_000.0, 5000))
        self.assertFalse(valid.any())
        self.assertTrue((idx == -1).all())

    def test_fsk_block_continuity(self):
        z, _ = fsk_signal(5_000.0, 15_000.0, 50, 100)
        dm = FskDemodulator(FS, [5_000.0, 15_000.0], 2_000.0)
        one_idx, one_valid = dm.process(z)
        dm.reset()
        parts = split(dm, z)
        self.assertTrue(np.array_equal(np.concatenate([p[0] for p in parts]), one_idx))
        self.assertTrue(np.array_equal(np.concatenate([p[1] for p in parts]), one_valid))

    def test_estimate_fsk_tones(self):
        z, _ = fsk_signal(5_000.0, 15_000.0, 50, 100)
        freq, valid = FmDemodulator(FS).process(z)
        tones = estimate_fsk_tones(freq, valid, n_tones=2)
        self.assertEqual(len(tones), 2)
        self.assertAlmostEqual(tones[0], 5_000.0, delta=300.0)
        self.assertAlmostEqual(tones[1], 15_000.0, delta=300.0)
        with self.assertRaises(ValueError):          # fewer valid samples than bins
            estimate_fsk_tones(freq[:100], valid[:100], n_tones=2)

    def test_fsk_invalid_params(self):
        with self.assertRaises(ValueError):
            FskDemodulator(FS, [5_000.0], 2_000.0)
        with self.assertRaises(ValueError):
            FskDemodulator(FS, [5_000.0, 15_000.0], 0.0)
        with self.assertRaises(ValueError):
            FskDemodulator(0.0, [5_000.0, 15_000.0], 2_000.0)
        dm = FskDemodulator(FS, [5_000.0, 15_000.0], 2_000.0)
        idx, valid = dm.process(np.zeros(0, dtype=np.complex128))
        self.assertEqual(idx.size, 0)
        self.assertEqual(valid.size, 0)


# -- the burst detector ------------------------------------------------------

class BurstTest(unittest.TestCase):

    @staticmethod
    def keyed_envelope(fs, baud, bits, seed=0):
        T = int(fs / baud)
        env = np.repeat(np.asarray(bits, dtype=np.float64), T)
        env = env + 1e-3 * (1.0 + 0.5 * np.random.default_rng(seed).random(env.size))
        return env, T

    @staticmethod
    def run_in_blocks(det, env, block):
        out = []
        for i in range(0, env.size, block):
            det.process(env[i:i + block])
            out.append(det.last_state.copy())
        return np.concatenate(out)

    def test_state_follows_the_keying(self):
        """Fed as the live path feeds it: in blocks, the first of which is
        background. (The floor starts from the first block's median, so one
        call with the whole 50%-duty stream would start it at the ON level.)"""
        fs = 6250.0
        bits = np.random.default_rng(1).integers(0, 2, 200)
        bits[:4] = 0                       # a quiet start, so the floor is the floor
        env, T = self.keyed_envelope(fs, 200.0, bits)
        det = BurstDetector(fs)
        state = self.run_in_blocks(det, env, 64)
        # judge each symbol by its middle, as the slicer would
        mid = np.arange(bits.size) * T + T // 2
        self.assertGreater(float(np.mean(state[mid] == bits.astype(bool))), 0.98)

    def test_block_continuity(self):
        """Splitting the stream must not split a burst. The floor's starting
        point is the first block's median, so both runs start inside the same
        quiet lead-in; after the floor has settled (tau 50 ms) the states
        must agree exactly."""
        fs = 6250.0
        bits = (np.random.default_rng(2).random(200) < 0.35).astype(np.int16)
        bits[:8] = 0                       # 250 samples of lead-in
        env, _ = self.keyed_envelope(fs, 200.0, bits)
        one = BurstDetector(fs)
        one.process(env)
        s1 = one.last_state.copy()
        rag = BurstDetector(fs)
        parts, i = [], 0
        for blk in (200, 0, 1, 3000, 500):
            rag.process(env[i:i + blk])
            parts.append(rag.last_state.copy())
            i += blk
        rag.process(env[i:])
        parts.append(rag.last_state.copy())
        s2 = np.concatenate(parts)
        settled = int(3 * 0.05 * fs)
        self.assertTrue(np.array_equal(s1[settled:], s2[settled:]))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            BurstDetector(0.0)
        with self.assertRaises(ValueError):
            BurstDetector(6250.0, ratio_on=2.0, ratio_off=3.0)


# -- the DDC -----------------------------------------------------------------

class DdcTest(unittest.TestCase):

    FS_IN = 250_000.0

    def test_tone_lands_at_its_baseband_offset(self):
        n = 50_000
        x = np.cos(2 * np.pi * 45_000.0 * np.arange(n) / self.FS_IN)
        ddc = ComplexDdc(40_000.0, 10_000.0, self.FS_IN, 5)
        z = ddc.process(x)
        self.assertEqual(ddc.fs_out, 50_000.0)
        self.assertEqual(z.size, n // 5)
        freq, _ = FmDemodulator(ddc.fs_out).process(z[200:])
        self.assertAlmostEqual(float(np.median(freq)), 5_000.0, delta=5.0)
        # gain 2 on a unit cosine -> unit amplitude analytic signal
        self.assertAlmostEqual(float(np.abs(z[200:]).mean()), 1.0, delta=0.02)

    def test_rejects_what_sits_below_the_band(self):
        """A symmetric lowpass would keep a tone just below f_lo (it lands at
        a negative baseband frequency); the one-sided filter must not. The
        121-tap skirt is about 3.4 kHz wide, so the check is 5 kHz below."""
        n = 50_000
        t = np.arange(n) / self.FS_IN
        inside = np.cos(2 * np.pi * 45_000.0 * t)
        below = np.cos(2 * np.pi * 35_000.0 * t)
        ddc = ComplexDdc(40_000.0, 10_000.0, self.FS_IN, 5)
        p_in = float(np.mean(np.abs(ddc.process(inside)[500:]) ** 2))
        ddc.reset()
        p_below = float(np.mean(np.abs(ddc.process(below)[500:]) ** 2))
        self.assertGreater(10 * np.log10(p_in / p_below), 40.0)

    def test_block_continuity(self):
        n = 30_000
        x = np.random.default_rng(3).normal(size=n)
        one = ComplexDdc(40_000.0, 10_000.0, self.FS_IN, 5).process(x)
        rag = ComplexDdc(40_000.0, 10_000.0, self.FS_IN, 5)
        parts, i = [], 0
        for blk in (2048, 0, 1, 4097, 777, 2048):
            parts.append(rag.process(x[i:i + blk]))
            i += blk
        parts.append(rag.process(x[i:]))
        self.assertTrue(np.allclose(one, np.concatenate(parts), atol=1e-6))

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            ComplexDdc(40_000.0, 10_000.0, 0.0, 5)
        with self.assertRaises(ValueError):
            ComplexDdc(40_000.0, 10_000.0, self.FS_IN, 0)
        with self.assertRaises(ValueError):
            ComplexDdc(40_000.0, 0.0, self.FS_IN, 5)
        with self.assertRaises(ValueError):
            ComplexDdc(130_000.0, 10_000.0, self.FS_IN, 5)


# -- the live chain, at the capture rate -------------------------------------

class BandDecoderTest(unittest.TestCase):
    """Synthetic keyed signals at 250 kHz, fed in 2048-sample blocks."""

    FS = 250_000.0
    BLOCK = 2048

    @classmethod
    def feed(cls, decoder, x):
        for i in range(0, x.size, cls.BLOCK):
            decoder.process(x[i:i + cls.BLOCK])

    @classmethod
    def fsk(cls, f0, f1, baud, bits, amp=0.5):
        T = cls.FS / baud
        n = int(len(bits) * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
        f_inst = np.where(np.asarray(bits)[idx] == 1, f1, f0)
        return amp * np.cos(2 * np.pi * np.cumsum(f_inst) / cls.FS)

    @classmethod
    def dbpsk(cls, fc, baud, bits, amp=0.5):
        sym = np.cumsum(bits) % 2
        T = cls.FS / baud
        n = int(len(bits) * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
        t = np.arange(n) / cls.FS
        return amp * np.cos(2 * np.pi * fc * t + np.pi * sym[idx])

    @classmethod
    def ook(cls, fc, baud, bits, amp=0.5, seed=0):
        T = cls.FS / baud
        n = int(len(bits) * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
        t = np.arange(n) / cls.FS
        carrier = amp * np.cos(2 * np.pi * fc * t) * np.asarray(bits)[idx]
        # a little noise, so the detector's floor is a level and not zero
        return carrier + 1e-4 * np.random.default_rng(seed).normal(size=n)

    @staticmethod
    def shown(decoder):
        return decoder.last_bits.replace(" ", "")

    @classmethod
    def through_a_room(cls, x):
        """A few strong reflections, as a room at 12-16 kHz measured: the
        direct path, then echoes 10 dB down at 2 ms, 14 dB at 5 ms, 16 dB at
        11 ms and 20 dB at 23 ms (the envelope after a tone stops fell 10 dB
        in 3 ms and then sat 15-20 dB down for 40 ms)."""
        echoes = ((0.002, 0.32), (0.005, 0.20), (0.011, 0.16), (0.023, 0.10))
        y = x.copy()
        for delay_s, gain in echoes:
            d = int(delay_s * cls.FS)
            y[d:] += gain * x[:-d]
        return y

    def test_fsk_through_a_room(self):
        """Reverberation beats with the new tone for the first milliseconds
        of every symbol. Without the beat-period smoothing this locked at the
        right baud and showed mostly undecided symbols."""
        rng = np.random.default_rng(30)
        bits = rng.integers(0, 2, 250)                     # 2.5 s at 100 Bd
        x = self.through_a_room(self.fsk(12_000.0, 16_000.0, 100.0, bits))
        dec = BandDecoder(self.FS, mode=FSK)
        dec.configure(10_000.0, 8_000.0)
        self.feed(dec, x)
        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 100.0, delta=2.0)
        self.assertGreater(dec.last_eye, 0.9)
        self.assertGreater(dec.last_fsk_assigned, 0.8)
        shown = self.shown(dec)
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, "".join(str(b) for b in bits))

    def test_ook_through_a_room(self):
        """During an OFF symbol the band holds the ON symbols' reverberation,
        15-20 dB down; a floor that tracks it makes 6x the floor sit above
        the carrier. The slicer must key on the carrier's own midpoint."""
        rng = np.random.default_rng(31)
        bits = rng.integers(0, 2, 350)                     # 7 s at 50 Bd
        bits[:6] = 0
        x = self.through_a_room(self.ook(14_000.0, 50.0, bits, amp=0.5))
        dec = BandDecoder(self.FS, mode=OOK)
        dec.configure(10_000.0, 8_000.0)
        self.feed(dec, x)
        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 50.0, delta=1.0)
        shown = self.shown(dec)
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, "".join(str(b) for b in bits))

    def test_fsk_end_to_end(self):
        rng = np.random.default_rng(20)
        bits = rng.integers(0, 2, 600)                     # 2 s at 300 Bd
        x = self.fsk(43_000.0, 47_000.0, 300.0, bits)
        dec = BandDecoder(self.FS, mode=FSK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)

        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 300.0, delta=6.0)
        self.assertGreater(dec.last_baud_conf_db, 15.0)
        self.assertGreater(dec.last_eye, 0.9)
        self.assertAlmostEqual(dec.last_fsk_tones[0], 43_000.0, delta=400.0)
        self.assertAlmostEqual(dec.last_fsk_tones[1], 47_000.0, delta=400.0)
        shown = self.shown(dec)
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, "".join(str(b) for b in bits))
        self.assertEqual(len(dec.last_symbols), len(dec.last_agreements))
        self.assertIn("Bd", dec.decode_text())

    def test_dbpsk_end_to_end(self):
        rng = np.random.default_rng(21)
        bits = rng.integers(0, 2, 2000)                    # 2 s at 1000 Bd
        x = self.dbpsk(45_000.0, 1000.0, bits)
        dec = BandDecoder(self.FS, mode=DBPSK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)

        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 1000.0, delta=20.0)
        self.assertAlmostEqual(dec.last_carrier_hz, 45_000.0, delta=100.0)
        self.assertGreater(dec.last_lock, 0.9)
        shown = self.shown(dec)
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, "".join(str(b) for b in bits))

    def test_dbpsk_slow_end_to_end(self):
        """200 Bd for 2.5 s: long enough for a lagging angle tracker to flip
        the polarity several times (it did -- see DbpskDemodulator)."""
        rng = np.random.default_rng(108)
        bits = rng.integers(0, 2, 500)
        x = self.dbpsk(45_000.0, 200.0, bits)
        dec = BandDecoder(self.FS, mode=DBPSK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)
        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 200.0, delta=4.0)
        shown = self.shown(dec)
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, "".join(str(b) for b in bits))

    def test_ook_end_to_end(self):
        rng = np.random.default_rng(22)
        bits = rng.integers(0, 2, 600)                     # 3 s at 200 Bd
        bits[:8] = 0
        x = self.ook(45_000.0, 200.0, bits)
        dec = BandDecoder(self.FS, mode=OOK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)

        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 200.0, delta=4.0)
        shown = self.shown(dec)
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, "".join(str(b) for b in bits))

    def test_noise_never_shows_bits(self):
        """The live-chain version of noise_reports_no_clock, for every mode."""
        x = 0.01 * np.random.default_rng(23).normal(size=int(2.0 * self.FS))
        for mode in MODES:
            with self.subTest(mode=mode):
                dec = BandDecoder(self.FS, mode=mode)
                dec.configure(40_000.0, 10_000.0)
                ever_locked = False
                for i in range(0, x.size, self.BLOCK):
                    dec.process(x[i:i + self.BLOCK])
                    ever_locked = ever_locked or dec.decode_locked
                self.assertFalse(ever_locked, "%s locked on noise: %s" % (mode, dec.decode_text()))
                self.assertEqual(dec.last_bits, "")
                self.assertEqual(dec.last_symbols, ())

    def test_a_chopped_tone_is_not_data(self):
        """One tone switched on and off at a steady rate is a device, not a
        link, and must not come back as bits.

        Found in this room: the 25 kHz pest repeller pulses at exactly
        1 kHz, so the frequency trace of its band flips on a 1 kHz grid and
        4-FSK locked at 1000 Bd with an eye of 0.95 -- while 75-84% of the
        symbols were the same one and a whole level was never used. Real
        four-level data spreads over its four symbols.
        """
        t = np.arange(int(3.0 * self.FS)) / self.FS
        gate = (np.sin(2 * np.pi * 1000.0 * t) > 0).astype(np.float64)
        x = 0.3 * np.cos(2 * np.pi * 45_000.0 * t) * (0.5 + 0.5 * gate)
        x += 0.002 * np.random.default_rng(50).normal(size=t.size)
        for mode in (FSK4, DQPSK, FSK, DBPSK):
            with self.subTest(mode=mode):
                dec = BandDecoder(self.FS, mode=mode)
                dec.configure(44_500.0, 1_000.0)
                ever = False
                for i in range(0, x.size, self.BLOCK):
                    dec.process(x[i:i + self.BLOCK])
                    ever = ever or dec.decode_locked
                self.assertFalse(ever, "%s decoded a chopped tone: %s" % (mode, dec.decode_text()))
                self.assertEqual(dec.last_bits, "")

    def test_tones_sounding_together_are_not_keying(self):
        """Several steady tones at once is a hum, not a link.

        Found in this room: the 100-1000 Hz band holds a dozen steady lines
        (a machine's harmonics), and the instantaneous frequency of their
        sum swings between them at the beat rate -- a perfectly regular
        grid, which the clock recovery locked onto at 21-25 dB. 4-FSK
        published bits for 690 blocks. Keying sends ONE tone at a time and
        so has a steady envelope; tones sounding together beat, and the
        envelope dips as they cancel.
        """
        t = np.arange(int(3.0 * self.FS)) / self.FS
        x = sum(0.2 * np.cos(2 * np.pi * f * t)
                for f in (44_800.0, 45_000.0, 45_260.0, 45_500.0))
        x += 0.001 * np.random.default_rng(52).normal(size=t.size)
        for mode in (FSK, FSK4):
            with self.subTest(mode=mode):
                dec = BandDecoder(self.FS, mode=mode)
                dec.configure(44_500.0, 1_200.0)
                ever = False
                for i in range(0, x.size, self.BLOCK):
                    dec.process(x[i:i + self.BLOCK])
                    ever = ever or dec.decode_locked
                self.assertFalse(ever, "%s decoded a hum: %s" % (mode, dec.decode_text()))
                self.assertEqual(dec.last_bits, "")
                self.assertGreater(dec.last_env_cv, BandDecoder.KEYED_MAX_ENVELOPE_CV)

    def test_real_fsk_has_a_steady_envelope(self):
        """The other side of the same gate: keying must stay under it."""
        rng = np.random.default_rng(53)
        bits = rng.integers(0, 2, 900)
        x = self.through_a_room(self.fsk(43_000.0, 47_000.0, 300.0, bits))
        dec = BandDecoder(self.FS, mode=FSK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)
        self.assertLess(dec.last_env_cv, BandDecoder.KEYED_MAX_ENVELOPE_CV)
        self.assertTrue(dec.decode_locked, dec.decode_text())

    def test_band_limited_noise_shows_nothing(self):
        """The room's own noise in a narrow band, in every keyed mode.

        White noise (test_noise_never_shows_bits) is the easy case: its
        transitions are dense and even. A NARROW band of it is what a real
        empty band looks like -- slow, structured wandering -- and it is
        what produced this decoder's only real false positives.
        """
        rng = np.random.default_rng(51)
        n = int(4.0 * self.FS)
        white = rng.normal(size=n)
        # a 1 kHz slice of it around 45 kHz, at a realistic level
        k = np.arange(-2000, 2001)
        h = np.sinc(2 * 500.0 * k / self.FS) * np.hamming(k.size)
        h = h / np.sum(h)
        base = np.convolve(white, h, mode="same")
        t = np.arange(n) / self.FS
        x = 0.02 * base * np.cos(2 * np.pi * 45_000.0 * t)
        for mode in DIGITAL_MODES:
            with self.subTest(mode=mode):
                dec = BandDecoder(self.FS, mode=mode)
                dec.configure(44_500.0, 1_000.0)
                ever = False
                for i in range(0, x.size, self.BLOCK):
                    dec.process(x[i:i + self.BLOCK])
                    ever = ever or dec.decode_locked
                self.assertFalse(ever, "%s decoded noise: %s" % (mode, dec.decode_text()))
                self.assertEqual(dec.last_bits, "")

    def test_band_change_starts_over(self):
        bits = np.random.default_rng(24).integers(0, 2, 600)
        x = self.fsk(43_000.0, 47_000.0, 300.0, bits)
        dec = BandDecoder(self.FS, mode=FSK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)
        self.assertTrue(dec.decode_locked)
        dec.configure(60_000.0, 10_000.0)
        self.assertFalse(dec.decode_locked)
        self.assertEqual(dec.last_bits, "")
        self.assertEqual(dec.last_baud_hz, 0.0)
        self.assertEqual(dec.band, (60_000.0, 10_000.0))
        # the same band again is not a change
        dec.configure(60_000.0, 10_000.0)
        dec.process(x[:self.BLOCK])
        self.assertEqual(dec.n_blocks, 1)

    def test_mode_switch_rebuilds(self):
        dec = BandDecoder(self.FS, mode=FSK)
        dec.configure(40_000.0, 10_000.0)
        dec.process(np.zeros(self.BLOCK))
        dec.set_mode(OOK)
        self.assertEqual(dec.mode, OOK)
        self.assertEqual(dec.fs_trace, 50_000.0 / 8)
        self.assertLess(dec.baud_range[1], 800.0)
        dec.set_mode(DBPSK)
        self.assertEqual(dec.baud_range, (BandDecoder.SLOW_BAUD_MIN, 5000.0))
        with self.assertRaises(ValueError):
            dec.set_mode("qpsk")

    def test_wide_band_is_capped_at_the_baseband(self):
        dec = BandDecoder(self.FS, mode=FSK)
        dec.configure(40_000.0, 40_000.0)
        self.assertEqual(dec.decoded_width_hz, 25_000.0)
        dec.configure(40_000.0, 10_000.0)
        self.assertEqual(dec.decoded_width_hz, 10_000.0)

    def test_decode_can_be_switched_off(self):
        bits = np.random.default_rng(25).integers(0, 2, 300)
        x = self.fsk(43_000.0, 47_000.0, 300.0, bits)
        dec = BandDecoder(self.FS, mode=FSK)
        dec.enabled = False
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)
        self.assertFalse(dec.decode_locked)
        self.assertEqual(dec.last_bits, "")
        self.assertEqual(dec.decode_text(), "decode off")
        # the signal readout still works without the decoder
        self.assertIn("FSK tones", dec.signal_text())

    def test_readout_before_a_band(self):
        dec = BandDecoder(self.FS)
        self.assertEqual(dec.signal_text(), "no band selected")
        self.assertEqual(dec.decode_text(), "")
        dec.process(np.zeros(self.BLOCK))          # harmless
        self.assertEqual(dec.n_blocks, 0)

    # -- the four-level keyed modes ------------------------------------------

    @classmethod
    def fsk4(cls, tones, baud, dibits, amp=0.5):
        T = cls.FS / baud
        n = int(len(dibits) * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, len(dibits) - 1)
        f_inst = np.asarray(tones, dtype=np.float64)[np.asarray(dibits)[idx]]
        return amp * np.cos(2 * np.pi * np.cumsum(f_inst) / cls.FS)

    @classmethod
    def dqpsk(cls, fc, baud, dibits, amp=0.5):
        sym = np.cumsum(dibits) % 4
        T = cls.FS / baud
        n = int(len(dibits) * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, len(dibits) - 1)
        t = np.arange(n) / cls.FS
        return amp * np.cos(2 * np.pi * fc * t + (np.pi / 2) * sym[idx])

    def test_fsk4_end_to_end(self):
        rng = np.random.default_rng(40)
        dibits = rng.integers(0, 4, 900)                   # 3 s at 300 Bd
        tones = (42_000.0, 44_000.0, 46_000.0, 48_000.0)
        x = self.fsk4(tones, 300.0, dibits)
        dec = BandDecoder(self.FS, mode=FSK4)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)
        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 300.0, delta=6.0)
        self.assertEqual(len(dec.last_fsk_tones), 4)
        for got, sent in zip(dec.last_fsk_tones, tones):
            self.assertAlmostEqual(got, sent, delta=400.0)
        self.assertEqual(dec.levels, 4)
        self.assertEqual(dec.bits_per_symbol, 2)
        shown = self.shown(dec)
        truth = "".join(str(b) for b in symbols_to_bits(dibits, 2))
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, truth)
        self.assertTrue(all(s in (0, 1, 2, 3) for s in dec.last_symbols))

    def test_dqpsk_end_to_end(self):
        rng = np.random.default_rng(41)
        dibits = rng.integers(0, 4, 1500)                  # 3 s at 500 Bd
        x = self.dqpsk(45_000.0, 500.0, dibits)
        dec = BandDecoder(self.FS, mode=DQPSK)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, x)
        self.assertTrue(dec.decode_locked, dec.decode_text())
        self.assertAlmostEqual(dec.last_baud_hz, 500.0, delta=10.0)
        self.assertAlmostEqual(dec.last_carrier_hz, 45_000.0, delta=100.0)
        self.assertGreater(dec.last_lock, 0.9)
        shown = self.shown(dec)
        truth = "".join(str(b) for b in symbols_to_bits(dibits, 2))
        self.assertGreaterEqual(len(shown), 32)
        self.assertIn(shown, truth)

    # -- the analog modes ---------------------------------------------------

    @classmethod
    def fm(cls, fc, deviation_hz, rate_hz, seconds, amp=0.5):
        t = np.arange(int(seconds * cls.FS)) / cls.FS
        return amp * np.cos(2 * np.pi * fc * t + (deviation_hz / rate_hz) * np.sin(2 * np.pi * rate_hz * t))

    @classmethod
    def am(cls, fc, depth, rate_hz, seconds, amp=0.3):
        t = np.arange(int(seconds * cls.FS)) / cls.FS
        return amp * (1.0 + depth * np.cos(2 * np.pi * rate_hz * t)) * np.cos(2 * np.pi * fc * t)

    @classmethod
    def pm(cls, fc, deviation_rad, rate_hz, seconds, amp=0.5):
        t = np.arange(int(seconds * cls.FS)) / cls.FS
        return amp * np.cos(2 * np.pi * fc * t + deviation_rad * np.sin(2 * np.pi * rate_hz * t))

    def test_fm_readout(self):
        """A 40 Hz modulation: slow enough for the 400-point, one-second
        display to draw it (a 200 Hz one sits at that display's Nyquist and
        only the numbers can report it -- see test_fm_readout_fast)."""
        dec = BandDecoder(self.FS, mode=FM)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, self.fm(45_000.0, 1_500.0, 40.0, 2.0))
        self.assertTrue(dec.analog_carrier, dec.signal_text())
        self.assertAlmostEqual(dec.last_carrier_hz, 45_000.0, delta=30.0)
        self.assertAlmostEqual(dec.last_deviation, 1_500.0, delta=225.0)
        self.assertAlmostEqual(dec.last_mod_rate_hz, 40.0, delta=3.0)
        self.assertIn("FM carrier", dec.signal_text())
        self.assertIn("modulated at", dec.signal_text())
        self.assertIn("analog", dec.decode_text())
        self.assertFalse(dec.decode_locked)
        self.assertEqual(dec.last_bits, "")
        self.assertEqual(len(dec.last_trace), BandDecoder.ANALOG_TRACE_POINTS)
        # the displayed waveform spans the deviation (its cells average a
        # few ms, so the very peaks are shaved)
        self.assertGreater(dec.last_trace_hi - dec.last_trace_lo, 2_000.0)
        self.assertEqual(dec.baud_range, (0.0, 0.0))
        self.assertTrue(dec.is_analog)

    def test_fm_readout_fast(self):
        """Modulated faster than the display can draw: the numbers still say."""
        dec = BandDecoder(self.FS, mode=FM)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, self.fm(45_000.0, 1_500.0, 200.0, 2.0))
        self.assertTrue(dec.analog_carrier, dec.signal_text())
        self.assertAlmostEqual(dec.last_deviation, 1_500.0, delta=225.0)
        self.assertAlmostEqual(dec.last_mod_rate_hz, 200.0, delta=10.0)

    def test_am_readout(self):
        dec = BandDecoder(self.FS, mode=AM)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, self.am(45_000.0, 0.5, 150.0, 2.0))
        self.assertTrue(dec.analog_carrier, dec.signal_text())
        self.assertAlmostEqual(dec.last_carrier_hz, 45_000.0, delta=30.0)
        self.assertAlmostEqual(dec.last_deviation, 0.5, delta=0.08)
        self.assertAlmostEqual(dec.last_mod_rate_hz, 150.0, delta=8.0)
        self.assertIn("AM carrier", dec.signal_text())
        self.assertIn("depth", dec.signal_text())

    def test_pm_readout(self):
        dec = BandDecoder(self.FS, mode=PM)
        dec.configure(40_000.0, 10_000.0)
        self.feed(dec, self.pm(45_000.0, 1.0, 100.0, 2.0))
        self.assertTrue(dec.analog_carrier, dec.signal_text())
        self.assertAlmostEqual(dec.last_carrier_hz, 45_000.0, delta=30.0)
        self.assertAlmostEqual(dec.last_deviation, 1.0, delta=0.25)
        self.assertAlmostEqual(dec.last_mod_rate_hz, 100.0, delta=5.0)
        self.assertIn("PM carrier", dec.signal_text())

    def test_analog_noise_is_not_a_carrier(self):
        x = 0.01 * np.random.default_rng(42).normal(size=int(2.0 * self.FS))
        for mode in ANALOG_MODES:
            with self.subTest(mode=mode):
                dec = BandDecoder(self.FS, mode=mode)
                dec.configure(40_000.0, 10_000.0)
                self.feed(dec, x)
                self.assertFalse(dec.analog_carrier, dec.signal_text())
                self.assertIn("no steady carrier", dec.signal_text())
                self.assertEqual(dec.last_mod_rate_hz, 0.0)

    def test_mode_lists(self):
        self.assertEqual(MODES, DIGITAL_MODES + ANALOG_MODES)
        self.assertEqual(len(MODES), 8)
        for mode in DIGITAL_MODES:
            dec = BandDecoder(self.FS, mode=mode)
            self.assertIn(dec.levels, (2, 4))
            self.assertEqual(dec.bits_per_symbol, 1 if dec.levels == 2 else 2)
        for mode in ANALOG_MODES:
            self.assertEqual(BandDecoder(self.FS, mode=mode).levels, 0)


# -- PM ----------------------------------------------------------------------

class PmTest(unittest.TestCase):

    N = 10_000

    @staticmethod
    def step_tone(f0, step_rad, n, fs=FS):
        t = np.arange(n) / fs
        extra = np.where(np.arange(n) >= n // 2, step_rad, 0.0)
        return np.exp(1j * (2 * np.pi * f0 * t + extra))

    def test_constant_tone_is_a_straight_ramp(self):
        phi = PmDemodulator(FS).process(tone(5_000.0, self.N))
        self.assertEqual(phi.dtype, np.float32)
        self.assertEqual(phi[0], 0.0)                      # nothing has elapsed yet
        n = np.arange(self.N, dtype=np.float64)
        slope, intercept = np.polyfit(n, phi.astype(np.float64), 1)
        self.assertAlmostEqual(slope, 2 * np.pi * 5_000.0 / FS, delta=1e-3 * 2 * np.pi * 5_000.0 / FS)
        resid = phi.astype(np.float64) - (slope * n + intercept)
        self.assertLess(np.sqrt(np.mean(resid ** 2)), 1e-3)
        with self.assertRaises(ValueError):
            PmDemodulator(0.0)

    def test_offset_removal(self):
        phi = PmDemodulator(FS, f_offset_hz=5_000.0).process(tone(5_000.0, self.N))
        self.assertLess(float(np.abs(phi).max()), 1e-3)

    def test_quarter_phase_step(self):
        phi = PmDemodulator(FS, f_offset_hz=5_000.0).process(self.step_tone(5_000.0, np.pi / 2, self.N))
        self.assertAlmostEqual(float(phi[:self.N // 2].mean()), 0.0, delta=1e-3)
        self.assertAlmostEqual(float(phi[self.N // 2:].mean()), np.pi / 2, delta=1e-3)

    def test_block_continuity(self):
        z = self.step_tone(5_000.0, np.pi / 2, self.N)
        one = PmDemodulator(FS, f_offset_hz=5_000.0).process(z)
        streamed = np.concatenate(split(PmDemodulator(FS, f_offset_hz=5_000.0), z))
        self.assertTrue(np.allclose(one, streamed, atol=1e-3))

    def test_offset_can_change_mid_stream(self):
        dm = PmDemodulator(FS)
        z = tone(5_000.0, self.N)
        first = dm.process(z[:self.N // 2])
        dm.set_offset(5_000.0)
        second = dm.process(z[self.N // 2:])
        # a ramp before, then flat: the phase already accumulated stays
        self.assertGreater(float(first[-1]), 100.0)
        self.assertLess(float(np.ptp(second)), 1e-2)
        self.assertAlmostEqual(float(second[0]), float(first[-1]), delta=1e-2)


# -- DQPSK -------------------------------------------------------------------

def make_dqpsk(fc, baud, n_sym, *, fs=FS, phase0=0.3, seed=0, snr_db=None):
    """Differentially-encoded QPSK on a complex carrier; returns (z, the sent dibits)."""
    r = np.random.default_rng(seed)
    dibits = r.integers(0, 4, n_sym)
    sym = np.cumsum(dibits) % 4
    T = fs / baud
    n = int(n_sym * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, n_sym - 1)
    z = np.exp(1j * (2 * np.pi * fc * np.arange(n) / fs + phase0 + (np.pi / 2) * sym[idx]))
    if snr_db is not None:
        amp = 10.0 ** (-snr_db / 20.0)
        z = z + amp * (r.normal(size=n) + 1j * r.normal(size=n)) / np.sqrt(2)
    return z, dibits


class DqpskTest(unittest.TestCase):

    @staticmethod
    def decode(z, dm):
        sym, valid = dm.process(z)
        est = estimate_symbol_clock(sym, valid, FS, baud_min=100.0, baud_max=10_000.0)
        sl = SymbolSlicer(n_levels=4)
        sl.set_clock(est.period_samples, est.phase_samples)
        labels, agree = sl.process(sym, valid)
        return est, differential_bits(labels, n_levels=4), agree

    def test_recovers_the_dibits(self):
        for fc, baud in ((5_000.0, 1_000.0), (2_000.0, 500.0), (12_000.0, 2_000.0)):
            with self.subTest(fc=fc, baud=baud):
                z, truth = make_dqpsk(fc, baud, 400, seed=0)
                dm = DqpskDemodulator(FS)
                est, steps, _ = self.decode(z, dm)
                self.assertEqual(dm.order, 4)
                self.assertAlmostEqual(dm.last_carrier_hz, fc, delta=0.01 * fc)
                self.assertGreater(dm.last_lock, 0.9)
                self.assertAlmostEqual(est.rate_hz, baud, delta=0.01 * baud)
                self.assertEqual(best_accuracy(steps, truth), 1.0)

    def test_survives_20db_snr(self):
        """Fed as the live chain feeds it, in 410-sample blocks, so the
        carrier loop iterates; judged on the last 60% once it has settled."""
        z, truth = make_dqpsk(5_000.0, 1_000.0, 1000, seed=2, snr_db=20.0)
        dm = DqpskDemodulator(FS)
        parts = [dm.process(z[i:i + 410]) for i in range(0, z.size, 410)]
        sym = np.concatenate([p[0] for p in parts])
        valid = np.concatenate([p[1] for p in parts])
        k = int(0.4 * z.size)
        est = estimate_symbol_clock(sym[k:], valid[k:], FS, baud_min=100.0, baud_max=10_000.0)
        sl = SymbolSlicer(n_levels=4)
        sl.set_clock(est.period_samples, est.phase_samples)
        labels, _ = sl.process(sym[k:], valid[k:])
        self.assertGreater(dm.last_lock, 0.3)
        self.assertAlmostEqual(dm.last_carrier_hz, 5_000.0, delta=5.0)
        self.assertEqual(best_accuracy(differential_bits(labels, n_levels=4), truth[400:]), 1.0)

    def test_block_continuity(self):
        z, _ = make_dqpsk(5_000.0, 1_000.0, 300, seed=3)
        s1, v1 = DqpskDemodulator(FS).process(z)
        parts = split(DqpskDemodulator(FS), z)
        s2 = np.concatenate([p[0] for p in parts])
        v2 = np.concatenate([p[1] for p in parts])
        self.assertTrue(np.array_equal(v1, v2))
        self.assertGreater(float(np.mean(s1 == s2)), 0.99)

    def test_noise_has_no_lock(self):
        rng = np.random.default_rng(11)
        noise = (rng.normal(size=20_000) + 1j * rng.normal(size=20_000)) / np.sqrt(2)
        dm = DqpskDemodulator(FS)
        dm.process(noise)
        self.assertLess(dm.last_lock, 0.1)

    def test_orders(self):
        self.assertEqual(DbpskDemodulator(FS).order, 2)
        self.assertEqual(DpskDemodulator(FS, order=4).order, 4)
        with self.assertRaises(ValueError):
            DpskDemodulator(FS, order=3)

    def test_symbols_to_bits(self):
        self.assertEqual(symbols_to_bits(np.array([0, 1, 2, 3, -1]), 2).tolist(),
                         [0, 0, 0, 1, 1, 0, 1, 1, -1, -1])
        self.assertEqual(symbols_to_bits(np.array([0, 1, -1]), 1).tolist(), [0, 1, -1])
        self.assertEqual(bits_to_text(symbols_to_bits(np.array([3, 0]), 2), group=4), "1100")


if __name__ == "__main__":
    unittest.main()
