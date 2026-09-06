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

"""The live demodulation chain: capture blocks in, a readout out.

This is the decode half of ultraScan's DemodAudifier (dsp/demod_audio.py,
M22-M26) without the audio: the band is not turned into sound here, only
into numbers, waveforms and bits. Friture's own listen path already does
the hearing.

    capture (250 kHz)
      -> ComplexDdc            band [f_lo, f_lo+width] as complex baseband, 50 kHz
      -> one of
         keyed (digital) modes, a per-sample symbol trace:
           OOK    AmDemodulator -> decimate x8 -> midpoint slicer           (6.25 kHz trace)
           FSK    FmDemodulator -> estimate_fsk_tones -> FskDemodulator      (50 kHz trace)
           4-FSK  the same with four tones, two bits a symbol
           DBPSK  DpskDemodulator, M = 2                                    (50 kHz trace)
           DQPSK  DpskDemodulator, M = 4, two bits a symbol
         analog modes, a per-sample waveform:
           FM     FmDemodulator   instantaneous frequency
           AM     AmDemodulator   envelope
           PM     PmDemodulator   phase, the carrier's own ramp removed
      -> keyed: SymbolClockTracker (the baud, and how much to believe it),
                SymbolSlicer (one decision per symbol, with its agreement),
                bits -- levels for OOK/FSK, changes for DPSK
      -> analog: a window of the waveform, its carrier, deviation and the
                 rate it is modulated at

Everything the dock shows is a plain attribute written here and read on the
GUI thread (``last_*``, ``decode_locked``). Bits are published only behind
two gates: the clock's confidence, and -- in DPSK, where a bad carrier
estimate produces a genuinely periodic trace that passes the first gate --
the constellation lock. Both numbers are published either way, which is the
difference between a decoder and a bit generator.

A change of band or mode is a rebuild, not a retune: the clock tracker and
the slicer share an absolute sample numbering that only means anything
within one stream, and a different band is a different stream.
"""

from __future__ import annotations

import numpy as np

from friture.demod.ddc import ComplexDdc
from friture.demod.detectors import (
    AmDemodulator,
    DpskDemodulator,
    FmDemodulator,
    FskDemodulator,
    PmDemodulator,
    estimate_fsk_tones,
)
from friture.demod.symbols import (
    SymbolClockTracker,
    SymbolSlicer,
    bits_to_text,
    differential_bits,
    symbols_to_bits,
)

OOK = "ook"
FSK = "fsk"
FSK4 = "fsk4"
DBPSK = "dbpsk"
DQPSK = "dqpsk"
FM = "fm"
AM = "am"
PM = "pm"
DIGITAL_MODES = (OOK, FSK, FSK4, DBPSK, DQPSK)
ANALOG_MODES = (FM, AM, PM)
MODES = DIGITAL_MODES + ANALOG_MODES
MODE_LABELS = {
    OOK: "OOK",
    FSK: "FSK",
    FSK4: "4-FSK",
    DBPSK: "DBPSK",
    DQPSK: "DQPSK",
    FM: "FM",
    AM: "AM",
    PM: "PM",
}
MODE_DESCRIPTIONS = {
    OOK: "On/off keying: a carrier switched on and off (rangefinders, remotes, beacons)",
    FSK: "Frequency-shift keying: two tones, one per symbol (found from the signal itself)",
    FSK4: "Four-tone frequency-shift keying: two bits per symbol (tones found from the signal)",
    DBPSK: "Differential phase-reversal keying: the carrier flips 180 degrees per change",
    DQPSK: "Differential four-phase keying: quarter-turn steps, two bits per symbol",
    FM: "Frequency modulation: the instantaneous frequency of the band's carrier, as a waveform",
    AM: "Amplitude modulation: the envelope of the band's carrier, as a waveform",
    PM: "Phase modulation: the carrier's phase with its own rotation removed, as a waveform",
}
# symbol levels and bits per symbol of the keyed modes
_LEVELS = {OOK: 2, FSK: 2, FSK4: 4, DBPSK: 2, DQPSK: 4}
_BITS_PER_SYMBOL = {OOK: 1, FSK: 1, FSK4: 2, DBPSK: 1, DQPSK: 2}
_TONE_COUNT = {FSK: 2, FSK4: 4}
_DIFFERENTIAL = (DBPSK, DQPSK)


class BandDecoder:
    """Capture blocks -> symbol trace -> clock -> bits, or -> waveform, for one band and mode."""

    # 250 kHz -> 50 kHz baseband. The same rate ultraScan's audio path uses,
    # so its measured baud windows carry over.
    DECIM = 5
    # The envelope is decimated before it is sliced. The slicer is a
    # per-sample Python loop; 8x down is still 6.25 kHz, i.e. 0.16 ms of
    # timing resolution.
    OOK_DECIM = 8
    # OOK edges are sliced at the MIDPOINT of the carrier's own level. The
    # burst detector's floor-relative thresholds (6x the floor) were tried
    # first and are right for finding a burst and wrong for timing one: the
    # ON edge trips at once, but the OFF edge only once the envelope has
    # fallen all the way to the floor -- through the DDC filter's tail -- so
    # with a quiet floor every ON run came out ~4 trace samples (0.6 ms)
    # long and every OFF run short. Measured through this chain: 700 Bd read
    # 598 Bd at eye 0.80. Half the tracked ON level is the step response's
    # midpoint, which a linear-phase filter crosses at the same delay going
    # up and going down.
    OOK_ON_FRACTION = 0.6       # hysteresis around the midpoint...
    OOK_OFF_FRACTION = 0.4      # ...as fractions of the tracked ON level
    OOK_LEVEL_TAU_S = 1.0       # peak-hold decay of that level between pulses
    # The OFF level is a running MINIMUM of the envelope, not an adaptive
    # floor. A floor that follows the envelope whenever it thinks it is OFF
    # cannot survive a keyed carrier in a room: once the reverberation of the
    # ON symbols has lifted it to within 15 dB of the carrier it averages ON
    # and OFF symbols alike (measured on a 50 Bd recording: level 0.058
    # learned at the first ON symbol, floor 0.032 half a second later,
    # nothing keyed for the rest of the run). A minimum over the last
    # OOK_OFF_WINDOW_S, allowed to rise slowly, is the OFF level whatever
    # the duty cycle.
    OOK_OFF_WINDOW_S = 0.1      # the minimum is taken over this much envelope
    OOK_OFF_RISE_DB_S = 6.0     # ...and may creep up this fast if the noise rises
    OOK_FLOOR_GUARD_ON = 2.0    # +6 dB over the OFF level to turn on...
    OOK_FLOOR_GUARD_OFF = 1.5   # ...and +3.5 dB to stay on
    OOK_LEARN_RATIO = 4.0       # a sample this far (+12 dB) over the OFF level
                                # may set the ON level; below it, nothing keys
    # -- FSK ---------------------------------------------------------------
    FSK_TOL_HZ = 1_000.0        # further than this from every tone = "none"
    FSK_BINS = 128              # histogram resolution of the tone estimate
    FSK_TONE_EMA = 0.3          # smoothing on the estimate (it jitters per block)
    # The tones are estimated from a rolling HISTORY of the frequency trace,
    # not from one block. A 2048-sample capture block is 410 baseband
    # samples, which at 300 Bd is under three symbols: a run of three equal
    # bits fills it with ONE tone, and a histogram of one tone hands back
    # the transition skirt as the second. Measured on synthetic 300 Bd FSK
    # through this chain: the estimate wandered on such blocks, the
    # tolerance then failed for a whole block, and a clean signal showed
    # undecided symbols. 0.2 s holds sixty symbols at 300 Bd, enough for
    # every tone to be present whatever the data says.
    FSK_TONE_HISTORY_S = 0.2
    # ...and an estimate is adopted only if EVERY tone actually carries
    # samples (a share of at least this, scaled by two over the tone count),
    # and adjacent tones are further apart than the tolerance: a spike and
    # its own shoulder are not a pair, and neither are two names for one tone.
    FSK_TONE_MIN_SHARE = 0.1
    # A real channel: the previous tone's reverberation beats with the new
    # tone and swings the instantaneous frequency around it (see
    # FskDemodulator.set_smoothing). So the trace is averaged over one beat
    # period of the ESTIMATED spacing before classification, and the
    # tolerance follows the spacing rather than staying at 1 kHz -- never
    # more than FSK_TOL_MAX_FRACTION of it, or adjacent zones would overlap.
    FSK_TOL_FRACTION = 0.25     # tolerance = max(FSK_TOL_HZ, this * spacing)...
    FSK_TOL_MAX_FRACTION = 0.45  # ...capped at this * spacing
    FSK_SMOOTH_MAX_TAPS = 64    # 1.3 ms at 50 kHz: never longer than this
    # KEYING KEEPS THE CARRIER ON, SO ITS ENVELOPE IS STEADY. Two or more
    # tones sounding TOGETHER -- a hum with harmonics, a pair of whines --
    # are not keying, but they look like it to every detector here: the
    # instantaneous frequency of their sum swings between them at the beat
    # rate, and the phase turns with it, on a perfectly regular grid that
    # the clock recovery locks onto. What gives it away is the envelope,
    # which dips as the tones cancel. Measured, envelope std/mean over half
    # a second:
    #
    #     synthetic FSK / 4-FSK            0.00 / 0.05
    #     synthetic DBPSK, 200-3000 Bd     0.04 - 0.33 (worst in a narrow band)
    #     FSK 25 Bd through this room      0.28 / 0.30
    #     two equal tones beating          0.48
    #     this room's 100-1000 Hz hum      0.65 - 0.68
    #
    # so the gate sits between the worst real signal and the mildest beat.
    # It does NOT apply to OOK, which is amplitude keying by definition.
    KEYED_MAX_ENVELOPE_CV = 0.45
    ENVELOPE_WINDOW_S = 0.5
    # -- DPSK --------------------------------------------------------------
    # Below this the constellation is not formed and the "bits" are the
    # decoder's own carrier error read back to it. Measured separation for
    # M = 2 (synthetic, fed in blocks): lock 0.91 at 10 dB SNR with every
    # bit right, 0.02 / 0.01 with a broken carrier. The 4th-power metric
    # falls faster: M = 4 read 0.88 at 15 dB with every bit right, and 0.62
    # at 10 dB with the bits wrong AND a spurious clock at 17 dB -- exactly
    # the confident wrong answer this gate exists for, so its bar is higher.
    # The M = 2 gate started at 0.3, which separated ultraScan's synthetic
    # cases (0.91-1.00 working, 0.01-0.02 broken). Real room noise sits in
    # between: measured on this room's 24.5-25.5 kHz band, the metric held
    # 0.31-0.40 for seconds at a time and published bits. Every synthetic
    # case that decoded correctly measured 0.88 or better, so the gate moved
    # to the middle of the gap that the measurements actually show.
    PSK_LOCK_MIN = {2: 0.6, 4: 0.75}
    # The lock is judged on a SMOOTHED value, not the instantaneous one.
    # Noise carries the metric across the gate for a block at a time:
    # measured on the room's own 25 kHz band, the DBPSK lock read 0.34 /
    # 0.04 / 0.42 / 0.08 / 0.60 / 0.16 on consecutive blocks and the decode
    # flickered on and off sixteen times, publishing a few bits each time. A
    # real constellation holds the metric up for as long as it is there.
    PSK_LOCK_EMA = 0.15
    # -- the clock ---------------------------------------------------------
    # THE THREE NUMBERS BELOW ARE ONE DECISION, not three. The clock estimate
    # needs MIN_TRANSITIONS (24) transitions inside its history, and random
    # data transitions on about half its symbol boundaries, so
    #
    #     usable baud_min  ~=  2 * MIN_TRANSITIONS / HISTORY_S  =  48 / 1.0  ~= 50
    #
    # Lowering BAUD_MIN without lengthening the history buys nothing: the
    # estimate simply returns "no clock" down there. And the search floor is
    # set BELOW the usable limit, not at it: a line sitting on the first bin
    # of the searched band is not an interior local maximum, so it is not a
    # candidate, and the scan walks up to the 2nd harmonic instead
    # (ultraScan measured 200 Bd at 17 dB for a 100 Bd signal with the floor
    # at the limit). Below the usable limit the transition count falls short
    # and the answer is "no clock", which is the safe side of the boundary.
    BAUD_MIN = 40.0
    BAUD_MAX = 5_000.0          # and clamped to fs_trace/8 per mode: the
                                # slicer's guard band drops a quarter off each
                                # end, so 8 samples/symbol is 4 left to vote on.
                                # For OOK (fs_bb/8) that caps the decode near
                                # 780 Bd.
    HISTORY_S = 1.0             # rolling trace the clock is estimated on
    ESTIMATE_S = 0.125          # how often it is re-estimated
    # TWO CLOCK TRACKERS, because a room needs a long look and a fast
    # signal needs a fine one. Measured on a 50 Bd FSK recording through
    # this room (scripts/verify/decode_recording.py): with 1 s of history
    # the estimate found 50 Bd in 14 windows of 20 at a median 13 dB, and
    # the 4th harmonic at 15-16 dB in the others -- the flips at every tone
    # switch blur the fundamental more than its harmonics. With 3 s it found
    # 50 Bd in 12 of 12 at 17-19 dB. But 3 s of a 50 kHz trace is a
    # million-point FFT eight times a second, and a 4000 Bd signal cannot
    # afford a coarser trace. So the full-rate trace keeps a short history
    # for fast rates, and a copy decimated by SLOW_DECIM (majority vote)
    # carries a long history for slow ones; OOK's trace is already at that
    # rate. The long one is believed when it is trustworthy; the short one
    # has to clear FAST_EXTRA_DB more to be believed inside the long one's
    # range, where a harmonic is the likelier explanation.
    SLOW_DECIM = 8
    SLOW_HISTORY_S = 4.0
    FAST_EXTRA_DB = 6.0
    # With 4 s of history the usable floor is ~12 Bd (2 * 24 / 4), so the
    # long tracker searches from below that. Slow signalling is what a
    # room allows (see the notes on the decode_over_the_air.py runs).
    SLOW_BAUD_MIN = 10.0
    # Trust, once given, is not withdrawn for a dip of a few dB: the
    # confidence is re-measured every ESTIMATE_S and jitters by that much
    # between windows of the same signal. Without this a marginal signal
    # flaps, and every flap discards the symbols in between.
    TRUST_HYSTERESIS_DB = 3.0
    GUARD = 0.25                # slicer guard band, each side of a boundary
    BITS_SHOWN = 64             # how much of the bit stream the readout keeps
    # A CLOCK YOU CANNOT SLICE SYMBOLS WITH IS NOT A CLOCK. The slicer marks
    # a symbol undecided when its guard window holds too few valid samples,
    # and on a trace that is mostly invalid -- an FSK classifier on noise,
    # where few samples land near any tone -- every symbol can come back
    # undecided while the clock still clears its threshold. Measured on the
    # room's own 5.3-6.3 kHz noise: 4-FSK locked at exactly 15.0 dB with an
    # eye of 0.00, i.e. not one symbol decided, and published bits from the
    # blocks that followed. Bits are withheld unless most of the recent
    # symbols were actually decided.
    MIN_DECIDED_SHARE = 0.5
    # ...and A CLOCK THAT DID NOT COME FROM THE SYMBOLS IS NOT A SYMBOL
    # CLOCK. A band holding one tone that pulses -- this room's 25 kHz pest
    # repeller pulses at exactly 1 kHz -- gives a frequency trace whose
    # labels flip on that grid, so the clock locks at 1000 Bd with an eye of
    # 0.95 while over 90% of the symbols are the SAME one: a tone being
    # chopped, not data. Real data spreads over its symbols (measured: 50%
    # for the top symbol of 2-level data, 25-30% for 4-level), and an idle
    # run of one symbol has no transitions, so it cannot produce a clock at
    # all. Above this share the readout says what it is instead of showing
    # bits. Four-level modes get a tighter bar: their data spreads over four
    # symbols (~25% each), and the chopped-tone case measured 75-84% on one
    # of four with a whole level never used at all.
    MAX_TOP_SYMBOL_SHARE = {2: 0.85, 4: 0.6}
    RECENT_SYMBOLS = 64         # the window both shares are measured over
    # ...and no fewer than this many symbols before either share is judged,
    # or a run of one is enough to fail the second test.
    MIN_SYMBOLS_TO_JUDGE = 8
    # The gate on showing bits at all. Measured through the estimator:
    # noise-only traces score 8-11 dB and move every window, real keying
    # 20-33 dB and holds still.
    DEFAULT_THRESHOLD_DB = 15.0
    # -- the analog readouts --------------------------------------------------
    ANALOG_WINDOW_S = 0.5       # the numbers are taken over this much waveform
    ANALOG_TRACE_S = 1.0        # the displayed waveform spans this...
    ANALOG_TRACE_POINTS = 400   # ...in this many points
    # Samples whose envelope sits this far under the running peak are not
    # read: with no carrier the instantaneous frequency of the noise is
    # anywhere in the band, and it must not be averaged into a "carrier".
    ANALOG_GATE_DB = 30.0
    ANALOG_PEAK_TAU_S = 2.0     # decay of that running peak
    ANALOG_MOD_MIN_HZ = 5.0     # the lowest modulation rate looked for
    ANALOG_MOD_MIN_DB = 10.0    # ...and how far its line must clear the spectrum's median
    ANALOG_MOD_EVERY = 4        # the modulation spectrum is recomputed every this many blocks
    # The demodulated waveform is averaged over this long before anything
    # reads it. In a room the direct sound and its reflections beat, and the
    # instantaneous frequency of the sum swings far beyond the transmitted
    # deviation at the beat rate: measured on a 12 kHz FM signal from the
    # speakers (+-500 Hz at 20 Hz), the dock read +-1250..1550 Hz. A
    # millisecond of averaging removes beats above ~1 kHz and leaves a
    # modulation up to ~200 Hz nearly untouched (0.94 at 200 Hz).
    ANALOG_SMOOTH_S = 0.001
    # ...and when the modulation is periodic, its deviation is read from
    # the height of its line in the spectrum rather than from the spread of
    # the waveform: the line holds the modulation, the spread also holds
    # whatever the room added.
    # A "carrier" whose frequency wanders over more than this share of the
    # band is noise, not a carrier: the 1st-99th percentile of the
    # instantaneous frequency of band-limited noise spans more than the
    # band (the FM "click" tails), while a modulated carrier's deviation is
    # designed to sit well inside it.
    ANALOG_CARRIER_MAX_SPREAD = 0.6

    def __init__(self, fs: float, mode: str = FSK,
                 threshold_db: float = DEFAULT_THRESHOLD_DB) -> None:
        if fs <= 0:
            raise ValueError(f"fs must be > 0, got {fs!r}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.fs = float(fs)
        self.fs_bb = self.fs / self.DECIM
        self._mode = mode
        self.threshold_db = float(threshold_db)
        self.enabled = True
        self._band: tuple[float, float] | None = None
        self._ddc: ComplexDdc | None = None
        self._clear_chain()
        self._clear_readout()

    # -- configuration -----------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_analog(self) -> bool:
        return self._mode in ANALOG_MODES

    @property
    def levels(self) -> int:
        """Symbol levels of a keyed mode (2 or 4); 0 for an analog one."""
        return _LEVELS.get(self._mode, 0)

    @property
    def bits_per_symbol(self) -> int:
        return _BITS_PER_SYMBOL.get(self._mode, 0)

    @property
    def psk_lock_min(self) -> float:
        """The constellation-lock gate of the current DPSK order (0 otherwise)."""
        return self.PSK_LOCK_MIN.get(self.levels, 0.0) if self._mode in _DIFFERENTIAL else 0.0

    @property
    def top_symbol_max(self) -> float:
        """How much of the symbol stream one symbol may take -- see MAX_TOP_SYMBOL_SHARE."""
        return self.MAX_TOP_SYMBOL_SHARE.get(self.levels, 1.0)

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == self._mode:
            return
        self._mode = mode
        self._rebuild()

    @property
    def band(self) -> tuple[float, float] | None:
        """(f_lo, width) being decoded, or None before the first configure()."""
        return self._band

    @property
    def decoded_width_hz(self) -> float:
        """How much of the band the decoder can actually see.

        The baseband runs at fs/5, so a band wider than fs_bb/2 is cut off at
        that: everything above f_lo + 25 kHz is outside the decoder.
        """
        return self._ddc.cutoff if self._ddc is not None else 0.0

    def configure(self, f_lo: float, width: float) -> None:
        """Point the decoder at a band. A change is a rebuild; no change is free."""
        band = (float(f_lo), float(width))
        if band == self._band:
            return
        self._band = band
        self._rebuild()

    @property
    def fs_trace(self) -> float:
        """Sample rate of the symbol trace the clock is recovered on."""
        return self.fs_bb / self.OOK_DECIM if self._mode == OOK else self.fs_bb

    @property
    def baud_range(self) -> tuple[float, float]:
        """(min, max) baud this mode can recover, for the readout; (0, 0) for analog."""
        if self.is_analog:
            return 0.0, 0.0
        return self.SLOW_BAUD_MIN, min(self.BAUD_MAX, self.fs_trace / 8.0)

    def reset(self) -> None:
        """Start over on the same band and mode."""
        self._rebuild()

    def _clear_chain(self) -> None:
        self._fm: FmDemodulator | None = None
        self._fsk: FskDemodulator | None = None
        self._fsk_tones: np.ndarray | None = None
        self._fsk_hist_len = 0
        self._fsk_freq_hist = np.zeros(0, dtype=np.float32)
        self._fsk_valid_hist = np.zeros(0, dtype=bool)
        self._env_hist = np.zeros(0, dtype=np.float32)
        self._env_len = 1
        self._dpsk: DpskDemodulator | None = None
        self._am: AmDemodulator | None = None
        self._pm: PmDemodulator | None = None
        self._ook_carry = np.zeros(0, dtype=np.float64)
        self._ook_level = 0.0          # tracked ON amplitude (peak hold)
        self._ook_level_decay = 1.0
        self._ook_on = False           # the midpoint slicer's state
        self._ook_off = 0.0            # tracked OFF level (running minimum)
        self._ook_off_hist = np.zeros(0, dtype=np.float64)
        self._ook_off_len = 1
        self._ook_off_rise = 1.0
        self._clock: SymbolClockTracker | None = None
        self._slow_clock: SymbolClockTracker | None = None
        self._slow_decim = 1
        self._slow_baud_max = 0.0
        self._slow_carry_sym = np.zeros(0, dtype=np.int16)
        self._slow_carry_valid = np.zeros(0, dtype=bool)
        self._clock_source: str | None = None
        self._slicer: SymbolSlicer | None = None
        self._bit_hist = np.zeros(0, dtype=np.int8)
        self._sym_hist = np.zeros(0, dtype=np.int8)
        self._agr_hist = np.zeros(0, dtype=np.float32)
        self._gap_pending = False
        self._lock_ema: float | None = None
        self._recent = np.zeros(0, dtype=np.int8)   # the last emitted symbols, trusted or not
        self._decided_share = 1.0     # of those, the share that was decided...
        self._top_share = 0.0         # ...and the share taken by the commonest one
        # analog
        self._analog_hist = np.zeros(0, dtype=np.float64)      # the waveform, ANALOG_WINDOW_S
        self._analog_valid_hist = np.zeros(0, dtype=bool)
        self._analog_freq_hist = np.zeros(0, dtype=np.float64)  # the FM trace alongside (for the carrier)
        self._analog_hist_len = 0
        self._analog_peak = 0.0
        self._analog_peak_decay = 1.0
        self._analog_disp = np.full(self.ANALOG_TRACE_POINTS, np.nan)
        self._analog_disp_decim = 1
        self._analog_disp_carry = np.zeros(0, dtype=np.float64)
        self._analog_disp_carry_valid = np.zeros(0, dtype=bool)
        self._pm_carrier: float | None = None
        self._analog_smooth_taps = 1
        self._pm_smooth_hist = np.zeros(0, dtype=np.float64)
        self._fm_raw: FmDemodulator | None = None

    def _clear_readout(self) -> None:
        # what the signal looks like
        self.last_fsk_tones: tuple[float, ...] = ()   # absolute Hz
        self.last_fsk_assigned = 0.0                   # share of samples on a tone
        self.last_env_cv = 0.0                         # keyed modes: envelope std/mean
        self.last_carrier_hz = 0.0                     # DPSK / analog, absolute Hz
        self.last_lock = 0.0                           # DPSK constellation lock
        self.amp_floor = 0.0                           # OOK OFF level
        self.last_on_fraction = 0.0                    # OOK share of ON samples
        # what the clock says
        self.last_baud_hz = 0.0
        self.last_baud_conf_db = 0.0
        self.decode_locked = False
        # what came out
        self.last_eye = 0.0
        self.last_bits = ""
        self.last_symbols: tuple[int, ...] = ()
        self.last_agreements: tuple[float, ...] = ()
        self.n_symbols = 0
        self.n_blocks = 0
        # analog
        self.analog_carrier = False                    # a carrier is there to read
        self.last_deviation = 0.0                      # FM: Hz peak; PM: rad peak; AM: depth 0..1
        self.last_mod_rate_hz = 0.0                    # 0.0 = no periodic modulation found
        self._mod_line_amp = 0.0                       # the modulation line's amplitude, waveform units
        self.last_level_db = -200.0                    # envelope, dBFS
        self.last_trace: tuple[float, ...] = ()        # the displayed waveform
        self.last_trace_lo = 0.0
        self.last_trace_hi = 0.0

    def _rebuild(self) -> None:
        self._clear_chain()
        self._clear_readout()
        if self._band is None:
            self._ddc = None
            return
        f_lo, width = self._band
        self._ddc = ComplexDdc(f_lo, width, self.fs, self.DECIM)
        fs_bb = self._ddc.fs_out
        if self._mode == OOK:
            self._am = AmDemodulator()
            fs_ook = fs_bb / self.OOK_DECIM
            self._ook_level_decay = 1.0 - 1.0 / (self.OOK_LEVEL_TAU_S * fs_ook)
            self._ook_off_len = max(1, int(self.OOK_OFF_WINDOW_S * fs_ook))
            self._ook_off_rise = 10.0 ** (self.OOK_OFF_RISE_DB_S / 20.0 / fs_ook)
        elif self._mode in _TONE_COUNT:
            self._fm = FmDemodulator(fs_bb)
            # Placeholder tones until the first estimate; process() does not
            # classify before one exists.
            n = _TONE_COUNT[self._mode]
            self._fsk = FskDemodulator(fs_bb, [k * fs_bb / (4.0 * n) for k in range(n)], self.FSK_TOL_HZ)
            self._fsk_hist_len = int(self.FSK_TONE_HISTORY_S * fs_bb)
        elif self._mode in _DIFFERENTIAL:
            self._dpsk = DpskDemodulator(fs_bb, order=_LEVELS[self._mode])
        else:
            taps = max(1, int(self.ANALOG_SMOOTH_S * fs_bb))
            self._fm = FmDemodulator(fs_bb, smooth_taps=taps)
            self._am = AmDemodulator(smooth_taps=taps)
            # unsmoothed, for the carrier test: averaging narrows the
            # spread of noise's instantaneous frequency too, and the test
            # is about that spread
            self._fm_raw = FmDemodulator(fs_bb)
            self._analog_smooth_taps = taps
            if self._mode == PM:
                self._pm = PmDemodulator(fs_bb)
            self._analog_hist_len = int(self.ANALOG_WINDOW_S * fs_bb)
            self._analog_peak_decay = 1.0 - 1.0 / (self.ANALOG_PEAK_TAU_S * fs_bb)
            self._analog_disp_decim = max(1, int(fs_bb * self.ANALOG_TRACE_S / self.ANALOG_TRACE_POINTS))
            return

        if self._mode != OOK:
            self._env_len = max(1, int(self.ENVELOPE_WINDOW_S * fs_bb))

        fs_trace = self.fs_trace
        baud_max = min(self.BAUD_MAX, fs_trace / 8.0)
        if baud_max > self.BAUD_MIN:
            history = max(2048, int(self.HISTORY_S * fs_trace))
            interval = max(512, min(history // 4, int(self.ESTIMATE_S * fs_trace)))
            self._clock = SymbolClockTracker(
                fs_trace, baud_min=self.BAUD_MIN, baud_max=baud_max,
                history=history, interval=interval)
            self._slicer = SymbolSlicer(n_levels=self.levels, guard=self.GUARD)
            # the long-history tracker on the decimated trace, see SLOW_DECIM
            self._slow_decim = 1 if self._mode == OOK else self.SLOW_DECIM
            fs_slow = fs_trace / self._slow_decim
            self._slow_baud_max = min(self.BAUD_MAX, fs_slow / 8.0)
            if self._slow_baud_max > self.BAUD_MIN:
                slow_history = max(2048, int(self.SLOW_HISTORY_S * fs_slow))
                slow_interval = max(256, min(slow_history // 4, int(self.ESTIMATE_S * fs_slow)))
                self._slow_clock = SymbolClockTracker(
                    fs_slow, baud_min=self.SLOW_BAUD_MIN, baud_max=self._slow_baud_max,
                    history=slow_history, interval=slow_interval)

    # -- the chain ---------------------------------------------------------

    def process(self, block: np.ndarray) -> None:
        """One capture block (real, at fs). Never raises on ordinary data."""
        if self._ddc is None:
            return
        z = self._ddc.process(block)
        if z.size == 0:
            return
        self.n_blocks += 1

        if self.is_analog:
            self._analog(z)
            return
        if self._mode != OOK:
            # How steady the envelope is -- see KEYED_MAX_ENVELOPE_CV.
            # Measured on the BAND, before any detector: it is a property of
            # what is in the band, and it is the reason a keyed model may be
            # the wrong one, so it must not depend on that model working.
            self._env_hist = np.concatenate(
                (self._env_hist, np.abs(z).astype(np.float32)))[-self._env_len:]
            if self._env_hist.size >= self._env_len // 2:
                e = self._env_hist.astype(np.float64)
                self.last_env_cv = float(e.std() / max(e.mean(), 1e-12))
        if self._mode == OOK:
            trace, valid = self._trace_ook(z)
        elif self._mode in _TONE_COUNT:
            trace, valid = self._trace_fsk(z)
        else:
            trace, valid = self._trace_dpsk(z)

        if trace.size:
            self._decode_trace(trace, valid)

    def _trace_ook(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._am is not None
        env = self._am.process(z).astype(np.float64)
        # x8 by block mean, with the remainder carried so the decimation grid
        # does not depend on how the capture was chopped
        e = np.concatenate((self._ook_carry, env))
        n8 = e.size // self.OOK_DECIM
        dec = e[:n8 * self.OOK_DECIM].reshape(-1, self.OOK_DECIM).mean(axis=1)
        self._ook_carry = e[n8 * self.OOK_DECIM:]
        if dec.size == 0:
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=bool)

        # The OFF level: a running minimum, see OOK_OFF_WINDOW_S.
        self._ook_off_hist = np.concatenate((self._ook_off_hist, dec))[-self._ook_off_len:]
        window_min = float(self._ook_off_hist.min())
        off = min(window_min, self._ook_off * self._ook_off_rise ** dec.size) if self._ook_off > 0.0 else window_min
        self._ook_off = off
        self.amp_floor = off

        # The midpoint slicer -- see OOK_ON_FRACTION. The ON level is only
        # ever learned from samples well above the OFF level, and nothing
        # keys until one has been: a band with no carrier in it stays OFF.
        on = np.zeros(dec.size, dtype=np.bool_)
        level, state, decay = self._ook_level, self._ook_on, self._ook_level_decay
        learn_from = self.OOK_LEARN_RATIO * off
        floor_on = self.OOK_FLOOR_GUARD_ON * off
        floor_off = self.OOK_FLOOR_GUARD_OFF * off
        for i in range(dec.size):
            x = float(dec[i])
            if x > level and x > learn_from:
                level = x
            else:
                level *= decay
            if level < learn_from:
                state = False                # no carrier over the OFF level: nothing to key
            elif state:
                state = x >= max(self.OOK_OFF_FRACTION * level, floor_off)
            else:
                state = x >= max(self.OOK_ON_FRACTION * level, floor_on)
            on[i] = state
        self._ook_level, self._ook_on = level, state
        self.last_on_fraction = float(np.mean(on))
        return on.astype(np.int8), np.ones(on.size, dtype=bool)

    def _trace_fsk(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._fm is not None and self._fsk is not None and self._band is not None
        n_tones = _TONE_COUNT[self._mode]
        freq, valid = self._fm.process(z)
        # The tone set is re-estimated every block from the frequency
        # histogram of the last FSK_TONE_HISTORY_S seconds and smoothed. Too
        # few samples for a histogram is not an error here, just no update.
        self._fsk_freq_hist = np.concatenate(
            (self._fsk_freq_hist, freq))[-self._fsk_hist_len:]
        self._fsk_valid_hist = np.concatenate(
            (self._fsk_valid_hist, valid))[-self._fsk_hist_len:]
        est: list[float] = []
        if int(self._fsk_valid_hist.sum()) >= 4 * self.FSK_BINS:
            # The histogram spans the 1st-99th percentile of the trace, not
            # its extremes: in a room the swings at a tone switch reach far
            # outside the tones, and a histogram stretched over them puts
            # them all in a couple of coarse bins.
            f_valid = self._fsk_freq_hist[self._fsk_valid_hist]
            lo, hi = np.percentile(f_valid, (1.0, 99.0))
            keep = self._fsk_valid_hist & (self._fsk_freq_hist >= lo) & (self._fsk_freq_hist <= hi)
            try:
                est = estimate_fsk_tones(self._fsk_freq_hist, keep, n_tones,
                                         bins=self.FSK_BINS, exclude_hz=self.FSK_TOL_HZ)
            except ValueError:
                est = []
        if len(est) == n_tones and self._tones_are_a_set(est):
            tones = np.sort(np.asarray(est, dtype=np.float64))
            self._fsk_tones = (tones if self._fsk_tones is None else
                               (1.0 - self.FSK_TONE_EMA) * self._fsk_tones
                               + self.FSK_TONE_EMA * tones)
            self._fsk.tone_freqs_hz = self._fsk_tones
            spacing = float(np.min(np.diff(self._fsk_tones)))
            taps = int(np.clip(round(self._fsk.fs_bb / spacing), 1, self.FSK_SMOOTH_MAX_TAPS))
            self._fsk.set_smoothing(taps)
            self._fm.set_smoothing(taps)
            self._fsk.tolerance_hz = min(max(self.FSK_TOL_HZ, self.FSK_TOL_FRACTION * spacing),
                                         self.FSK_TOL_MAX_FRACTION * spacing)
        if self._fsk_tones is None:
            return np.empty(0, dtype=np.int8), np.empty(0, dtype=bool)
        f_lo = self._band[0]
        self.last_fsk_tones = tuple(float(t + f_lo) for t in self._fsk_tones)

        idx, ok = self._fsk.process(z)
        self.last_fsk_assigned = float(np.mean(ok)) if ok.size else 0.0
        return idx, ok

    def _tones_are_a_set(self, est: list[float]) -> bool:
        """Every candidate tone carries samples, and no two are one tone twice."""
        tones = np.sort(np.asarray(est, dtype=np.float64))
        if tones.size < 2 or float(np.min(np.diff(tones))) <= self.FSK_TOL_HZ:
            return False
        f = self._fsk_freq_hist[self._fsk_valid_hist].astype(np.float64)
        if f.size == 0:
            return False
        min_share = self.FSK_TONE_MIN_SHARE * 2.0 / tones.size
        for tone in tones:
            share = float(np.mean(np.abs(f - tone) <= self.FSK_TOL_HZ))
            if share < min_share:
                return False
        return True

    def _trace_dpsk(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._dpsk is not None and self._band is not None
        sym, valid = self._dpsk.process(z)
        raw = self._dpsk.last_lock
        # smoothed, see PSK_LOCK_EMA; the raw value is still what the
        # detector reports, this is what the gate reads
        self._lock_ema = (raw if self._lock_ema is None
                          else (1.0 - self.PSK_LOCK_EMA) * self._lock_ema + self.PSK_LOCK_EMA * raw)
        self.last_lock = self._lock_ema
        self.last_carrier_hz = self._dpsk.last_carrier_hz + self._band[0]
        return sym, valid

    def _decode_trace(self, sym: np.ndarray, valid: np.ndarray) -> None:
        """One block of symbol trace -> the bit readout.

        THE SLICER IS FED EVEN WHEN NOTHING IS LOCKED. Both it and the clock
        tracker index the stream by ABSOLUTE sample number, and that is the
        only reason the tracker's phase can be handed to the slicer as-is.
        Skipping the slicer while unlocked would desynchronise those two
        counts, so it is fed always and discards internally.
        """
        if self._clock is None or self._slicer is None:
            return
        if not self.enabled:
            self.decode_locked = False
            return
        slicer = self._slicer
        self._clock.process(sym, valid)
        if self._slow_clock is not None:
            self._feed_slow(sym, valid)

        needed_db = self.threshold_db - (self.TRUST_HYSTERESIS_DB if self.decode_locked else 0.0)
        rate, conf, period, phase, believed = self._pick_clock(needed_db)
        self.last_baud_hz = rate
        self.last_baud_conf_db = conf
        trusted = (believed
                   and (self._mode not in _DIFFERENTIAL or self.last_lock >= self.psk_lock_min)
                   and (self._mode == OOK or self.last_env_cv <= self.KEYED_MAX_ENVELOPE_CV))
        slicer.set_clock(period if trusted else 0.0, phase)
        labels, agree = slicer.process(sym, valid)
        # A rolling window of what the slicer actually emitted, whether or
        # not it was trusted, so the gates below see the symbols the readout
        # would show. A clock with no symbols coming out of it is not a
        # decode at all: measured on this room's 5.3-6.3 kHz noise, 4-FSK
        # reported "locked" at 11.2 Bd -- one symbol per eleven blocks --
        # with nothing emitted and an eye of 0.00.
        if labels.size:
            self._recent = np.concatenate((self._recent, labels))[-self.RECENT_SYMBOLS:]
        if self._recent.size < self.MIN_SYMBOLS_TO_JUDGE:
            trusted = False
        else:
            self._decided_share = float(np.mean(self._recent >= 0))
            decided = self._recent[self._recent >= 0]
            self._top_share = (float(np.bincount(decided, minlength=self.levels).max()) / decided.size
                               if decided.size else 1.0)
            if self._decided_share < self.MIN_DECIDED_SHARE:
                trusted = False          # see MIN_DECIDED_SHARE
            elif self._top_share > self.top_symbol_max:
                trusted = False          # see MAX_TOP_SYMBOL_SHARE
        self.decode_locked = bool(trusted)
        if not trusted:
            # Whatever arrives while untrusted is discarded by the slicer.
            # If bits are shown again later they must not read as one run
            # with the ones before: an undecided marker says where the gap is.
            self._gap_pending = self._sym_hist.size > 0
            return
        if labels.size == 0:
            return
        if self._gap_pending:
            self._sym_hist = np.concatenate((self._sym_hist, np.int8([-1])))
            self._agr_hist = np.concatenate((self._agr_hist, np.float32([0.0])))
            self._bit_hist = np.concatenate((self._bit_hist, np.int8([-1])))
            self._gap_pending = False

        self.n_symbols = slicer.n_symbols
        self._sym_hist = np.concatenate((self._sym_hist, labels))[-self.BITS_SHOWN:]
        self._agr_hist = np.concatenate((self._agr_hist, agree))[-self.BITS_SHOWN:]
        self.last_symbols = tuple(int(x) for x in self._sym_hist)
        self.last_agreements = tuple(float(x) for x in self._agr_hist)
        decided = agree[labels >= 0]
        if decided.size:
            self.last_eye = float(decided.mean())

        # DPSK carries data in the CHANGES (raising to the M-th power leaves
        # the labelling unknowable, by construction). FSK and OOK carry it
        # in the level, and whether a given link then adds NRZI on top is
        # not something this can know -- so the level is what is shown.
        bps = self.bits_per_symbol
        if self._mode in _DIFFERENTIAL:
            steps = differential_bits(self._sym_hist, n_levels=self.levels)
            self._bit_hist = symbols_to_bits(steps, bps)[-self.BITS_SHOWN:]
        else:
            self._bit_hist = np.concatenate(
                (self._bit_hist, symbols_to_bits(labels, bps)))[-self.BITS_SHOWN:]
        self.last_bits = bits_to_text(self._bit_hist, group=8)

    def _feed_slow(self, sym: np.ndarray, valid: np.ndarray) -> None:
        """The decimated copy of the trace for the long-history tracker.

        Groups of SLOW_DECIM samples become one label by majority of the
        valid ones, undecided (-1) when fewer than half are valid. Decimated
        sample j covers full-rate samples [j*d, (j+1)*d), which is what
        _pick_clock relies on to hand its clock to the full-rate slicer.
        """
        assert self._slow_clock is not None
        d = self._slow_decim
        if d == 1:
            self._slow_clock.process(sym, valid)
            return
        s = np.concatenate((self._slow_carry_sym, np.asarray(sym, dtype=np.int16)))
        v = np.concatenate((self._slow_carry_valid, np.asarray(valid, dtype=bool)))
        n = s.size // d
        if n:
            S = s[:n * d].reshape(n, d)
            V = v[:n * d].reshape(n, d) & (S >= 0)
            counts = np.stack([(V & (S == k)).sum(axis=1) for k in range(self.levels)], axis=1)
            count = V.sum(axis=1)
            ok = count * 2 >= d
            label = counts.argmax(axis=1).astype(np.int8)
            label[~ok] = -1
            self._slow_clock.process(label, ok)
        self._slow_carry_sym = s[n * d:]
        self._slow_carry_valid = v[n * d:]

    def _pick_clock(self, needed_db: float):
        """(rate, confidence, period, phase, believed) from the two trackers.

        The long history is believed whenever it clears the threshold; the
        short one is believed on its own only above the long one's range,
        and inside that range only with FAST_EXTRA_DB to spare. With
        nothing believed, the better candidate is reported -- as a
        candidate. Period and phase are in FULL-RATE samples either way.
        """
        fast, slow = self._clock, self._slow_clock
        assert fast is not None
        cands = {}
        if slow is not None and slow.locked:
            d = self._slow_decim
            cands["slow"] = (slow.rate_hz, slow.confidence_db,
                             slow.period_samples * d, slow.phase_abs * d + d / 2.0,
                             slow.confidence_db >= needed_db)
        if fast.locked:
            bar = needed_db
            if slow is not None and fast.rate_hz <= self._slow_baud_max:
                bar += self.FAST_EXTRA_DB
            cands["fast"] = (fast.rate_hz, fast.confidence_db,
                             fast.period_samples, fast.phase_abs,
                             fast.confidence_db >= bar)
        if not cands:
            self._clock_source = None
            return 0.0, 0.0, 0.0, 0.0, False
        believed = {name: c for name, c in cands.items() if c[4]}
        if believed:
            # A believed rate above the long history's range can only have
            # come from the full-rate tracker, and whatever the decimated
            # trace made of that signal is an alias: measured, 4000 Bd
            # decimated by 8 gave the long history a 250 Bd line at 18 dB.
            # This outranks the stickiness below.
            if "fast" in believed and believed["fast"][0] > self._slow_baud_max:
                self._clock_source = "fast"
            # STICKY: otherwise the source in use keeps the job while it is
            # believed. The two trackers' grids can sit half a period apart,
            # and if the choice flapped between them the slicer would snap
            # back and forth and emit symbols twice. Measured on a 25 Bd FSK
            # recording through this room: 334 duplicated symbols in 580.
            elif self._clock_source not in believed:
                self._clock_source = "slow" if "slow" in believed else "fast"
            return believed[self._clock_source]
        self._clock_source = None
        return max(cands.values(), key=lambda c: c[1])

    # -- the analog modes ----------------------------------------------------

    def _analog(self, z: np.ndarray) -> None:
        """FM / AM / PM: a waveform, and the numbers that describe it."""
        assert self._fm is not None and self._fm_raw is not None and self._am is not None and self._band is not None
        f_lo = self._band[0]
        env = self._am.process(z).astype(np.float64)
        freq, _ = self._fm.process(z)
        freq = freq.astype(np.float64)
        freq_raw = self._fm_raw.process(z)[0].astype(np.float64)

        # The gate: a running peak of the envelope, and everything more than
        # ANALOG_GATE_DB under it is not read.
        peak = max(self._analog_peak * self._analog_peak_decay ** env.size, float(env.max()))
        self._analog_peak = peak
        valid = env >= peak * 10.0 ** (-self.ANALOG_GATE_DB / 20.0)

        if self._mode == FM:
            x = freq + f_lo
        elif self._mode == AM:
            x = env
        else:
            assert self._pm is not None
            # The carrier's own rotation is removed from the phase: a slow
            # estimate from the frequency trace, and whatever is left of it
            # detrended per window below.
            if valid.any():
                f_med = float(np.median(freq_raw[valid]))
                self._pm_carrier = (f_med if self._pm_carrier is None
                                    else 0.9 * self._pm_carrier + 0.1 * f_med)
                self._pm.set_offset(self._pm_carrier)
            x = self._pm.process(z).astype(np.float64)
            # the same averaging the FM and AM detectors apply internally
            taps = self._analog_smooth_taps
            if taps > 1:
                stream = np.concatenate((self._pm_smooth_hist, x))
                if stream.size >= taps:
                    smoothed = np.convolve(stream, np.full(taps, 1.0 / taps), mode="valid")
                    x = smoothed[-x.size:] if smoothed.size >= x.size else np.concatenate(
                        (np.full(x.size - smoothed.size, smoothed[0] if smoothed.size else x[0]), smoothed))
                self._pm_smooth_hist = stream[-(taps - 1):]

        self._analog_hist = np.concatenate((self._analog_hist, x))[-self._analog_hist_len:]
        self._analog_valid_hist = np.concatenate((self._analog_valid_hist, valid))[-self._analog_hist_len:]
        self._analog_freq_hist = np.concatenate((self._analog_freq_hist, freq_raw))[-self._analog_hist_len:]
        self._analog_display(x, valid)
        self._analog_readout(env)

    def _analog_display(self, x: np.ndarray, valid: np.ndarray) -> None:
        """Decimate the waveform onto the display's grid, valid samples only."""
        d = self._analog_disp_decim
        s = np.concatenate((self._analog_disp_carry, x))
        v = np.concatenate((self._analog_disp_carry_valid, valid))
        n = s.size // d
        if n:
            S = s[:n * d].reshape(n, d)
            V = v[:n * d].reshape(n, d)
            count = V.sum(axis=1)
            points = np.where(count > 0, (S * V).sum(axis=1) / np.maximum(count, 1), np.nan)
            self._analog_disp = np.concatenate((self._analog_disp, points))[-self.ANALOG_TRACE_POINTS:]
        self._analog_disp_carry = s[n * d:]
        self._analog_disp_carry_valid = v[n * d:]

    def _analog_readout(self, env: np.ndarray) -> None:
        h, v, f = self._analog_hist, self._analog_valid_hist, self._analog_freq_hist
        assert self._band is not None
        width = self.decoded_width_hz
        good = h[v]
        self.last_level_db = float(20.0 * np.log10(max(float(np.sqrt(np.mean(env ** 2))), 1e-10)))
        # A carrier is there if enough of the window passed the gate AND the
        # frequency trace stays in one place: the FM readout of white noise
        # wanders over the whole band.
        f_good = f[v]
        spread = float(np.percentile(f_good, 99) - np.percentile(f_good, 1)) if f_good.size > 10 else width
        has_carrier = (good.size >= 0.2 * max(h.size, 1) and h.size >= self._analog_hist_len // 2
                       and spread < self.ANALOG_CARRIER_MAX_SPREAD * width)
        self.analog_carrier = bool(has_carrier)

        if self._mode == PM and good.size > 2:
            # the residual ramp of an imperfect carrier estimate is a line;
            # what is left is the modulation
            idx = np.flatnonzero(v).astype(np.float64)
            slope, intercept = np.polyfit(idx, good, 1)
            detrended = good - (slope * idx + intercept)
        elif good.size:
            detrended = good - float(np.median(good))
        else:
            detrended = good

        if has_carrier and good.size:
            if self.n_blocks % self.ANALOG_MOD_EVERY == 0:
                self.last_mod_rate_hz, self._mod_line_amp = self._modulation_rate(detrended, v)
            p1, p99 = np.percentile(detrended, (1.0, 99.0))
            # the swing of the waveform, or -- when it is periodic -- the
            # height of its line, see ANALOG_SMOOTH_S
            swing = 0.5 * float(p99 - p1)
            amplitude = self._mod_line_amp if self.last_mod_rate_hz > 0.0 else swing
            if self._mode == FM:
                # the carrier from the UNSMOOTHED trace, as the other modes
                # read it: averaging spreads the room's one-sided spikes into
                # their neighbours and moved this median by 40-80 Hz on a
                # 12 kHz signal from the speakers; the raw median ignores them
                self.last_carrier_hz = float(np.median(f_good)) + self._band[0] if f_good.size else 0.0
                self.last_deviation = amplitude
            elif self._mode == AM:
                self.last_carrier_hz = float(np.median(f_good)) + self._band[0] if f_good.size else 0.0
                mean_env = float(np.mean(good))
                self.last_deviation = float(amplitude / max(mean_env, 1e-12))
            else:
                self.last_carrier_hz = (self._pm_carrier or 0.0) + self._band[0]
                self.last_deviation = amplitude
        else:
            self.last_deviation = 0.0
            self.last_mod_rate_hz = 0.0
            self._mod_line_amp = 0.0

        # the display: NaN (nothing valid in that cell) carried forward
        disp = self._analog_disp.copy()
        if np.isnan(disp).all():
            self.last_trace = ()
            self.last_trace_lo = self.last_trace_hi = 0.0
            return
        known = ~np.isnan(disp)
        first = float(disp[known][0])
        for i in range(disp.size):
            if np.isnan(disp[i]):
                disp[i] = first
            else:
                first = disp[i]
        if self._mode == PM and known.sum() > 2:
            # show the same detrended phase the numbers describe
            k = np.arange(disp.size, dtype=np.float64)
            disp = disp - np.polyval(np.polyfit(k[known], disp[known], 1), k)
        self.last_trace = tuple(float(p) for p in disp)
        self.last_trace_lo = float(disp.min())
        self.last_trace_hi = float(disp.max())

    def _modulation_rate(self, detrended: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
        """(dominant rate of the modulation, its amplitude), or (0.0, 0.0).

        The window's spectrum, its tallest line above ANALOG_MOD_MIN_HZ, and
        that only if it clears the median by ANALOG_MOD_MIN_DB: a waveform
        with no periodicity has a flat spectrum, and reporting its tallest
        bin would be reporting noise as a rate. The amplitude is the line's
        POWER, summed over the three bins of the Hann main lobe so a rate
        between bin centres is not read low, scaled back to the waveform's
        units by Parseval: a real sinusoid of amplitude A windowed by w
        puts N * A^2 * sum(w^2) / 4 of |X|^2 into its positive-frequency
        lobe. (Summing magnitudes instead read 18% high.)
        """
        full = np.zeros(valid.size, dtype=np.float64)
        full[valid] = detrended
        n = full.size
        if n < 64:
            return 0.0, 0.0
        window = np.hanning(n)
        spec = np.abs(np.fft.rfft(full * window))
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs_bb)
        band = freqs >= self.ANALOG_MOD_MIN_HZ
        if band.sum() < 8:
            return 0.0, 0.0
        s = spec[band]
        floor = float(np.median(s))
        k = int(np.argmax(s))
        if floor <= 0.0 or 20.0 * np.log10(float(s[k]) / floor) < self.ANALOG_MOD_MIN_DB:
            return 0.0, 0.0
        k_abs = int(np.flatnonzero(band)[k])
        power = float(np.sum(spec[max(k_abs - 1, 0):k_abs + 2] ** 2))
        amplitude = float(np.sqrt(4.0 * power / (n * float(np.sum(window ** 2)))))
        return float(freqs[band][k]), amplitude

    # -- the readout, in words ---------------------------------------------

    def signal_text(self) -> str:
        """What the detector sees, before any clock: tones / carrier / keying."""
        if self._ddc is None:
            return "no band selected"
        label = MODE_LABELS[self._mode]
        if self._mode in _TONE_COUNT:
            if not self.last_fsk_tones:
                return "%s: looking for %d tones" % (label, _TONE_COUNT[self._mode])
            tones = " / ".join("%.2f" % (t / 1e3) for t in self.last_fsk_tones)
            return ("%s tones %s kHz, %d%% of samples on a tone, envelope %.2f%s"
                    % (label, tones, round(100.0 * self.last_fsk_assigned), self.last_env_cv,
                       "" if self.last_env_cv <= self.KEYED_MAX_ENVELOPE_CV else " (too unsteady for keying)"))
        if self._mode in _DIFFERENTIAL:
            return ("%s carrier %.2f kHz, constellation lock %.2f%s"
                    % (label, self.last_carrier_hz / 1e3, self.last_lock,
                       "" if self.last_lock >= self.psk_lock_min else " (no lock)"))
        if self._mode == OOK:
            return ("OOK: %d%% of the time on, floor %.1e"
                    % (round(100.0 * self.last_on_fraction), self.amp_floor))
        # analog
        if not self.analog_carrier:
            return "%s: no steady carrier in the band" % label
        rate = ("modulated at %.1f Hz" % self.last_mod_rate_hz if self.last_mod_rate_hz > 0.0
                else "no periodic modulation")
        if self._mode == FM:
            return ("FM carrier %.3f kHz, deviation ±%.0f Hz, %s"
                    % (self.last_carrier_hz / 1e3, self.last_deviation, rate))
        if self._mode == AM:
            return ("AM carrier %.3f kHz at %.0f dBFS, depth %.0f%%, %s"
                    % (self.last_carrier_hz / 1e3, self.last_level_db,
                       100.0 * self.last_deviation, rate))
        return ("PM carrier %.3f kHz, deviation ±%.2f rad, %s"
                % (self.last_carrier_hz / 1e3, self.last_deviation, rate))

    def decode_text(self) -> str:
        """The clock's verdict. Always the baud AND its confidence, never the baud alone."""
        if self._ddc is None:
            return ""
        if self.is_analog:
            return "analog: the waveform below is the demodulated signal; a keyed mode gives bits"
        if not self.enabled:
            return "decode off"
        if self.decode_locked:
            return ("%.0f Bd, confidence %.0f dB, eye %.2f, %d symbols"
                    % (self.last_baud_hz, self.last_baud_conf_db,
                       self.last_eye, self.n_symbols))
        if self._mode in _DIFFERENTIAL and self.last_baud_hz > 0.0 \
                and self.last_lock < self.psk_lock_min:
            return ("candidate %.0f Bd at %.0f dB, but the constellation is not locked"
                    % (self.last_baud_hz, self.last_baud_conf_db))
        if self._mode != OOK and self.last_baud_hz > 0.0 \
                and self.last_env_cv > self.KEYED_MAX_ENVELOPE_CV:
            return ("%.0f Bd periodicity at %.0f dB, but the envelope swings too much for keying "
                    "(%.2f): these are tones sounding together and beating, not one tone at a time"
                    % (self.last_baud_hz, self.last_baud_conf_db, self.last_env_cv))
        if self.last_baud_hz > 0.0 and self._decided_share < self.MIN_DECIDED_SHARE:
            return ("candidate %.0f Bd at %.0f dB, but only %.0f%% of its symbols can be decided"
                    % (self.last_baud_hz, self.last_baud_conf_db, 100.0 * self._decided_share))
        if self.last_baud_hz > 0.0 and self._recent.size < self.MIN_SYMBOLS_TO_JUDGE:
            return ("candidate %.0f Bd at %.0f dB, but no symbols have come out of it yet"
                    % (self.last_baud_hz, self.last_baud_conf_db))
        if self.last_baud_hz > 0.0 and self._top_share > self.top_symbol_max:
            return ("%.0f Bd periodicity at %.0f dB, but %.0f%% of its symbols are the same one: "
                    "something in this band pulses at that rate, it is not carrying data"
                    % (self.last_baud_hz, self.last_baud_conf_db, 100.0 * self._top_share))
        if self.last_baud_hz > 0.0:
            return ("candidate %.0f Bd at %.0f dB (below the %.0f dB threshold)"
                    % (self.last_baud_hz, self.last_baud_conf_db, self.threshold_db))
        return "no clock found"

    def trace_unit(self) -> str:
        """Unit of the analog waveform, for the display's axis labels."""
        return {FM: "Hz", AM: "", PM: "rad"}.get(self._mode, "")
