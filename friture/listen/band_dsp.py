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

"""Two ways to hear only the selected band, both retunable while running.

BANDPASS
    Keep [f_lo, f_hi] where it is and mute everything else. Pitch is
    preserved, so what you hear lines up with what the plot shows.

HETERODYNE
    Move [f_lo, f_hi] down onto [0, width]. This is ultraScan's audification
    chain, and on a 250 kHz capture it is not one mode of two but the only
    one that can produce a sound: playback runs at 48 kHz, so a band-pass at
    45 kHz selects something the speakers then throw away. The cost is that
    the pitch changes -- and while the band is being dragged, a fixed source
    glides in pitch as the band moves under it.

One structure serves both (a digital down-converter):

    x -> * e^-j2*pi*f_c*t -> lowpass(width/2) -> * e^+j2*pi*f_up*t -> 2*Re()

with f_up = f_c for band-pass (put it back where it was) and f_up = width/2
for heterodyne (leave it at baseband). The band's *position* lives entirely in
the two mixer frequencies and the *width* entirely in the lowpass, which is
what makes dragging the band around possible without breaking the audio:

  moving the band   changes only the mixer frequency. Keeping the phase
                    accumulator and bending its increment leaves the mixing
                    sinusoid continuous in amplitude and phase, so there is
                    no discontinuity for a click to come from. (This is
                    ultraScan's M14 trick, which the older structure here --
                    taps modulated to the band centre -- could not use,
                    because there the taps moved with the band.)

  changing the width changes the taps, which cannot be free. But the overlap-
                    save history holds *input* samples rather than filter
                    state, so the old and new impulse responses can both be
                    applied to it and cross-faded. Inaudible, though the
                    timbre morphs over the fade rather than switching.

Input and output are both at the capture rate, so n samples in always gives
exactly n samples out.

How narrow a band can be, and what it costs
-------------------------------------------
A windowed-sinc FIR of N taps rolls off over ~3.3*fs/N Hz, so resolving a
narrow band means a long filter, and a long filter delays what you hear by
its group delay of (N-1)/2 samples. That trade is not an implementation
detail to be optimised away -- it is the time-frequency uncertainty
principle. You cannot separate a 50 Hz-wide band without listening to ~33 ms
of signal first. So the tap count follows the requested bandwidth, and only a
deliberately narrow band pays the delay. At 250 kHz:

    band      taps    delay
    10 kHz+    255    0.5 ms
     2 kHz    1023    2.0 ms
    500 Hz    4095    8.2 ms
    100 Hz   16383   32.8 ms      <- N_TAPS_MAX, the floor (~50 Hz)

Filtering runs as FFT overlap-save at a transform size fixed to the longest
filter, so a width change never has to rebuild the history -- and the cost is
the same whatever the width. See N_TAPS_MAX for what that cost is.

Implemented on numpy alone. Friture does not depend on scipy, and one feature
is not a reason to add it.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Shortest filter we ever build: below this the passband of a typical few-kHz
# band stops being flat, and there is nothing to gain.
N_TAPS_MIN = 255

# Longest. Sets the narrowest selectable band (~50 Hz at 250 kHz) and the
# worst case delay (~33 ms). Both scale with the capture rate, so this is not
# the same number that suited 48 kHz.
#
# Unlike at 48 kHz, CPU is what caps it here rather than taste: the overlap-
# save transform grows with the tap count while the block it has to keep up
# with is only 8.2 ms long. Measured on this machine, per 2048-sample block:
#
#     8191 taps   101 Hz    16 ms delay    0.31 ms   4% of the block
#    16383 taps    50 Hz    33 ms delay    1.39 ms  17%      <- here
#    32767 taps    25 Hz    66 ms delay    3.06 ms  37%
#    65535 taps    13 Hz   131 ms delay    7.81 ms  95%      hopeless
#
# and two of these run at once (the live monitor and the player), so 37% each
# is already most of a core. 50 Hz is 0.1% of a 40 kHz bat call in any case.
N_TAPS_MAX = 16383

# Aim for a roll-off half the width of the band, so the middle of the band is
# flat rather than all skirt.
_SKIRT_FRACTION = 0.5

# Block size the overlap-save transform is sized for. Bigger blocks still work
# (they are split), this only picks the transform size.
_BLOCK_HINT = 1024

BANDPASS = 0
HETERODYNE = 1


def transition_width(fs: float, n_taps: int) -> float:
    """Width of a Hamming-windowed sinc's roll-off, in Hz."""
    return 3.3 * fs / n_taps


def min_bandwidth(fs: float) -> float:
    """Narrowest band the longest filter can still resolve."""
    return transition_width(fs, N_TAPS_MAX)


def group_delay_seconds(n_taps: int, fs: float) -> float:
    """How far behind the input the filtered audio comes out."""
    return (n_taps - 1) / 2.0 / fs


def taps_for_bandwidth(bandwidth: float, fs: float) -> int:
    """Shortest filter that can resolve this band, as 2^k - 1.

    Odd lengths keep the group delay a whole number of samples, and powers of
    two minus one keep the overlap-save FFT on a friendly size.
    """
    wanted = 3.3 * fs / max(bandwidth * _SKIRT_FRACTION, 1e-9)
    n_taps = N_TAPS_MIN
    while n_taps < wanted and n_taps < N_TAPS_MAX:
        n_taps = 2 * n_taps + 1
    return min(n_taps, N_TAPS_MAX)


def band_error(f_lo: float, bandwidth: float, fs: float) -> Optional[str]:
    """Why this band cannot be filtered, or None if it can.

    Returned as text rather than raised so callers can clamp a band into
    range (see ListenBandViewModel) instead of handling an exception.
    """
    if fs <= 0:
        return "sample rate must be positive"
    if bandwidth < min_bandwidth(fs):
        return ("band narrower than the %.0f Hz roll-off of the longest "
                "filter: nothing but skirt would pass" % min_bandwidth(fs))
    if f_lo < 0:
        return "band starts below 0 Hz"
    if f_lo + bandwidth > fs / 2.0:
        # Not cosmetic: past Nyquist the mix product of a real tone's negative
        # line wraps back *inside* the passband at full gain, which is the
        # mirror image this structure exists to reject.
        return ("band ends above the %.0f Hz Nyquist limit" % (fs / 2.0))
    return None


def _lowpass_taps(cutoff_hz: float, fs: float, n_taps: int) -> np.ndarray:
    """Hamming-windowed sinc lowpass, unity gain at DC, linear phase."""
    k = np.arange(n_taps) - (n_taps - 1) / 2.0
    h = np.sinc(2.0 * cutoff_hz * k / fs) * np.hamming(n_taps)
    return h / h.sum()


class _StreamingFir:
    """Complex FIR by FFT overlap-save, with a cross-faded tap swap.

    process(a) then process(b) gives the same samples as process(a+b), the
    same as scipy's lfilter(taps, 1, x) with zi carried -- to within FFT
    rounding, ~1e-13 relative, which is far below anything audible.

    The transform size is fixed to the longest filter that will ever be
    loaded, so set_taps() never has to rebuild or re-window the history. That
    is what lets the width change mid-stream: the history is input samples,
    not filter state, so both impulse responses are equally valid against it
    and the two outputs can simply be faded between.
    """

    # ~5 ms at 48 kHz. Long enough that no step is audible, short enough to
    # finish inside one audio block, so a drag never stacks fades.
    FADE_SAMPLES = 256

    def __init__(self, taps: np.ndarray, max_taps: int = N_TAPS_MAX):
        size = 1
        while size < max_taps + _BLOCK_HINT - 1:
            size *= 2

        self._size = size
        self._max_block = size - max_taps + 1
        self._history = np.zeros(size, dtype=np.complex128)
        self._spectrum = np.fft.fft(taps, size)
        self._fading_from: Optional[np.ndarray] = None
        self._faded = 0

    def set_taps(self, taps: np.ndarray) -> None:
        """Swap the impulse response, fading rather than switching."""
        if self._fading_from is None:
            self._fading_from = self._spectrum
            self._faded = 0
        # If a fade is already running, keep where it started and just move
        # the destination: the blend still ends on the newest taps. In
        # practice this cannot happen, since the band is read once per audio
        # block and a fade finishes well inside one.
        self._spectrum = np.fft.fft(taps, self._size)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.complex128)

        if x.size > self._max_block:
            # More than the transform can absorb at once: chop it rather than
            # resize, so one long block cannot leave every later block paying
            # for an oversized FFT.
            return np.concatenate([self.process(x[i:i + self._max_block])
                                   for i in range(0, x.size, self._max_block)])

        self._history = np.concatenate((self._history[x.size:], x))
        transformed = np.fft.fft(self._history)
        y = np.fft.ifft(transformed * self._spectrum)[-x.size:]

        if self._fading_from is not None:
            # The expensive half, fft(history), is shared with the line above
            previous = np.fft.ifft(transformed * self._fading_from)[-x.size:]
            n = min(self.FADE_SAMPLES - self._faded, x.size)
            ramp = (self._faded + np.arange(1, n + 1)) / self.FADE_SAMPLES
            y[:n] = previous[:n] * (1.0 - ramp) + y[:n] * ramp
            self._faded += n
            if self._faded >= self.FADE_SAMPLES:
                self._fading_from = None

        return y


@runtime_checkable
class BandFilter(Protocol):
    def configure(self, f_lo: float, bandwidth: float, fs: float) -> None:
        """Select the band from scratch, resetting the stream."""
        ...

    def retune(self, f_lo: float, bandwidth: float) -> None:
        """Move the running band without breaking the audio."""
        ...

    def process(self, block: np.ndarray) -> np.ndarray:
        """Real block in, real block of the same length out."""
        ...


class _DdcBandFilter:
    """Shared down-convert / filter / up-convert machinery.

    Subclasses differ only in where the band is put back: see _up_mix_hz.
    """

    def __init__(self) -> None:
        self._fir: Optional[_StreamingFir] = None
        self.fs = 0.0
        self.n_taps = 0
        self._bandwidth = 0.0

        # The mixers. Two increments each: what the current block starts at,
        # and what it should end at -- see process().
        self._phase = 0.0
        self._dphi = 0.0
        self._dphi_next = 0.0
        self._up_phase = 0.0
        self._up_dphi = 0.0
        self._up_dphi_next = 0.0

    # -- what a subclass decides ------------------------------------------

    def _up_mix_hz(self, centre: float, bandwidth: float) -> float:
        raise NotImplementedError

    @property
    def _up_is_down_conjugate(self) -> bool:
        """True when the band goes back exactly where it came from."""
        return False

    # -- setup and retuning ------------------------------------------------

    @property
    def group_delay_s(self) -> float:
        return group_delay_seconds(self.n_taps, self.fs) if self._fir is not None else 0.0

    def configure(self, f_lo: float, bandwidth: float, fs: float) -> None:
        self._check(f_lo, bandwidth, fs)

        self.fs = fs
        self._fir = _StreamingFir(self._taps_for(bandwidth, fs))
        self._phase = 0.0
        self._up_phase = 0.0
        self._aim(f_lo, bandwidth)
        # nothing is running yet, so start at the target rather than ramp to it
        self._dphi = self._dphi_next
        self._up_dphi = self._up_dphi_next

    def retune(self, f_lo: float, bandwidth: float) -> None:
        """Move the band with the stream intact.

        A position-only move is free and perfectly click-free: of everything
        configure() sets up, only the mixer increments depend on where the
        band is. A width change also swaps the taps, which cross-fades.
        """
        if self._fir is None:
            raise RuntimeError("configure() must be called before retune()")
        self._check(f_lo, bandwidth, self.fs)

        if bandwidth != self._bandwidth:
            self._fir.set_taps(self._taps_for(bandwidth, self.fs))
        self._aim(f_lo, bandwidth)

    def _check(self, f_lo: float, bandwidth: float, fs: float) -> None:
        error = band_error(f_lo, bandwidth, fs)
        if error is not None:
            raise ValueError(error)

    def _taps_for(self, bandwidth: float, fs: float) -> np.ndarray:
        self.n_taps = taps_for_bandwidth(bandwidth, fs)
        return _lowpass_taps(bandwidth / 2.0, fs, self.n_taps).astype(np.complex128)

    def _aim(self, f_lo: float, bandwidth: float) -> None:
        """Point the mixers at the new band. Phases are deliberately untouched."""
        self._bandwidth = bandwidth
        centre = f_lo + bandwidth / 2.0
        self._dphi_next = 2.0 * np.pi * centre / self.fs
        self._up_dphi_next = 2.0 * np.pi * self._up_mix_hz(centre, bandwidth) / self.fs

    # -- the audio path ----------------------------------------------------

    def process(self, block: np.ndarray) -> np.ndarray:
        if self._fir is None:
            raise RuntimeError("configure() must be called before process()")
        s = np.asarray(block, dtype=np.float64).reshape(-1)
        if s.size == 0:
            return np.zeros(0, dtype=np.float64)

        down = np.exp(-1j * self._ramp_phase(s.size, down=True))
        y = self._fir.process(s * down)

        if self._up_is_down_conjugate:
            # Exactly the conjugate, so the mixers cancel and the band lands
            # back where it started with no accumulated phase error.
            up = np.conj(down)
        else:
            up = np.exp(1j * self._ramp_phase(s.size, down=False))

        # 2*Re(): the lowpass kept one of the two lines a real tone splits
        # into, so the other has to be put back.
        return 2.0 * np.real(y * up)

    def _ramp_phase(self, n: int, down: bool) -> np.ndarray:
        """Phase over the block, sliding the frequency to its new value.

        The band is read once per audio block, so a drag would otherwise step
        the mixer 47 times a second and be heard as a staircase. Sliding
        across the block turns that into a glide. The phase itself always
        continues from where the last block left it -- that continuity is the
        whole reason a moving band makes no click.
        """
        if down:
            start, end = self._dphi, self._dphi_next
        else:
            start, end = self._up_dphi, self._up_dphi_next

        increments = np.linspace(start, end, n) if start != end else np.full(n, end)
        phase = (self._phase if down else self._up_phase) + np.cumsum(increments)
        wrapped = float(phase[-1] % (2.0 * np.pi))

        if down:
            self._phase, self._dphi = wrapped, end
        else:
            self._up_phase, self._up_dphi = wrapped, end
        return phase


class BandpassFilter(_DdcBandFilter):
    """Keep [f_lo, f_lo+bandwidth] where it is, mute the rest.

    The band is mixed down to baseband, lowpassed, and mixed straight back up
    by the same phase, so it returns to its own frequency. Gain is 1 across
    the band and 0.5 (-6 dB) at each edge, the usual windowed-sinc shape.

    The output carries a fixed phase offset of 2*pi*f_c*D against a
    tap-modulated band-pass of the same shape (D being the group delay): the
    envelope is delayed, the carrier is not. Inaudible, and it is the price
    of the taps not depending on where the band sits.
    """

    def _up_mix_hz(self, centre: float, bandwidth: float) -> float:
        return centre

    @property
    def _up_is_down_conjugate(self) -> bool:
        return True


class HeterodyneFilter(_DdcBandFilter):
    """Move [f_lo, f_lo+bandwidth] down onto [0, bandwidth].

    Mixed down to baseband and lowpassed as above, then put back up by only
    half the width, which leaves the band's lower edge at 0 Hz.

    Image rejection comes from the lowpass being applied to a *complex*
    signal: a tone below f_lo lands at a negative baseband offset beyond the
    cutoff and is rejected, where a real mix would have folded it onto the
    wanted side and made it indistinguishable. Rejection reaches full depth
    for content more than about half a roll-off outside the band; the edges
    themselves taper over the same width.
    """

    def _up_mix_hz(self, centre: float, bandwidth: float) -> float:
        return bandwidth / 2.0


def make_filter(mode: int) -> BandFilter:
    if mode == HETERODYNE:
        return HeterodyneFilter()
    return BandpassFilter()
