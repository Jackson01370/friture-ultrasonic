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

"""Plays the selected band through the speakers, live.

Capture runs at 250 kHz and playback at 48 kHz, so everything here goes
through Playout on its way out. Above ~24 kHz that resampling is a wall:
band-pass mode simply cannot deliver an ultrasonic band to a speaker, and
heterodyne -- which moves the band down to baseband before this stage sees
it -- is the mode that makes the ultrasound audible.

Feedback warning: this puts the microphone back out of the speakers. Use
headphones -- a band-limited howl is still a howl, and the band-pass mode
happily sustains one inside its own passband.

Clock drift: the input and output devices run off separate clocks that are
only nominally both 48 kHz. Nothing here resamples, so over long runs the
FIFO drifts one way and either drops samples or underruns, roughly once per
few minutes for a typical few-ppm mismatch. Both are counted rather than
hidden; the click that comes with an underrun is the honest signal that the
stream re-primed.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from PyQt5.QtCore import QObject
from sounddevice import OutputStream

from friture.audiobackend import AudioBackend, OUTPUT_FRAMES_PER_BUFFER
from friture.listen.listen_band_view_model import ListenBandViewModel
from friture.listen.playout import Playout
from friture.listen.processor import BandProcessor

logger = logging.getLogger(__name__)

# Enough headroom to ride out a late GUI-thread tick, little enough that the
# band still responds to a click promptly: ~43 ms primed, ~170 ms worst case.
PREBUFFER_SAMPLES = 2 * OUTPUT_FRAMES_PER_BUFFER
CAPACITY_SAMPLES = 8 * OUTPUT_FRAMES_PER_BUFFER

_INT16_SCALE = 32767.0


class LiveMonitor(QObject):

    def __init__(self, parent: Optional[QObject], band: ListenBandViewModel) -> None:
        super().__init__(parent)

        self._band = band
        self._processor = BandProcessor(self, band)
        self._playout = Playout(CAPACITY_SAMPLES, PREBUFFER_SAMPLES)
        self._device: Optional[Dict[str, Any]] = None
        self._stream: Optional[OutputStream] = None
        self._scratch = np.zeros(OUTPUT_FRAMES_PER_BUFFER, dtype=np.float32)

        band.monitoring_changed.connect(self.set_running)

    def set_running(self, running: bool) -> None:
        if running:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        if self._stream is not None:
            return

        self._playout.reset()
        self._processor.reset()

        for device in AudioBackend().output_devices:
            try:
                self._device = device
                self._stream = AudioBackend().open_output_stream(device, self._output_callback)
                self._stream.start()
            except Exception:
                logger.exception("Failed to open monitor output stream for '%s'", device['name'])
                self._stream = None
            else:
                logger.info("Band monitor playing on '%s'", device['name'])
                return

        self._device = None
        # No device took it, so the switch is lying -- put it back.
        self._band.set_monitoring(False)

    def _stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception:
            logger.exception("Failed to close the monitor output stream")
        if self._playout.n_underruns or self._playout.n_dropped:
            logger.info(
                "Band monitor stopped after %d underruns and %d dropped samples",
                self._playout.n_underruns, self._playout.n_dropped)

    # -- producer: the GUI thread, on each block of captured audio ----------
    def handle_new_data(self, floatdata: np.ndarray) -> None:
        if self._stream is None or not self._band.enabled:
            return
        # First channel, matching what the spectrogram analyses -- the band
        # you hear has to be the band you clicked on. Band-limited at the
        # capture rate, then brought down to the playback rate.
        self._playout.push(self._processor.process(floatdata[0, :]))

    # -- consumer: the PortAudio callback thread ---------------------------
    def _output_callback(self, out_data: np.ndarray, samples: int, time_info: Any, status: Any) -> None:
        if status:
            logger.info("Monitor output status: %s", status)

        if self._scratch.shape[0] != samples:
            self._scratch = np.zeros(samples, dtype=np.float32)
        self._playout.pop_into(self._scratch)

        mono = (_INT16_SCALE * np.clip(self._scratch, -1.0, 1.0)).astype(np.int16)
        # out_data is (frames, channels): the same mono band in every channel.
        out_data[:] = mono[:, np.newaxis]
