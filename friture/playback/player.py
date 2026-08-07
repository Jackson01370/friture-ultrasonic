# -*- coding: utf-8 -*-

# Copyright (C) 2024 Celeste Sinéad

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

from enum import Enum
import logging
from typing import Any, Dict, Optional

from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np
from sounddevice import OutputStream

from friture.audiobackend import (
    AudioBackend,
    FRAMES_PER_BUFFER,
    OUTPUT_FRAMES_PER_BUFFER,
    SAMPLING_RATE,
)
from friture.listen.playout import Playout
from friture.listen.processor import BandProcessor
from friture.ringbuffer import RingBuffer

log = logging.getLogger(__name__)

DEFAULT_HISTORY_LENGTH_S = 30

# Room for a few output blocks. This queue is pulled, not pushed -- the
# callback fills it on demand -- so it only has to absorb the ragged number of
# samples a fractional resampler returns per chunk.
PLAYOUT_CAPACITY = 8 * OUTPUT_FRAMES_PER_BUFFER

class PlayState(Enum):
    STOPPED = 0
    PLAYING = 1
    STOPPING = 2

class Player(QObject):
    stopping = pyqtSignal()
    stopped = pyqtSignal()
    recorded_length_changed = pyqtSignal(float)
    playback_time_changed = pyqtSignal(float)

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.history_sec = DEFAULT_HISTORY_LENGTH_S
        self.history_samples = self.history_sec * SAMPLING_RATE
        self.buffer = RingBuffer()
        self.buffer.grow_if_needed(self.history_samples)
        self.recorded_len = 0
        self.stopping.connect(self.on_stopping)

        # If zero, play starts at beginning of buffer. Otherwise, this
        # is a negative offset in seconds from the end of the buffer.
        self.play_start_time = 0.0

        self.device: Optional[Dict[str, Any]] = None
        self.stream: Optional[OutputStream] = None
        self.play_offset = 0
        self.state = PlayState.STOPPED

        # When band listening is on, playback is band-limited too, so what
        # you hear from the history matches what you heard live.
        self.band_processor: Optional[BandProcessor] = None

        # The history is at the capture rate and the sound card is not, so
        # nothing reaches the speakers without passing through here.
        self.playout = Playout(PLAYOUT_CAPACITY)
        self._block = np.zeros(OUTPUT_FRAMES_PER_BUFFER, dtype=np.float32)

    def set_band_processor(self, band_processor: BandProcessor) -> None:
        self.band_processor = band_processor

    def set_history_seconds(self, new_len: int) -> None:
        if new_len <= 0:
            raise ValueError("History must have positive length")

        self.history_sec = new_len
        self.history_samples = self.history_sec * SAMPLING_RATE
        self.buffer.grow_if_needed(self.history_samples)

        # Handle the case where the current play position is truncated out
        # (will result in a skip in playback)
        if self.state == PlayState.PLAYING:
            if self.play_offset < (self.buffer.offset - self.history_samples):
                self.play_offset = self.buffer.offset - self.history_samples
                self.playback_time_changed.emit(
                    (self.play_offset - self.buffer.offset) / SAMPLING_RATE)

        # Handle the case where the selected start time is truncated
        if self.play_start_time < -self.history_sec:
            self.play_start_time = -self.history_sec
            # Start time == playback time only when not playing
            if self.state != PlayState.PLAYING:
                self.playback_time_changed.emit(-self.history_sec)

        # Note that this sets the valid range for the slider, and the current
        # position will have been adjusted to fit, above.
        if self.history_samples < self.recorded_len:
            self.recorded_len = self.history_samples
            self.recorded_length_changed.emit(self.recorded_len / SAMPLING_RATE)

    def handle_new_data(self, data: np.ndarray) -> None:
        # this will zero out history if channel count changes, not ideal but
        # probably doesn't matter
        self.buffer.push(data, 0)
        new_len = min(
            self.recorded_len + data.shape[1], self.history_samples)
        if new_len != self.recorded_len:
            self.recorded_len = new_len
            self.recorded_length_changed.emit(self.recorded_len / SAMPLING_RATE)

    def play(self) -> None:
        if self.state != PlayState.STOPPED:
            log.info("Already playing!")
            return
        self.state = PlayState.PLAYING

        # A filter's tail belongs to one continuous stream; carrying the last
        # playback's tail into this one would bleed it into the first
        # milliseconds here. The resampler holds a tail for the same reason.
        if self.band_processor is not None:
            self.band_processor.reset()
        self.playout.reset()

        log.info(f"Playing back {self.recorded_len} samples")
        if self.stream is None:
            for device in AudioBackend().output_devices:
                log.info(f"Opening stream for {device['name']}")
                try:
                    self.device = device
                    self.stream = AudioBackend().open_output_stream(
                        device, self.output_callback)
                    self.stream.start()
                except Exception:
                    log.exception("Failed to open stream")
                else:
                    break

        if self.play_start_time == 0.0:
            self.play_offset = self.buffer.offset - self.recorded_len
        else:
            start_offset = max(
                -self.recorded_len,
                int(self.play_start_time * SAMPLING_RATE)
            )
            self.play_offset = self.buffer.offset + start_offset

    def stop(self) -> None:
        if self.state == PlayState.PLAYING:
            self.on_stopping()

    def is_stopped(self) -> bool:
        return self.state == PlayState.STOPPED

    def output_callback(
        self,
        out_data: np.ndarray,
        samples: int,
        time_info: float,
        status: str
    ) -> None:
        if status:
            log.info(status)

        out_channels = AudioBackend().get_device_outputchannels_count(self.device)

        # Pull from the history until there is enough playback-rate audio for
        # this block. The two rates differ by 125/24, so a fixed number of
        # history samples per callback would not divide evenly -- the queue
        # inside Playout is what absorbs that.
        ran_out = False
        while self.playout.available < samples:
            remaining = self.buffer.offset - self.play_offset
            if remaining <= 0:
                ran_out = True
                break
            chunk = int(min(remaining, FRAMES_PER_BUFFER))
            # data_indexed returns the samples ENDING at the index it is given
            source = self.buffer.data_indexed(self.play_offset + chunk, chunk)
            mono = source[0] if source.shape[0] == 1 else source.mean(axis=0)
            if self.band_processor is not None and self.band_processor.active:
                # Band listening collapses to mono: the band is one thing, and
                # the first channel is the one the plots analyse, so it is the
                # one whose peaks were clicked.
                mono = self.band_processor.process(mono)
            self.playout.push(mono)
            self.play_offset += chunk

        self.playback_time_changed.emit(
            (self.play_offset - self.buffer.offset) / SAMPLING_RATE)

        if ran_out and self.playout.available < samples and self.state == PlayState.PLAYING:
            log.info("Reached end of playback")
            self.state = PlayState.STOPPING
            self.stopping.emit()

        if self._block.shape[0] != samples:
            self._block = np.zeros(samples, dtype=np.float32)
        self.playout.pop_into(self._block)

        # Buffer is float in [-1, 1], output is int16, need to convert scales
        int16info = np.iinfo(np.int16)
        scale = min(abs(int16info.min), int16info.max)
        mono_out = (scale * np.clip(self._block, -1.0, 1.0)).astype(np.int16)

        # out_data is time-major (frames, channels): one mono stream everywhere
        out_data[:] = mono_out[:, np.newaxis]


    def on_stopping(self) -> None:
        assert self.stream is not None
        self.stream.stop()
        self.stream = None
        self.state = PlayState.STOPPED
        self.stopped.emit()
