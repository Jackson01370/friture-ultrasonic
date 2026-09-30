#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2009 Timothée Lecomte

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

import atexit
import collections
import logging
import math
import threading
import time

from PyQt5 import QtCore
import sounddevice
import rtmixer
from numpy import ndarray, vstack, int8, int16, float64, float32, frombuffer, concatenate
import numpy as np

# This build captures ultrasound. 250 kHz is not a preference: it is the one
# rate the UltraMic 250K will open at, and only in WASAPI exclusive mode with
# a single channel -- see open_stream. Nyquist is therefore 125 kHz.
#
# Everything downstream reads this constant, so an ordinary 48 kHz microphone
# cannot be used with this build. That is the trade for not having to thread a
# runtime rate through 78 call sites; use the released Friture for normal audio.
SAMPLING_RATE = 250000

# Playback is a different rate, and has to be: no sound card takes 250 kHz.
# Whatever is heard is resampled down to this on the way out, which is also
# why heterodyne is the only listening mode that means anything up here --
# no filter makes a 40 kHz sound audible, only moving it does.
OUTPUT_SAMPLING_RATE = 48000

# 2048 frames is 8.2 ms at 250 kHz, about what 512 was at 48 kHz. Kept in that
# range on purpose: the display timer drains whole buffers every 10 ms, so a
# much smaller one just multiplies signal emissions per tick.
FRAMES_PER_BUFFER = 2048

# The same 10.7 ms, counted at the playback rate.
OUTPUT_FRAMES_PER_BUFFER = 512

# ★ THE RING BUFFER IS DRAINED BY ITS OWN THREAD, NOT BY THE DISPLAY TIMER ★
#
# It used to be drained by the display timer on the GUI thread, and that has
# a failure nobody could see: rtmixer's record action STOPS for good the
# moment its ring buffer fills, and says nothing about it. Measured on the
# UltraMic: after 5 s without a fetch, exactly the 2.10 s the ring held was
# delivered and then nothing more, ever -- no overflow flag, input_overflows
# still 0, the stream still "active". (The ring is sized for 3 s but rounded
# DOWN to a power of two, 2**19 frames, which is 2.1 s at 250 kHz.) So any
# stall of the GUI thread longer than two seconds silently ended the capture
# for the rest of the session.
#
# A thread that does nothing but move blocks out of the ring cannot be
# stalled by a slow dock or a busy window. What it reads goes two ways:
#   - to the raw sinks, immediately, on this thread -- the continuous
#     recorder is one, and it must never miss a block whatever the GUI does;
#   - to a queue the display timer empties, which is allowed to lose its
#     oldest blocks if the GUI falls far behind, because a picture that is
#     late is not worth having.
CAPTURE_POLL_S = 0.004
DISPLAY_BACKLOG_S = 10.0
# The action is presumed dead if the stream runs this long with nothing
# arriving; it is then re-issued and the break is reported as a new run.
DEAD_ACTION_S = 0.5

__audiobackendInstance = None

# python-sounddevice (bindings to PortAudio)
# > no device friendly name
# > suffer from PortAudio bugs
# > uses old PortAudio binaries
# > sounddevice provides nice Python bindngs
# > rtmixer provides nice C ringbuffer on top of sounddevice

# rtaudio
# > better maintained than PortAudio
# > no device friendly name
# > no ios/android support
# > no nice Python bindings

# qtmultimedia
# > shipped with Qt5
# > no device friendly name
# > supports iOS and android
# > opaque

# python-soundcard
# > not a lot of devs / users
# > no android support
# > provides device ids and friendly name
# > doc, features are lacking


def AudioBackend():
    global __audiobackendInstance
    if __audiobackendInstance is None:
        __audiobackendInstance = __AudioBackend()
    return __audiobackendInstance


class __AudioBackend(QtCore.QObject):

    underflow = QtCore.pyqtSignal()
    new_data_available = QtCore.pyqtSignal(ndarray, float, bool)

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.logger = logging.getLogger(__name__)

        self.duo_input = False

        self.logger.info("Initializing audio backend")

        # look for devices
        self.input_devices = self.get_input_devices()
        self.output_devices = self.get_output_devices()

        self.logger.info(f"Found {len(self.input_devices)} input devices and {len(self.output_devices)} output devices")

        self.device = None
        self.first_channel = None
        self.second_channel = None

        self.stream = None
        self.ringBuffer = None
        self.action = None
        self.nchannels_max = 0
        self.stream_start_time = 0.0
        self.stream_read_index = 0

        # Everything the capture thread and the GUI thread both touch --
        # stream, ringBuffer, action and the run bookkeeping -- is guarded
        # by this lock. The thread holds it for one drain at a time, well
        # under a millisecond, so a GUI call waits at most that long.
        self._lock = threading.RLock()
        self._display_queue = collections.deque()
        self._display_dropped = 0
        self._raw_sinks = []
        # A RUN is a stretch of samples with no break in it. It starts again
        # whenever continuity cannot be vouched for: a stop and start, a new
        # device, a record action that died and had to be re-issued. Sinks
        # get (run, index within the run) with every block, so a recording
        # can split exactly where a break happened instead of splicing
        # across it.
        self._run_id = 0
        self._run_index = 0
        self.capture_restarts = 0
        self._quiet_since = None
        # Delivery happens only between restart() and pause() -- the span
        # the user sees as "capturing". Outside it the ring is still read,
        # to keep rtmixer's action alive, but what is read is dropped: it
        # belongs to no run anyone asked for.
        self._running = False

        # we will try to open all the input devices until one
        # works, starting by the default input device
        for device in self.input_devices:
            try:
                (self.stream, self.ringBuffer, self.action, self.nchannels_max) = self.open_stream(device)
                self.stream.start()
                self.device = device
                self.logger.info("Success")
                break
            except Exception:
                self.logger.exception("Failed to open stream")

        if self.device is not None:
            self.first_channel = 0
            nchannels = self.get_current_device_nchannels()
            if nchannels == 1:
                self.second_channel = 0
            else:
                self.second_channel = 1

        # counter for the number of input buffer overflows
        self.xruns = 0

        self.chunk_number = 0

        self.devices_with_timing_errors = []

        self._capture_stop = threading.Event()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="friture-capture", daemon=True)
        self._capture_thread.start()
        # sounddevice terminates PortAudio in its own atexit handler, and any
        # program that exits without calling close() would leave this thread
        # polling a terminated library -- measured: a stream of "PortAudio
        # not initialized" tracebacks on the way out. atexit runs handlers
        # last-registered first, and sounddevice registered its handler when
        # it was imported, before this, so this one runs first.
        atexit.register(self._stop_capture_thread)

    def _stop_capture_thread(self):
        self._capture_stop.set()
        if self._capture_thread.is_alive() and threading.current_thread() is not self._capture_thread:
            self._capture_thread.join(timeout=1.0)

    def add_raw_sink(self, sink):
        """Register sink(block, run_id, run_index, t_first, overflow).

        Called ON THE CAPTURE THREAD for every block, before the display
        sees it: block is float32 (frames, channels) exactly as captured,
        run_id / run_index place it in an unbroken run (see __init__),
        t_first estimates the wall-clock time of its first sample, and
        overflow says PortAudio reported lost input just before it. A sink
        must be quick and thread-safe -- hand the block to a queue and
        return.
        """
        with self._lock:
            self._raw_sinks.append(sink)

    def remove_raw_sink(self, sink):
        with self._lock:
            if sink in self._raw_sinks:
                self._raw_sinks.remove(sink)

    def _new_run(self):
        """Declare that continuity is broken from here on. Lock held."""
        self._run_id += 1
        self._run_index = 0

    def _capture_loop(self):
        while not self._capture_stop.is_set():
            try:
                with self._lock:
                    self._drain()
            except Exception:
                self.logger.exception("Capture thread failed to drain the ring buffer")
            time.sleep(CAPTURE_POLL_S)

    def _discard_ring(self):
        """Throw away whatever is in the ring. Lock held."""
        if self.ringBuffer is not None:
            left = self.ringBuffer.read_available
            if left:
                self.ringBuffer.advance_read_index(left)

    def _drain_tail(self):
        """Deliver the last partial block of a run to the raw sinks. Lock held.

        Called by pause() after the stream has stopped, so what is left is
        the end of this run and nothing will follow it. The display has no
        use for less than a block; a recording does -- it is audio.
        """
        if self.ringBuffer is None:
            return
        left = self.ringBuffer.read_available
        if left <= 0:
            return
        read, buf1, buf2 = self.ringBuffer.get_read_buffers(left)
        block = concatenate((frombuffer(buf1, dtype='float32'),
                             frombuffer(buf2, dtype='float32')))
        block.shape = -1, self.nchannels_max
        self.ringBuffer.advance_read_index(read)
        t_first = time.time() - read / SAMPLING_RATE
        for sink in self._raw_sinks:
            try:
                sink(block, self._run_id, self._run_index, t_first, False)
            except Exception:
                self.logger.exception("A raw capture sink failed")
        self._run_index += read

    def _drain(self):
        """Move every whole block out of the ring. Lock held."""
        if self.stream is None or self.ringBuffer is None or self.action is None:
            return
        available = self.ringBuffer.read_available
        if available < FRAMES_PER_BUFFER:
            self._check_action_alive()
            return
        self._quiet_since = None
        if not self._running:
            self._discard_ring()
            return

        now_wall = time.time()
        now_stream = self.get_stream_time()
        input_overflows = self.action.stats.input_overflows
        overflow = input_overflows > self.xruns
        if overflow:
            self.xruns = input_overflows
            self.logger.info("Stream overflow!")

        k = 0
        while self.ringBuffer.read_available >= FRAMES_PER_BUFFER:
            read, buf1, buf2 = self.ringBuffer.get_read_buffers(FRAMES_PER_BUFFER)
            assert read == FRAMES_PER_BUFFER
            block = concatenate((frombuffer(buf1, dtype='float32'),
                                 frombuffer(buf2, dtype='float32')))
            block.shape = -1, self.nchannels_max
            self.ringBuffer.advance_read_index(FRAMES_PER_BUFFER)

            self.stream_read_index += read
            stream_read_time = self.stream_start_time + self.stream_read_index / SAMPLING_RATE
            # when starting a stream, PortAudio hands over some data that was
            # already buffered, so the stream start time is actually older
            if stream_read_time > now_stream and self.stream_read_index < 100000:
                self.stream_start_time -= stream_read_time - now_stream
                stream_read_time = now_stream

            # the first sample of this block sat (backlog behind it) frames
            # before the moment the drain began
            t_first = now_wall - (available - k * FRAMES_PER_BUFFER) / SAMPLING_RATE
            block_overflow = overflow and k == 0
            for sink in self._raw_sinks:
                try:
                    sink(block, self._run_id, self._run_index, t_first, block_overflow)
                except Exception:
                    self.logger.exception("A raw capture sink failed")
            self._run_index += read

            self._display_queue.append((block, stream_read_time, block_overflow))
            k += 1

        limit = int(DISPLAY_BACKLOG_S * SAMPLING_RATE / FRAMES_PER_BUFFER)
        while len(self._display_queue) > limit:
            self._display_queue.popleft()
            self._display_dropped += 1
            if self._display_dropped == 1 or self._display_dropped % 1000 == 0:
                self.logger.warning("Display fell %.0f s behind the capture; %d blocks skipped "
                                    "for display only (recording is unaffected)",
                                    DISPLAY_BACKLOG_S, self._display_dropped)

    def _check_action_alive(self):
        """Re-issue the record action if it has died. Lock held.

        This is the safety net under the thread, not the fix: the thread
        keeps the ring from filling. But if something ever does starve it,
        the difference between a gap and a capture that silently ends for
        the rest of the session is this check.
        """
        if not self.stream.active:
            self._quiet_since = None
            return
        now = time.monotonic()
        if self._quiet_since is None:
            self._quiet_since = now
            return
        if now - self._quiet_since < DEAD_ACTION_S:
            return
        self._quiet_since = None
        if self.action in self.stream.actions:
            return
        self.logger.warning("The record action had stopped (its ring buffer filled); "
                            "re-issuing it. Samples in between are lost and a new run begins.")
        self.action = self.stream.record_ringbuffer(self.ringBuffer)
        self.xruns = 0                       # a new action counts from zero
        self.capture_restarts += 1
        self._new_run()

    def close(self):
        self._stop_capture_thread()
        self.release_stream()

    def release_stream(self):
        """Stop AND close the capture stream, giving the device back.

        Closing matters here in a way it did not before: WASAPI grants
        exclusive access to one stream at a time, and a merely stopped stream
        still holds it. Leave one open and the next exclusive attempt fails
        with "Invalid device" -- from the message alone it looks like the
        microphone is at fault rather than us.
        """
        with self._lock:
            if self.stream is None:
                return
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                self.logger.exception("Failed to release the capture stream")
            self.stream = None
            self._display_queue.clear()
            self._new_run()

    # method
    def get_readable_devices_list(self):
        input_devices = self.get_input_devices()

        raw_devices = sounddevice.query_devices()

        try:
            default_input_device = sounddevice.query_devices(kind='input')
            default_input_device['index'] = raw_devices.index(default_input_device)
        except sounddevice.PortAudioError:
            self.logger.exception("Failed to query the default input device")
            default_input_device = None

        devices_list = []
        for device in input_devices:
            api = sounddevice.query_hostapis(device['hostapi'])['name']

            if default_input_device is not None and device['index'] == default_input_device['index']:
                extra_info = ' (default)'
            else:
                extra_info = ''

            nchannels = device['max_input_channels']

            desc = "%s (%d channels) (%s) %s" % (device['name'], nchannels, api, extra_info)

            devices_list += [desc]

        return devices_list

    # method
    def get_readable_output_devices_list(self):
        output_devices = self.get_output_devices()

        raw_devices = sounddevice.query_devices()
        default_output_device = sounddevice.query_devices(kind='output')
        default_output_device['index'] = raw_devices.index(default_output_device)

        devices_list = []
        for device in output_devices:
            api = sounddevice.query_hostapis(device['hostapi'])['name']

            if default_output_device is not None and device['index'] == default_output_device['index']:
                extra_info = ' (default)'
            else:
                extra_info = ''

            nchannels = device['max_output_channels']

            desc = "%s (%d channels) (%s) %s" % (device['name'], nchannels, api, extra_info)

            devices_list += [desc]

        return devices_list

    # method
    def get_default_input_device(self):
        try:
            index = sounddevice.default.device[0]
        except IOError:
            index = None

        return index

    # method
    def get_default_output_device(self):
        try:
            index = sounddevice.default.device[1]
        except IOError:
            index = None

        return index

    # method
    # returns a list of input devices index
    def get_input_devices(self):
        """The devices that could actually capture at SAMPLING_RATE.

        WASAPI only, and deliberately so. Every microphone on this machine
        also appears under MME and DirectSound, and those entries would open
        happily at 250 kHz and hand back upsampled 48 kHz audio -- so offering
        them is offering six ways to silently record nothing above 24 kHz.
        The same device under WASAPI either works or says why.

        This also drops the "default input device first" ordering: the system
        default is whatever Windows uses for calls, which is never the one
        wanted here, and putting it first only meant a failed attempt before
        reaching a usable entry.
        """
        devices = sounddevice.query_devices()

        input_devices = []
        for index, device in enumerate(devices):
            if device['max_input_channels'] <= 0:
                continue
            host_api = sounddevice.query_hostapis(device['hostapi'])['name']
            if "WASAPI" not in host_api.upper():
                continue
            device['index'] = index
            input_devices += [device]

        if not input_devices:
            self.logger.warning(
                "No WASAPI input device found, so nothing can be captured at "
                "%d Hz. Other host APIs are not offered because they would "
                "resample rather than refuse.", SAMPLING_RATE)

        return input_devices

    # method
    # returns a list of output devices index, starting with the system default
    def get_output_devices(self):
        devices = sounddevice.query_devices()

        default_output_device = sounddevice.query_devices(kind='output')

        output_devices = []
        if default_output_device is not None:
            # start by the default input device
            default_output_device['index'] = devices.index(default_output_device)
            output_devices += [default_output_device]

        for device in devices:
            # select only the output devices by looking at the number of output channels
            if device['max_output_channels'] > 0:
                device['index'] = devices.index(device)
                # default output device has already been inserted
                if default_output_device is not None and device['index'] != default_output_device['index']:
                    output_devices += [device]

        return output_devices

    # method.
    # The index parameter is the index in the self.input_devices list of devices !
    # The return parameter is also an index in the same list.
    def select_input_device(self, index):
        device = self.input_devices[index]

        previous_device = self.device

        self.logger.info("Trying to open input device #%d", index)

        # The old stream goes FIRST, before the new one is attempted. Opening
        # the new one first and keeping the old as a fallback is the friendlier
        # order, and it is what this did -- but WASAPI exclusive mode cannot be
        # granted while any other capture stream is running, so with a stream
        # still open every attempt here fails with "Invalid device". Closing
        # first costs the fallback; there is no way to have both.
        self.release_stream()

        with self._lock:
            try:
                (self.stream, self.ringBuffer, self.action, self.nchannels_max) = self.open_stream(device)
                self.device = device
                self.stream.start()
                self.stream_start_time = self.stream.time
                self.stream_read_index = 0
                self.xruns = 0
                self._new_run()
                success = True
            except Exception:
                self.logger.exception("Failed to open input device")
                success = False
                self.stream = None
                self.device = previous_device

        if success:
            self.logger.info("Success")

            self.first_channel = 0
            nchannels = self.device['max_input_channels']
            if nchannels == 1:
                self.second_channel = 0
            else:
                self.second_channel = 1

        return success, self.input_devices.index(self.device)

    # method
    def select_first_channel(self, index):
        self.first_channel = index
        success = True
        return success, self.first_channel

    # method
    def select_second_channel(self, index):
        self.second_channel = index
        success = True
        return success, self.second_channel

    # method
    def open_stream(self, device):
        self.log_supported_input_formats(device)

        self.logger.info("Opening the stream for device '%s'", device['name'])

        stream, nchannels_max = self.open_recorder(device)

        sampleSize = 4  # the sample size in bytes (float32)
        elementSize = nchannels_max * sampleSize

        # arbitrary size to avoid overflows without using too much memory
        ringbufferSeconds = 3.

        # The number of elements in the buffer (must be a power of 2)
        ringbufferSize = 2**int(math.log2(ringbufferSeconds * SAMPLING_RATE))

        ringBuffer = rtmixer.RingBuffer(elementSize, ringbufferSize)

        # action can be used to read the count of input overflows
        action = stream.record_ringbuffer(ringBuffer)

        lat_ms = 1000 * stream.latency
        self.logger.info("Device claims %d ms latency", lat_ms)

        return (stream, ringBuffer, action, nchannels_max)

    def open_recorder(self, device):
        """Open the capture stream, insisting on a mode that is really 250 kHz.

        WASAPI exclusive mode, and a single channel, is the only combination
        the UltraMic 250K actually runs at this rate. It matters that this
        does not quietly settle for less: ask MME or DirectSound for 250 kHz
        and they say yes, hand back samples at roughly the right count, and
        fill them with 48 kHz audio stretched to fit -- a spectrogram with a
        hard edge at 24 kHz and nothing above it, looking for all the world
        like a quiet night.

        So the exclusive attempts come first, and a shared-mode stream is
        opened only as a last resort and shouted about.
        """
        host_api = sounddevice.query_hostapis(device['hostapi'])['name']
        if "WASAPI" not in host_api.upper():
            # Not a fallback worth having. MME and DirectSound accept 250 kHz,
            # return roughly the right number of samples, and fill them with
            # 48 kHz audio stretched to fit: a spectrogram with a hard edge at
            # 24 kHz and nothing above it, which reads as a quiet night rather
            # than as a broken capture. Refusing is the honest answer.
            raise RuntimeError(
                "'%s' is on %s, which cannot capture at %d Hz -- it would "
                "resample and the ultrasound would be silently lost. Use the "
                "WASAPI entry for this microphone."
                % (device['name'], host_api, SAMPLING_RATE))

        exclusive = sounddevice.WasapiSettings(exclusive=True)
        failures = []
        # Mono first: the shared-mode mix format may claim two channels while
        # the hardware only offers one at its native rate.
        for channels in (1, device['max_input_channels']):
            if channels < 1 or any(channels == tried for tried, _ in failures):
                continue
            try:
                stream = rtmixer.Recorder(
                    device=device['index'],
                    channels=channels,
                    blocksize=FRAMES_PER_BUFFER,
                    samplerate=SAMPLING_RATE,
                    extra_settings=exclusive)
            except Exception as exception:
                failures.append((channels, exception))
                continue

            self.logger.info("Opened '%s' at %d Hz, WASAPI exclusive, %d channel(s)",
                             device['name'], SAMPLING_RATE, channels)
            return stream, channels

        raise RuntimeError(
            "Could not open '%s' in WASAPI exclusive mode at %d Hz.\n  %s"
            % (device['name'], SAMPLING_RATE,
               "\n  ".join("%d channel(s): %s" % f for f in failures)))

    def log_supported_input_formats(self, device):
        samplerates = [22050, 44100, 48000, 96000, 192000, 250000, 384000]
        dtypes = [float32, int16, int8]
        supported_formats = []
        for samplerate in samplerates:
            for dtype in dtypes:
                try:
                    sounddevice.check_input_settings(
                        device=device['index'],
                        channels=device['max_input_channels'],
                        dtype=dtype,
                        extra_settings=None,
                        samplerate=samplerate)
                    supported_formats += [f"{samplerate} Hz, {np.dtype(dtype).name}"]
                except Exception:
                    pass # check_input_settings throws when the format is not supported

        api = sounddevice.query_hostapis(device['hostapi'])['name']
        self.logger.info(f"Supported formats for '{device['name']}' on '{api}': {supported_formats}")

    # method
    def open_output_stream(self, device, callback):
        # OUTPUT_SAMPLING_RATE, not SAMPLING_RATE: no sound card takes the
        # capture rate. Callers are responsible for arriving here at 48 kHz.
        stream = sounddevice.OutputStream(
            samplerate=OUTPUT_SAMPLING_RATE,
            blocksize=OUTPUT_FRAMES_PER_BUFFER,
            device=device['index'],
            channels=device['max_output_channels'],
            dtype=int16,
            callback=callback)

        return stream

    def is_output_format_supported(self, device, output_format):
        # raise sounddevice.PortAudioError if the format is not supported
        # the exception message contains the details, such as an invalid sample rate
        sounddevice.check_output_settings(
            device=device['index'],
            channels=device['max_output_channels'],
            dtype=output_format,
            samplerate=OUTPUT_SAMPLING_RATE)

    # method
    # return the index of the current input device in the input devices list
    # (not the same as the PortAudio index, since the latter is the index
    # in the list of *all* devices, not only input ones)
    def get_readable_current_device(self):
        return self.input_devices.index(self.device)

    # method
    def get_readable_current_channels(self):
        nchannels = self.device['max_input_channels']

        if nchannels == 2:
            channels = ['L', 'R']
        else:
            channels = []
            for channel in range(0, nchannels):
                channels += ["%d" % channel]

        return channels

    # method
    def get_current_first_channel(self):
        return self.first_channel

    # method
    def get_current_second_channel(self):
        return self.second_channel

    # method
    def get_current_device_nchannels(self):
        return self.device['max_input_channels']

    def get_device_outputchannels_count(self, device):
        return device['max_output_channels']

    def fetchAudioData(self):
        """Hand the blocks the capture thread has queued to the display.

        Runs on the GUI thread, from the display timer. It no longer touches
        the ring buffer -- see CAPTURE_POLL_S -- so however late it runs,
        the capture carries on and only the display falls behind.
        """
        while True:
            with self._lock:
                if not self._display_queue:
                    return
                raw, stream_read_time, input_overflow = self._display_queue.popleft()
            buffer = raw.astype(float64)

            channel = self.get_current_first_channel()
            if self.duo_input:
                channel_2 = self.get_current_second_channel()

            floatdata1 = buffer[:, channel]

            if self.duo_input:
                floatdata2 = buffer[:, channel_2]
                floatdata = vstack((floatdata1, floatdata2))
            else:
                floatdata = floatdata1.reshape(1, -1)

            if input_overflow:
                self.underflow.emit()

            self.new_data_available.emit(floatdata, stream_read_time, input_overflow)

            self.chunk_number += 1

    def set_single_input(self):
        self.duo_input = False

    def set_duo_input(self):
        self.duo_input = True

    def get_stream_time(self) -> float:
        """The current stream time in seconds.

        The time values are monotonically increasing and have
        unspecified origin.

        This provides valid time values for the entire life of the
        stream, from when the stream is opened until it is closed.
        Starting and stopping the stream does not affect the passage of
        time as provided here.

        This time may be used for synchronizing other events to the
        audio stream.
        """

        if self.stream is None:
            return 0

        try:
            return self.stream.time
        except (sounddevice.PortAudioError, OSError):
            if self.stream.device not in self.devices_with_timing_errors:
                self.devices_with_timing_errors.append(self.stream.device)
                self.logger.exception("Failed to read stream time")
            return 0

    def pause(self):
        """Stop capturing. The blocks already captured stay in this run.

        They used not to: the run was renumbered first and the capture
        thread drained what was left in the ring afterwards, so the last
        blocks before a stop came out as a separate run. Measured on a
        32-minute recording: a stop left a 16 ms file of its own, recorded
        as following "a 0.022 s gap" that never happened -- those two
        blocks were the direct continuation of the file before. stop()
        returns once PortAudio has handed over its last buffer, so draining
        after it and only then starting a new run keeps them where they
        belong.
        """
        with self._lock:
            if self.stream is not None:
                self.stream.stop()
                if self._running:
                    self._drain()
                    self._drain_tail()
            self._running = False
            self._display_queue.clear()
            self._new_run()

    def restart(self):
        """Start capturing. Whatever sat in the ring from before is stale."""
        with self._lock:
            if self.stream is not None:
                self._discard_ring()
                self.stream.start()
                self.stream_start_time = self.stream.time
                self.stream_read_index = 0
            self._display_queue.clear()
            self._new_run()
            self._running = True
