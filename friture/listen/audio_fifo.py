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

"""Bounded FIFO between the GUI thread and the audio output callback.

Adapted from ultraScan's SpscAudioRing (its DESIGN §3), with the blocking
producer removed: there the producer is a worker thread that can afford to
wait, here it is the GUI thread and waiting would freeze the window.

So neither side ever blocks. Overflow drops the newest samples, underflow
zero-fills, and both are counted -- an output callback that blocks *is* the
underrun, and a producer that blocks is a frozen UI.

Priming is kept from ultraScan: the consumer is fed silence until `prebuffer`
samples are queued, and drops back to priming after an underrun instead of
starving every subsequent callback by the same one sample forever. The gap
that inserts is an audible click, deliberately: keep the stream alive rather
than chase the last sample.
"""

from __future__ import annotations

import threading

import numpy as np


class AudioFifo:
    """Strict-FIFO float32 queue with priming and underrun accounting.

    Counters are plain ints written under the lock. Reading them without it
    (for a status display) is safe under the GIL and at worst one tick stale.
    """

    def __init__(self, capacity: int, prebuffer: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not 0 <= prebuffer <= capacity:
            raise ValueError("prebuffer must be in [0, capacity]")

        self._buf = np.zeros(int(capacity), dtype=np.float32)
        self._cap = int(capacity)
        self._prebuffer = int(prebuffer)
        self._read = 0    # absolute samples consumed
        self._write = 0   # absolute samples produced
        self._primed = False
        self._lock = threading.Lock()

        self.n_dropped = 0       # samples the producer could not fit
        self.n_underruns = 0     # primed-but-short pop events

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def occupancy(self) -> int:
        with self._lock:
            return self._write - self._read

    def clear(self) -> None:
        """Drop everything queued and go back to priming."""
        with self._lock:
            self._read = self._write
            self._primed = False

    # -- producer side (GUI thread) -----------------------------------------
    def push(self, samples: np.ndarray) -> int:
        """Append what fits, drop the rest. Returns the number appended."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0

        with self._lock:
            free = self._cap - (self._write - self._read)
            n = min(free, x.size)
            if n < x.size:
                self.n_dropped += x.size - n
            if n == 0:
                return 0

            w = self._write % self._cap
            end = w + n
            if end <= self._cap:
                self._buf[w:end] = x[:n]
            else:
                first = self._cap - w
                self._buf[w:] = x[:first]
                self._buf[:n - first] = x[first:n]
            self._write += n
        return n

    # -- consumer side (output callback) ------------------------------------
    def pop_into(self, out: np.ndarray) -> int:
        """Fill `out` from the queue, zero-filling any shortfall.

        Never blocks -- a copy and a few counter bumps, nothing else, because
        this runs inside the PortAudio callback. Returns the number of real
        (not zero-filled) samples written.
        """
        n = out.shape[0]
        if n == 0:
            return 0

        with self._lock:
            occupancy = self._write - self._read
            if not self._primed:
                if occupancy < self._prebuffer:
                    out[:] = 0.0
                    return 0
                self._primed = True

            take = min(occupancy, n)
            r = self._read % self._cap
            end = r + take
            if end <= self._cap:
                out[:take] = self._buf[r:end]
            else:
                first = self._cap - r
                out[:first] = self._buf[r:]
                out[first:take] = self._buf[:take - first]
            self._read += take

            if take < n:
                out[take:] = 0.0
                self.n_underruns += 1
                self._primed = False  # rebuild headroom rather than starve
        return take
