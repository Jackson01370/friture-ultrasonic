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

"""Replay mode: the whole application looking at a recording instead of the room.

Entering it points AudioBackend's display at a ReplaySource; leaving it
points it back at the microphone. Nothing else in the application changes
-- which is the point: the docks cannot tell a replay from the room, so
everything they do live they do to the recording.

The microphone is NOT stopped. The continuous recorder takes its blocks
from the capture thread, so the room goes on being recorded while an older
recording is being looked at.

Every seek, and every gap the replay jumps over, restarts the docks: a
Band Survey average or a decoder clock carried across a jump would mix two
different moments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PyQt5 import QtCore

from friture.audiobackend import FRAMES_PER_BUFFER, AudioBackend
from friture.recording.replay import RecordingTimeline, ReplaySource

log = logging.getLogger(__name__)

# Where a replay starts when it is opened: this far before the latest audio,
# because what is usually wanted is "what just happened".
OPEN_BEFORE_END_S = 60.0
# Near the end of a file that is still being written, the replay is
# following the recording as it happens.
FOLLOWING_S = 5.0
# Measured in the application with a spectrum, a spectrogram, a Band Survey
# and a Digital Decode dock open: 2x reached 2.00x, 4x 3.77x, 8x 3.89x. The
# docks are the limit, not the disk, so 8x is not offered.
SPEEDS = (1.0, 2.0, 4.0)


def _clock_text(epoch: float, with_date: bool = True) -> str:
    t = datetime.fromtimestamp(epoch)
    s = t.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")
    return s + ".%d" % int((epoch % 1) * 10)


class ReplayController(QtCore.QObject):

    mode_changed = QtCore.pyqtSignal(bool)      # True when replaying
    playing_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent, view_model, get_folder, restart_docks) -> None:
        super().__init__(parent)
        self.vm = view_model
        self._get_folder = get_folder
        self._restart_docks = restart_docks
        self.source: ReplaySource | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._update_view)

        vm = view_model
        vm.toggle_requested.connect(self.toggle)
        vm.play_pause_requested.connect(self.play_pause)
        vm.seek_requested.connect(self.seek_fraction)
        vm.step_requested.connect(self.step)
        vm.speed_requested.connect(self.set_speed)
        vm.latest_requested.connect(self.latest)
        AudioBackend().display_discontinuity.connect(self._on_jump)

    @property
    def active(self) -> bool:
        return self.source is not None

    @property
    def playing(self) -> bool:
        return self.source is not None and self.source.playing

    # -- entering and leaving -------------------------------------------------------

    def toggle(self) -> None:
        if self.active:
            self.leave()
        else:
            self.enter()

    def enter(self) -> bool:
        folder = Path(self._get_folder())
        self.vm.folder_text = str(folder)
        timeline = RecordingTimeline(folder)
        if timeline.empty:
            self.vm.status_text = "No recordings in %s yet." % folder
            self.vm.active = True               # show the bar, with the reason
            return False
        self.source = ReplaySource(timeline, FRAMES_PER_BUFFER)
        start, end = timeline.span
        self.source.seek(max(start, end - OPEN_BEFORE_END_S))
        self.source.play()
        AudioBackend().set_display_source(self.source)
        self._restart_docks()
        self.vm.active = True
        self.vm.speed = self.source.speed
        self._timer.start()
        self._update_view()
        self.mode_changed.emit(True)
        self.playing_changed.emit(True)
        log.info("Replay of %s from %s", folder, _clock_text(self.source.position))
        return True

    def leave(self) -> None:
        if self.source is not None:
            AudioBackend().set_display_source(None)
            self.source.close()
            self.source = None
            self._restart_docks()
        self._timer.stop()
        self.vm.active = False
        self.vm.playing = False
        self.vm.status_text = ""
        self.mode_changed.emit(False)

    # -- controls --------------------------------------------------------------------

    def play_pause(self) -> None:
        if self.source is None:
            return
        if self.source.playing:
            self.source.pause()
        else:
            if self.source.at_end:
                self.latest()
            self.source.play()
        self._update_view()
        self.playing_changed.emit(self.source.playing)

    def seek(self, epoch: float) -> None:
        if self.source is None:
            return
        self.source.seek(epoch)
        self._restart_docks()
        self._update_view()

    def seek_fraction(self, fraction: float) -> None:
        if self.source is None:
            return
        start, end = self.source.timeline.span
        self.seek(start + max(0.0, min(1.0, fraction)) * (end - start))

    def step(self, seconds: float) -> None:
        if self.source is not None:
            self.seek(self.source.position + seconds)

    def set_speed(self, speed: float) -> None:
        if self.source is not None:
            self.source.set_speed(speed)
            self.vm.speed = self.source.speed

    def latest(self) -> None:
        if self.source is not None:
            self.source.timeline.refresh()
            _, end = self.source.timeline.span
            self.seek(end - FOLLOWING_S)
            self.source.play()

    def _on_jump(self) -> None:
        # a gap in the recording was jumped during playback
        self._restart_docks()

    # -- the bar ---------------------------------------------------------------------

    def _update_view(self) -> None:
        src = self.source
        vm = self.vm
        if src is None or src.timeline.empty:
            return
        start, end = src.timeline.span
        span = max(end - start, 1e-9)
        pos = src.position
        vm.playing = src.playing
        vm.position_text = _clock_text(pos)
        vm.start_text = _clock_text(start)
        vm.end_text = _clock_text(end)
        vm.fraction = (pos - start) / span
        vm.set_ranges([((a - start) / span, (b - start) / span) for a, b in src.timeline.ranges()])
        last = src.timeline.entries[-1]
        if src.at_end:
            status = "End of the recordings."
        elif not src.playing:
            status = "Paused."
        elif last.growing and end - pos < FOLLOWING_S + 2.0:
            status = "Following the recording as it is written."
        else:
            status = "Replaying %s." % src.current_wav
        achieved = src.achieved_speed
        if src.playing and src.speed > 1.0 and achieved < 0.9 * src.speed:
            status += "  This machine is keeping up at %.1fx, not %.0fx." % (achieved, src.speed)
        vm.status_text = status
