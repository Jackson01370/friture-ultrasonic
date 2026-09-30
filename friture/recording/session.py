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

"""Where a dock finds the recording folder, the recorder and the replay.

Docks are built by the DockManager with nothing but a parent, so this is
the one place the application registers what they need -- the same pattern
as GetListenBand(). Before registration (a dock built in a test, say) every
member is None and callers must cope.
"""

from __future__ import annotations

from pathlib import Path

from friture.recording.store import SegmentStore

# Protecting an event keeps this much either side of it too: the moment
# before and after is usually what makes it make sense.
PROTECT_PAD_S = 30.0


class RecordingSession:

    def __init__(self) -> None:
        self.get_folder = None       # () -> str
        self.recorder = None         # ContinuousRecorder
        self.replay = None           # ReplayController

    @property
    def folder(self) -> Path | None:
        if self.get_folder is None:
            return None
        text = self.get_folder()
        return Path(text) if text else None

    def play_from(self, epoch: float, lead_s: float = 3.0) -> bool:
        """Replay from a little before a moment; enters replay if need be."""
        replay = self.replay
        if replay is None:
            return False
        if not replay.active and not replay.enter():
            return False
        replay.seek(epoch - lead_s)
        if not replay.playing:
            replay.play_pause()
        return True

    def protect(self, start: float, end: float, on: bool) -> list[str]:
        """Protect (or release) every segment that holds [start, end], with padding."""
        folder = self.folder
        if folder is None:
            return []
        store = SegmentStore(folder)
        wavs = [s.wav for s in store.overlapping(start - PROTECT_PAD_S, end + PROTECT_PAD_S)]
        changed = store.set_protected(wavs, on)
        if self.recorder is not None:
            for wav in wavs:
                self.recorder.set_protected(wav, on)
        return changed

    def is_protected(self, start: float, end: float) -> bool:
        """True if every segment holding [start, end] is protected."""
        folder = self.folder
        if folder is None:
            return False
        segs = SegmentStore(folder).overlapping(start, end)
        return bool(segs) and all(s.protected for s in segs)


_session = RecordingSession()


def GetRecordingSession() -> RecordingSession:
    return _session
