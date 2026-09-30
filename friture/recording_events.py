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

"""The Recording Events dock: when something happened, and a way back to it.

Reads the analysis logs beside the continuous recordings (see
friture.recording.events for what counts as an event and why), lists the
events newest first, replays any of them with a click, and keeps the files
around an event from being deleted when the folder fills.

It takes no audio: the list is about what was recorded, not what is heard
now, so it works the same live and in replay.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject

from friture.recording.event_index import EventIndex
from friture.recording.session import GetRecordingSession
from friture.recording.store import BYTES_PER_GB, SegmentStore
from friture.recording_events_view_model import RecordingEventsViewModel

log = logging.getLogger(__name__)

# How often the list is recomputed: the newest log grows by a record a second.
REFRESH_S = 20.0
KIND_LABELS = {"loud": "Loud", "voice": "Voice-like", "line_on": "Line appeared", "line_off": "Line went away"}
FILTERS = ("show_loud", "show_voice", "show_brief", "show_lines", "protected_only")


def _key(e) -> str:
    return "%s|%.3f|%s" % (e.kind, e.start, "%.1f" % e.freq_hz if e.freq_hz else "")


class RecordingEvents_Widget(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audiobuffer = None
        self._vm = RecordingEventsViewModel(self)
        self._index = EventIndex()
        self._events = []
        self._by_key = {}
        self._applied_at = 0.0
        self._next_refresh = 0.0
        vm = self._vm
        vm.play_requested.connect(self._play)
        vm.protect_requested.connect(self._protect)
        vm.refresh_requested.connect(lambda: self._request(force=True))
        for name in FILTERS:
            getattr(vm, name + "_changed").connect(lambda *_: self._apply())
        vm.status_text = "Reading the analysis logs ..."
        self.settings_dialog = RecordingEvents_Settings_Dialog(parent)

    # -- the dock interface --------------------------------------------------------

    def view_model(self):
        return self._vm

    def qml_file_name(self):
        return "RecordingEvents.qml"

    def set_buffer(self, buffer):
        self.audiobuffer = buffer

    def handle_new_data(self, floatdata):
        pass

    def pause(self):
        pass

    def restart(self):
        pass

    def settings_called(self, checked):
        self.settings_dialog.show()

    def canvasUpdate(self):
        now = time.monotonic()
        if now >= self._next_refresh:
            self._request()
        events, when, took, folder = self._index.result()
        if when > self._applied_at:
            self._applied_at = when
            self._events = events
            self._took = took
            self._apply()

    def saveState(self, settings):
        for name in FILTERS:
            settings.setValue(name, getattr(self._vm, name))

    def restoreState(self, settings):
        for name in FILTERS:
            default = name != "protected_only"
            setattr(self._vm, name, settings.value(name, default, type=bool))

    # -- work ----------------------------------------------------------------------------

    def _request(self, force: bool = False) -> None:
        folder = GetRecordingSession().folder
        self._next_refresh = time.monotonic() + REFRESH_S
        if folder is None:
            self._vm.status_text = "No recording folder is set (Settings, Continuous recording)."
            return
        if not folder.is_dir():
            self._vm.status_text = "No recordings in %s yet." % folder
            return
        self._index.refresh(folder)

    def _apply(self) -> None:
        vm = self._vm
        folder = GetRecordingSession().folder
        segs = SegmentStore(folder).segments() if folder is not None and folder.is_dir() else []
        spans = [(s.start_epoch,
                  s.end_epoch if s.end_epoch else s.start_epoch + s.n_frames / max(s.fs, 1),
                  s.protected) for s in segs]

        def kept(e) -> bool:
            over = [p for a, b, p in spans if a <= e.end and b >= e.start]
            return bool(over) and all(over)

        rows = []
        self._by_key = {}
        for e in sorted(self._events, key=lambda e: e.start, reverse=True):
            is_kept = kept(e)
            if e.kind == "loud" and not vm.show_loud:
                continue
            if e.kind == "voice" and (not vm.show_voice or (e.brief and not vm.show_brief)):
                continue
            if e.kind.startswith("line") and not vm.show_lines:
                continue
            if vm.protected_only and not is_kept:
                continue
            key = _key(e)
            self._by_key[key] = e
            label = KIND_LABELS.get(e.kind, e.kind) + (" (brief)" if e.brief else "")
            rows.append({"key": key, "start": e.start,
                         "time_text": datetime.fromtimestamp(e.start).strftime("%m-%d %H:%M:%S"),
                         "kind": e.kind, "kind_label": label, "detail": e.detail,
                         "brief": bool(e.brief), "protected": is_kept})
        vm.events.set_rows(rows)

        if segs:
            first = datetime.fromtimestamp(segs[0].start_epoch).strftime("%m-%d %H:%M")
            last = datetime.fromtimestamp(spans[-1][1]).strftime("%m-%d %H:%M")
            hours = sum(s.n_frames / max(s.fs, 1) for s in segs) / 3600.0
            vm.summary_text = "%d event(s) in %.1f h recorded, %s to %s" % (len(self._events), hours, first, last)
            n, size = SegmentStore(folder).protected_bytes()
            vm.protected_text = ("Kept: %d file(s), %.2f GB -- not deleted when space is needed" % (n, size / BYTES_PER_GB)
                                 if n else "Nothing is kept yet; press \"keep\" on an event to protect its files.")
        if not rows:
            vm.status_text = ("No events match the filters." if self._events
                              else "Nothing unusual in the recordings so far.")

    def _play(self, key: str) -> None:
        e = self._by_key.get(key)
        if e is not None:
            GetRecordingSession().play_from(e.start)

    def _protect(self, key: str) -> None:
        e = self._by_key.get(key)
        if e is None:
            return
        session = GetRecordingSession()
        on = not session.is_protected(e.start, e.end)
        changed = session.protect(e.start, e.end, on)
        log.info("%s %d file(s) around %s", "Kept" if on else "Released", len(changed), e.detail)
        self._apply()


class RecordingEvents_Settings_Dialog(QtWidgets.QDialog):
    """Nothing to set here: the folder belongs to the recorder's settings."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Recording events")
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "The events are read from the continuous recordings. The folder and how much\n"
            "is kept are set in Settings, under \"Continuous recording\".", self)
        layout.addWidget(label)

    def saveState(self, settings):
        pass

    def restoreState(self, settings):
        pass
