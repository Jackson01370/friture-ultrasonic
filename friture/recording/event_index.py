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

"""The events of a whole recording folder, kept current cheaply.

A full folder is 28 hours of logs, about 34 MB of JSON. Reading it all again
every time the list refreshes would take seconds; only one file changes
between refreshes -- the one being written -- so each log is parsed once
and kept, keyed by its size and modification time, and only files that
changed are read again. A log whose WAV was deleted by rotation drops out.

The work runs on a thread of its own: refresh() starts it and returns;
result() hands back the latest finished answer. The GUI never waits.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from friture.recording.analysis_log import SUFFIX, read_log
from friture.recording.events import Event, detect


class EventIndex:

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[tuple[int, int], list[dict]]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: tuple[list[Event], float, float, Path | None] = ([], 0.0, 0.0, None)
        self._folder: Path | None = None

    def records(self, folder: Path) -> list[dict]:
        """Every record in the folder's logs, re-reading only what changed."""
        folder = Path(folder)
        if self._folder != folder:
            self._cache.clear()
            self._folder = folder
        seen = set()
        out: list[dict] = []
        for path in sorted(folder.glob("*" + SUFFIX)):
            if not path.with_name(path.name[:-len(SUFFIX)] + ".wav").exists():
                continue                    # the audio is gone; so is the event
            try:
                st = path.stat()
            except OSError:
                continue
            key = (st.st_size, st.st_mtime_ns)
            seen.add(path)
            cached = self._cache.get(path)
            if cached is None or cached[0] != key:
                cached = (key, read_log(path)[1])
                self._cache[path] = cached
            out.extend(cached[1])
        for gone in set(self._cache) - seen:
            del self._cache[gone]
        return out

    def compute(self, folder: Path) -> list[Event]:
        """Synchronously: the events of the folder now."""
        return detect(self.records(folder))

    # -- the background version, for the GUI ----------------------------------------

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def refresh(self, folder: Path) -> bool:
        """Start recomputing in the background; False if one is already running."""
        if self.busy or folder is None:
            return False
        self._thread = threading.Thread(target=self._work, args=(Path(folder),),
                                        name="friture-events", daemon=True)
        self._thread.start()
        return True

    def _work(self, folder: Path) -> None:
        t0 = time.perf_counter()
        try:
            events = self.compute(folder)
        except Exception:
            events = []
        with self._lock:
            self._result = (events, time.time(), time.perf_counter() - t0, folder)

    def result(self) -> tuple[list[Event], float, float, Path | None]:
        """(events, when they were computed, how long it took, for which folder)."""
        with self._lock:
            return self._result
