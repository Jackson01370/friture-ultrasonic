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

"""The <stem>.analysis.jsonl files: writing them, and reading them back.

One JSON object per line, a "header" record first. Each record is flushed
as it is written, so a crash costs at most the line being written -- and
the reader skips a last line that was cut short, rather than refusing the
whole file.
"""

from __future__ import annotations

import json
from pathlib import Path

from friture.recording.analysis import header

SUFFIX = ".analysis.jsonl"


def log_path(folder: Path, stem: str) -> Path:
    return Path(folder) / (stem + SUFFIX)


class AnalysisLog:
    """Appends records to the log of whichever segment each one belongs to."""

    def __init__(self, fs: int) -> None:
        self.fs = fs
        self._folder: Path | None = None
        self._stem: str | None = None
        self._f = None

    def write(self, folder: Path, record: dict) -> None:
        folder = Path(folder)
        stem = record["seg"]
        if self._f is None or stem != self._stem or folder != self._folder:
            self.close()
            path = log_path(folder, stem)
            new = not path.exists()
            self._f = open(path, "a", encoding="utf-8")
            self._folder, self._stem = folder, stem
            if new:
                self._f.write(json.dumps(header(self.fs)) + "\n")
        self._f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None
                self._stem = None


def read_log(path: Path) -> tuple[dict | None, list[dict], int]:
    """(header, records, lines that could not be read) of one log file."""
    head, records, bad = None, [], 0
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None, [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if rec.get("kind") == "header":
            head = rec
        else:
            records.append(rec)
    return head, records, bad
