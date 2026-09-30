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

"""The folder of segments: what is in it, how big it is, what to delete.

Rotation deletes the OLDEST segments first until the folder is back under
its cap, and never touches two kinds: a segment marked protected, and the
one being written. If protected segments alone exceed the cap the folder
stays over it -- deleting something the user asked to keep is worse than
using more disk than asked, and the status says so.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from friture.recording.wav_segment import HEADER_BYTES, SegmentInfo, repair

BYTES_PER_GB = 1_000_000_000


def default_directory() -> Path:
    """D:\\friture-recordings when there is a D: drive, else Documents.

    Deliberately not the source tree: the repository is public, and these
    are recordings of a room.
    """
    if os.name == "nt" and os.path.isdir("D:\\"):
        return Path("D:\\friture-recordings")
    return Path.home() / "Documents" / "friture-recordings"


def inside_git_worktree(folder: Path) -> Path | None:
    """The repository root if folder sits inside one, so the UI can warn."""
    p = Path(folder).resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists():
            return parent
    return None


ANALYSIS_SUFFIX = ".analysis.jsonl"      # see friture.recording.analysis_log


def companions(wav: Path) -> list[Path]:
    """Every file that belongs to one segment: the audio, its sidecar, its log.

    Rotation deletes them together and the cap counts them together; a log
    left behind by a deleted WAV would describe audio that no longer exists.
    """
    return [wav, wav.with_suffix(".json"), wav.parent / (wav.stem + ANALYSIS_SUFFIX)]


class SegmentStore:

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.removed_empty = 0        # header-less files cleared by repair_interrupted

    def ensure(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

    def wav_files(self) -> list[Path]:
        if not self.folder.is_dir():
            return []
        return sorted(self.folder.glob("*.wav"))

    def segments(self) -> list[SegmentInfo]:
        out = []
        for wav in self.wav_files():
            side = wav.with_suffix(".json")
            if side.exists():
                try:
                    out.append(SegmentInfo.load(side))
                except Exception:
                    continue
        return sorted(out, key=lambda s: s.start_epoch)

    def total_bytes(self) -> int:
        total = 0
        for wav in self.wav_files():
            for p in companions(wav):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.folder).free

    def repair_interrupted(self, exclude: str | None = None) -> list[SegmentInfo]:
        """Repair every segment a crash left open, except the one in use.

        A <stem>.json.tmp beside a segment is a save that Windows would not
        let finish (the sidecar was held open by someone else); it is the
        newer state, so it is adopted first.
        """
        fixed = []
        for wav in self.wav_files():
            if wav.name == exclude:
                continue
            side = wav.with_suffix(".json")
            pending = wav.with_suffix(".json.tmp")
            if pending.exists():
                try:
                    os.replace(pending, side)
                except OSError:
                    pass
            needs = not side.exists()
            if not needs:
                try:
                    needs = SegmentInfo.load(side).state == "open"
                except Exception:
                    needs = True
            if needs:
                # Shorter than a header: nothing reached the disk, so there
                # is nothing to recover. Left alone it stayed "open" for
                # ever, because repair() cannot read a header that is not
                # there; it goes, with its sidecar.
                try:
                    if wav.stat().st_size < HEADER_BYTES:
                        for p in companions(wav):
                            if p.exists():
                                p.unlink()
                        self.removed_empty += 1
                        continue
                except OSError:
                    continue
                try:
                    fixed.append(repair(self.folder, wav.name))
                except Exception:
                    continue
        return fixed

    def overlapping(self, start: float, end: float) -> list[SegmentInfo]:
        """The segments holding any audio between two wall-clock times."""
        out = []
        for s in self.segments():
            seg_end = s.end_epoch if s.end_epoch else s.start_epoch + s.n_frames / max(s.fs, 1)
            if s.state == "open":
                wav = self.folder / s.wav
                try:
                    frames = (wav.stat().st_size - 44) // (2 * s.channels)
                    seg_end = s.start_epoch + frames / s.fs
                except OSError:
                    pass
            if s.start_epoch <= end and seg_end >= start:
                out.append(s)
        return out

    def set_protected(self, wavs, on: bool) -> list[str]:
        """Mark segments protected (or not); returns the ones whose sidecar changed."""
        changed = []
        for wav in wavs:
            side = self.folder / (Path(wav).stem + ".json")
            try:
                info = SegmentInfo.load(side)
            except Exception:
                continue
            if info.protected != on:
                info.protected = on
                if info.save(self.folder):
                    changed.append(info.wav)
        return changed

    def protected_bytes(self) -> tuple[int, int]:
        """(number of protected segments, bytes they take)."""
        n = size = 0
        for s in self.segments():
            if s.protected:
                n += 1
                for p in companions(self.folder / s.wav):
                    try:
                        size += p.stat().st_size
                    except OSError:
                        pass
        return n, size

    def rotate(self, cap_bytes: int, keep: str | None = None) -> tuple[int, int]:
        """Delete the oldest unprotected segments until under cap_bytes.

        Returns (segments deleted, bytes still over the cap -- 0 if under).
        """
        deleted = 0
        total = self.total_bytes()
        if total <= cap_bytes:
            return 0, 0
        candidates = []
        for wav in self.wav_files():
            if wav.name == keep:
                continue
            side = wav.with_suffix(".json")
            protected = False
            if side.exists():
                try:
                    protected = SegmentInfo.load(side).protected
                except Exception:
                    pass
            if not protected:
                candidates.append(wav)
        for wav in candidates:            # sorted by name = by start time
            if total <= cap_bytes:
                break
            for p in companions(wav):
                try:
                    size = p.stat().st_size
                    p.unlink()
                    total -= size
                except OSError:
                    pass
            deleted += 1
        return deleted, max(total - cap_bytes, 0)
