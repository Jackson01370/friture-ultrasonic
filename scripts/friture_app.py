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

"""Start Friture from a desktop shortcut, with no console anywhere.

Run under pythonw.exe, which is what the .lnk points at (see
scripts/make_shortcut.ps1). friture.bat does the same job from a terminal;
this exists because a .lnk to a .bat blinks a cmd window on its way through,
and because pythonw needs two things a console build does not:

  Qt's plugin paths. This PyQt5 does not register them itself. Without them
  Qt cannot find its platform plugin and the process dies before it can say
  why -- and from a shortcut there would be nothing to read even if it did.

  Somewhere for stdout and stderr to go. Under pythonw both are None, and
  Friture's own startup attaches a logging.StreamHandler() to stderr when it
  is not running frozen. Handing it a live file keeps that working and gives
  a place to look when a shortcut launch fails silently.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def point_qt_at_its_own_files():
    """Tell Qt where its plugins and QML modules are.

    Asked of PyQt5 rather than spelled out, so this keeps working across
    interpreters and virtual environments. Existing values win: someone who
    set them deliberately means it.
    """
    import PyQt5

    qt_root = Path(PyQt5.__file__).parent / "Qt5"
    # the plugins root, not just platforms: imageformats is where the SVG
    # plugin lives, and the toolbar icons are SVGs
    os.environ.setdefault("QT_PLUGIN_PATH", str(qt_root / "plugins"))
    os.environ.setdefault("QML2_IMPORT_PATH", str(qt_root / "qml"))


def give_output_somewhere_to_go():
    """Replace the missing pythonw streams with a file. Returns its path."""
    if sys.stdout is not None and sys.stderr is not None:
        return None

    import platformdirs

    log_dir = Path(platformdirs.user_log_dir("Friture", ""))
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "friture-launch.log"

    stream = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    sys.stdout = stream
    sys.stderr = stream
    return path


def main():
    sys.path.insert(0, str(ROOT))

    point_qt_at_its_own_files()
    log_path = give_output_somewhere_to_go()

    try:
        from friture.analyzer import main as friture_main
    except Exception:
        # Nothing has a console and the GUI does not exist yet, so a message
        # box is the only way this reaches anyone.
        _report_startup_failure(log_path)
        raise

    friture_main()


def _report_startup_failure(log_path):
    import traceback

    traceback.print_exc()
    details = "Friture could not start.\n\n%s" % traceback.format_exc(limit=3)
    if log_path is not None:
        details += "\nFull log: %s" % log_path
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, details, "Friture", 0x10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
