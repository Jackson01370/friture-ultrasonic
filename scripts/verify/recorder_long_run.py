"""Run the real application with continuous recording, and abuse it on purpose.

    python scripts\verify\recorder_long_run.py FOLDER [--minutes M]
                                               [--stall-at MIN] [--stop-at MIN]

The whole application -- window, docks, timers -- records into FOLDER for M
minutes (default 32, so three 10-minute segments close on schedule). Two
things are done to it along the way, because they are how a drive recorder
fails:

  --stall-at  the GUI thread is frozen for 5 s. The old capture path died
              for good after 2.1 s of this; the recording must not notice.
  --stop-at   the capture is stopped for 10 s and started again. The file
              must end there, and the next one must say a gap of about 10 s
              came before it.

Every minute it logs the recorder state and the stream clock, so afterwards
the frames on disk can be checked against how long the stream really ran.
The folder is then judged by recording_check.py.

The recording folder is put into Friture's settings for the run and the
user's own settings are put back afterwards by the caller (see the report).
"""

import ctypes
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")

import logging
from PyQt5.QtCore import QSettings, QTimer
from PyQt5.QtWidgets import QApplication

logging.basicConfig(level=logging.WARNING)

args = sys.argv[1:]
folder = Path(args[0])


def option(name, default):
    return args[args.index(name) + 1] if name in args else default


minutes = float(option("--minutes", 32))
stall_at = float(option("--stall-at", 5))
stop_at = float(option("--stop-at", 13))
folder.mkdir(parents=True, exist_ok=True)

# the folder goes into the settings BEFORE the window exists, so not one
# block goes to the user's own recording folder during the test
s = QSettings("Friture", "Friture")
s.beginGroup("AudioBackend")
s.setValue("continuousRecording", True)
s.setValue("recordingDir", str(folder))
s.setValue("recordingCapGB", 50)
s.endGroup()
s.sync()

app = QApplication(sys.argv)
from friture.analyzer import Friture
from friture.audiobackend import AudioBackend, SAMPLING_RATE
from friture.recording.recorder import ContinuousRecorder

# a short segment length makes a quick trial exercise the splitting too;
# the real run leaves it at the 10 minutes the application uses
if "--segment-s" in args:
    ContinuousRecorder.SEGMENT_S = float(option("--segment-s", 600))

window = Friture()
window.show()
try:
    h = int(window.winId())
    ctypes.windll.user32.ShowWindow(h, 3)
except Exception:
    pass

backend = AudioBackend()
log_path = folder / "_run_log.jsonl"
log = open(log_path, "w", encoding="utf-8")
t0 = time.monotonic()
events = {"stalled": False, "stopped": False, "restarted": False}


def record(kind, **extra):
    st = window.recorder.status()
    row = dict(kind=kind, t=round(time.monotonic() - t0, 3), wall=time.time(),
               stream_time=backend.get_stream_time() if backend.stream is not None else None,
               run_id=backend._run_id, run_index=backend._run_index,
               capture_restarts=backend.capture_restarts,
               state=st.state, file=st.current_file, seconds_in_file=round(st.current_seconds, 3),
               stored_gb=round(st.stored_bytes / 1e9, 3), dropped=st.dropped_blocks,
               closed=st.segments_closed, message=st.message, **extra)
    log.write(json.dumps(row) + "\n")
    log.flush()
    print("%6.1f min  %-9s %-28s seg %6.1f s  stored %.2f GB  dropped %d  %s"
          % (row["t"] / 60, st.state, st.current_file, st.current_seconds,
             st.stored_bytes / 1e9, st.dropped_blocks, kind), flush=True)


def tick():
    m = (time.monotonic() - t0) / 60.0
    if not events["stalled"] and m >= stall_at:
        events["stalled"] = True
        record("before-stall")
        time.sleep(5.0)                     # the GUI thread, frozen
        record("after-stall")
    if not events["stopped"] and m >= stop_at:
        events["stopped"] = True
        record("before-stop")
        window.timer_toggle()               # the Stop button
        QTimer.singleShot(10_000, restart)
    if m >= minutes:
        record("end")
        grab = window.quick_view.grabWindow()
        grab.save(str(folder / "_window.png"))
        timer.stop()
        QTimer.singleShot(500, window.close)


def restart():
    record("stopped-for-10s")
    window.timer_toggle()                   # the Start button
    events["restarted"] = True
    record("after-restart")


minute = QTimer()
minute.timeout.connect(lambda: record("minute"))
minute.start(60_000)
timer = QTimer()
timer.timeout.connect(tick)
timer.start(1000)
QTimer.singleShot(3000, lambda: record("start"))
app.exec_()
log.close()
print("done; log in %s" % log_path)
