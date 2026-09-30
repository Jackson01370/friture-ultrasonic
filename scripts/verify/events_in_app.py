"""The whole chain with real audio: a recording of known events, found, replayed, kept.

    python scripts\verify\events_in_app.py FOLDER SPEECH.wav

Writes a 6-minute recording into FOLDER, in two contiguous 3-minute files,
dated an hour ago, holding three things at known moments:

    90 s    a bang: 0.3 s of broadband noise 25 dB over the room
    150 s   25 s of speech (SPEECH.wav) at about +3 dB in the speech band
    240 s   a 47,000 Hz tone that switches on and stays to the end

analyses it with the code the recorder runs live, opens the application on
FOLDER with a Recording Events dock, and checks:

  1. the three events are listed, each within a few seconds of its moment
  2. clicking the speech event enters replay 3 s before it
  3. keep on the bang protects the file holding it, and the replay bar
     shows the file as kept
  4. the live recording made while the application runs is in the list's
     folder too, with its own analysis log growing

Needs the UltraMic (the room is recorded meanwhile, into FOLDER).
"""

import ctypes
import os
import subprocess
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

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

logging.basicConfig(level=logging.WARNING)

folder = Path(sys.argv[1])
speech_path = Path(sys.argv[2])
folder.mkdir(parents=True, exist_ok=True)
os.environ["FRITURE_RECORDING_DIR"] = str(folder)       # this run only, never saved
FS = 250_000
BANG_S, SPEECH_S, LINE_S = 90.0, 150.0, 240.0

# -- the recording ----------------------------------------------------------------------
from friture.recording.wav_segment import SegmentInfo, WavSegmentWriter, float_to_int16, segment_name

rng = np.random.default_rng(7)
T0 = time.time() - 3600.0
n = int(360 * FS)
x = 0.003 * rng.normal(size=n)
# the bang
b0 = int(BANG_S * FS)
x[b0:b0 + int(0.3 * FS)] += 0.003 * 18 * rng.normal(size=int(0.3 * FS))
# the speech, at 250 kHz, scaled to about +3 dB over the room in 80-7800 Hz
fs_w, w = wavfile.read(speech_path)
if w.ndim > 1:
    w = w.mean(axis=1)
sp = resample_poly(w.astype(np.float64) / 32768.0, 250_000 // 50, fs_w // 50)[:int(25 * FS)]
room_band = 0.003 * np.sqrt(7720 / 125_000)            # white noise's share of 80-7800 Hz
sp *= room_band * 10 ** (3 / 20) / np.sqrt(np.mean(sp[np.abs(sp) > 0.02 * np.abs(sp).max()] ** 2))
s0 = int(SPEECH_S * FS)
x[s0:s0 + sp.size] += sp
# the line
l0 = int(LINE_S * FS)
t = np.arange(n - l0) / FS
x[l0:] += 0.004 * np.sin(2 * np.pi * 47_000 * t)
names = []
for k in range(2):
    part = x[k * 180 * FS:(k + 1) * 180 * FS].astype(np.float32)
    start = T0 + 180 * k
    info = SegmentInfo(wav=segment_name(start) + ".wav", fs=FS, channels=1, start_epoch=start,
                       start_local="known events", run_id="events-test", run_index=k * 180 * FS,
                       continues=names[-1] if names else None,
                       notes=["synthetic recording written by events_in_app.py"])
    wr = WavSegmentWriter(folder, info)
    for i in range(0, part.size, 1 << 18):
        wr.write(float_to_int16(part[i:i + (1 << 18)].reshape(-1, 1)))
    wr.close(start + 180)
    names.append(info.wav)
del x
print("wrote %s and %s" % tuple(names))
subprocess.run([sys.executable, str(ROOT / "scripts" / "verify" / "analyse_recordings.py"), str(folder)],
               check=True, stdout=subprocess.DEVNULL)
print("analysed them")

# -- the application --------------------------------------------------------------------------
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
from friture.analyzer import Friture
from friture.recording.analysis_log import log_path
from friture.recording.store import SegmentStore
from friture.recording.wav_segment import SegmentInfo as _SI
from friture.recording_events import RecordingEvents_Widget

window = Friture()
window.show()
try:
    ctypes.windll.user32.ShowWindow(int(window.winId()), 3)
except Exception:
    pass
dock = None
for d in window.dockmanager.docks:
    if isinstance(d.audiowidget, RecordingEvents_Widget):
        dock = d.audiowidget
if dock is None:
    window.dockmanager.new_dock()
    window.dockmanager.docks[-1].widget_select(11)
    dock = window.dockmanager.docks[-1].audiowidget
vm = dock.view_model()

ok = True


def check(label, cond, detail=""):
    global ok
    print("%-62s %s   %s" % (label, "ok " if cond else "FAIL", detail), flush=True)
    ok = ok and cond


def rows():
    return [vm.events.get(i) for i in range(vm.events.count)]


def near(kind, at, brief=None, tol=6.0):
    return [r for r in rows() if r["kind"] == kind and abs(r["start"] - (T0 + at)) <= tol
            and (brief is None or r["brief"] == brief)]


state = {}


def listed():
    r = rows()
    print("   the list:")
    for x in sorted(r, key=lambda x: x["start"]):
        print("     %+7.1f s  %-22s %s" % (x["start"] - T0, x["kind_label"], x["detail"]))
    check("1. the bang is listed as loud, at 90 s", bool(near("loud", BANG_S)))
    check("1. the speech is listed as sustained voice-like, at 150 s", bool(near("voice", SPEECH_S, brief=False)))
    check("1. the 47 kHz tone is listed as a line appearing, at 240 s",
          any(abs((r["start"] - T0) - LINE_S) <= 60 and "47.0" in r["detail"] for r in r if r["kind"] == "line_on"))
    # anything else in the known 6 minutes, and whether it lies inside one
    # of the planted events (speech lifts the low band while it lasts)
    def inside(x):
        at = x["start"] - T0
        return (abs(at - BANG_S) <= 2 or SPEECH_S - 2 <= at <= SPEECH_S + 27
                or (x["kind"] == "line_on" and abs(at - LINE_S) <= 60))
    known = [x for x in r if x["start"] < T0 + 360]
    extras = [x for x in known if not inside(x)]
    during = [x for x in known if inside(x) and x["kind"] == "loud" and SPEECH_S - 2 <= x["start"] - T0]
    print("   loud events during the speech (the speech itself): %d; anything else: %d %s" % (
        len(during), len(extras), [(round(x["start"] - T0), x["kind"]) for x in extras]))
    check("1. nothing listed that was not planted", not extras)
    speech = near("voice", SPEECH_S, brief=False)[0]
    vm.play(speech["key"])
    # read at once: from here on the replay is playing and the position moves
    rp = window.replay
    state["pos_after_click"] = rp.source.position - T0 if rp.source else None


def after_play():
    rp = window.replay
    pos = state["pos_after_click"]
    check("2. clicking the speech entered replay 3 s before it",
          rp.active and pos is not None and abs(pos - (SPEECH_S - 3.0)) < 0.5,
          "replay at %s s" % (None if pos is None else round(pos, 2)))
    bang = near("loud", BANG_S)[0]
    vm.toggle_protect(bang["key"])
    kept = [s.wav for s in SegmentStore(folder).segments() if s.protected]
    check("3. keep on the bang protects the file holding it", kept == [names[0]], "%s" % kept)
    rp.seek(T0 + BANG_S)


def after_seek():
    rp = window.replay
    rp._update_view()
    check("3. the replay bar shows that file as kept", rp.vm.current_protected, rp.vm.status_text)
    img = window.quick_view.grabWindow()
    img.save(str(folder / "_events.png"))
    rp.leave()
    live = [s for s in SegmentStore(folder).segments() if s.run_id != "events-test"]
    grown = [log_path(folder, s.wav[:-4]).exists() for s in live]
    check("4. the room is being recorded into the same folder, with its log",
          bool(live) and all(grown), "%d live file(s)" % len(live))
    QTimer.singleShot(300, window.close)


t_start = time.monotonic()
steps = [(12.0, listed), (16.0, after_play), (19.0, after_seek)]


def tick():
    while steps and steps[0][0] <= time.monotonic() - t_start:
        _, fn = steps.pop(0)
        try:
            fn()
        except Exception as e:
            check("a step raised", False, repr(e))


timer = QTimer()
timer.timeout.connect(tick)
timer.start(100)
app.exec_()
print("\n%s" % ("ALL OK" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
