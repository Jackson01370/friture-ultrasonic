"""Drive the Recording Events dock offscreen, without the microphone.

The dock's plumbing on a folder of known content: the rows it lists, the
filters, a click that asks for a replay at the right moment, and the keep
button that protects the files around an event -- and survives rotation.
The event detection itself is tested in friture/test/test_recording_events.py
and the whole chain with real audio in replay_in_app.py; this is the dock.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")

import numpy as np
from PyQt5.QtCore import QObject
from PyQt5.QtQml import QQmlEngine
from PyQt5.QtQuick import QQuickView
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

from friture.qml_tools import qml_url
from friture.recording.analysis_log import log_path
from friture.recording.session import GetRecordingSession
from friture.recording.store import SegmentStore
from friture.recording.wav_segment import SegmentInfo, WavSegmentWriter
from friture.recording_events import RecordingEvents_Widget

FS = 250_000
ok = True


def check(label, cond, detail=""):
    global ok
    print("%-58s %s   %s" % (label, "ok " if cond else "FAIL", detail))
    ok = ok and cond


folder = Path(tempfile.mkdtemp())
T0 = 1_790_000_000.0


def write_segment(k, protected=False):
    """Segment k covers [T0 + 600k, T0 + 600(k+1)): a tiny WAV, a sidecar that
    says 10 minutes, and a log of records for those 10 minutes."""
    start = T0 + 600 * k
    name = "2026-09-01_00-%02d-00.000.wav" % (10 * k)
    info = SegmentInfo(wav=name, fs=FS, channels=1, start_epoch=start, start_local=name,
                       run_id="r", run_index=600 * k * FS, protected=protected)
    w = WavSegmentWriter(folder, info)
    w.write(np.zeros((10, 1), dtype=np.int16))
    w.close(start + 600)
    side = SegmentInfo.load(folder / (name[:-4] + ".json"))
    side.n_frames, side.end_epoch = 600 * FS, start + 600
    side.save(folder)
    rng = np.random.default_rng(k)
    stem = name[:-4]
    with open(log_path(folder, stem), "w", encoding="utf-8") as f:
        for s in range(600):
            bands = list(-60 + rng.normal(0, 0.8, 7))
            if k == 1 and s == 100:
                bands[0] += 22                      # a bang at segment 1 + 100 s
            f.write(json.dumps({"kind": "levels", "t": start + s, "seg": stem, "pos": s * FS, "run": "r",
                                "idx": (600 * k + s) * FS, "n": FS, "dur": 1.0, "bands_dbfs": bands}) + "\n")
        for w5 in range(120):
            score = 3.0 + rng.normal(0, 0.1)
            if k == 2 and 40 <= w5 < 44:
                score = 8.0                         # speech at segment 2 + 200 s
            if k == 0 and w5 == 60:
                score = 7.0                         # a knock at segment 0 + 300 s
            f.write(json.dumps({"kind": "voice", "t": start + 5 * w5, "seg": stem, "pos": 5 * w5 * FS,
                                "run": "r", "idx": (600 * k + 5 * w5) * FS, "n": 5 * FS, "dur": 5.0,
                                "score": score, "voiced": 1.0, "f0_hz": 180.0, "longest_s": 4.0}) + "\n")
    return name


names = [write_segment(k) for k in range(4)]
BANG, SPEECH, KNOCK = T0 + 600 + 100, T0 + 1200 + 200, T0 + 300

played = []


class FakeReplay:
    active = True
    playing = True

    def enter(self):
        return True

    def seek(self, epoch):
        played.append(epoch)

    def play_pause(self):
        pass


session = GetRecordingSession()
session.get_folder = lambda: str(folder)
session.replay = FakeReplay()

widget = RecordingEvents_Widget(None)
vm = widget.view_model()
check("qml file name", widget.qml_file_name() == "RecordingEvents.qml")


def pump(seconds=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        widget.canvasUpdate()
        app.processEvents()
        if vm.events.count and not widget._index.busy:
            break
        time.sleep(0.02)


def rows():
    return [vm.events.get(i) for i in range(vm.events.count)]


pump()
r = rows()
kinds = [(x["kind"], x["brief"]) for x in r]
check("lists the bang, the speech and the knock", sorted(kinds) == sorted(
    [("loud", False), ("voice", False), ("voice", True)]), "%s" % kinds)
check("newest first", [x["start"] for x in r] == sorted((x["start"] for x in r), reverse=True))
check("the summary counts them", "3 event" in vm.summary_text, vm.summary_text)

vm.show_brief = False
widget.canvasUpdate()
check("hiding brief events hides the knock", all(not x["brief"] for x in rows()), "%d rows" % vm.events.count)
vm.show_brief = True
vm.show_loud = False
widget.canvasUpdate()
check("hiding loud events hides the bang", all(x["kind"] != "loud" for x in rows()))
vm.show_loud = True
widget.canvasUpdate()

speech = next(x for x in rows() if x["kind"] == "voice" and not x["brief"])
vm.play(speech["key"])
check("clicking an event replays from 3 s before it",
      played and abs(played[-1] - (SPEECH - 3.0)) < 1e-6, "seek to %s" % (played[-1] - T0 if played else None))

bang = next(x for x in rows() if x["kind"] == "loud")
check("nothing is kept to begin with", not bang["protected"])
vm.toggle_protect(bang["key"])
bang = next(x for x in rows() if x["kind"] == "loud")
check("keep protects the files around the event", bang["protected"],
      "%s" % [(s.wav, s.protected) for s in SegmentStore(folder).segments()])
kept = [s.wav for s in SegmentStore(folder).segments() if s.protected]
check("...exactly the file holding it (the 30 s pad stays inside it)", kept == [names[1]], "%s" % kept)
check("the kept files are counted", "Kept: 1 file" in vm.protected_text, vm.protected_text)
vm.protected_only = True
widget.canvasUpdate()
check("'protected only' lists just that event", [x["kind"] for x in rows()] == ["loud"])
vm.protected_only = False

SegmentStore(folder).rotate(cap_bytes=1)
left = sorted(p.name for p in folder.glob("*.wav"))
check("rotation to nothing leaves exactly the kept file", left == [names[1]], "%s" % left)
widget._request(force=True)
time.sleep(0.3)
pump()
check("the events of deleted audio drop out of the list",
      [x["kind"] for x in rows()] == ["loud"], "%s" % [x["kind"] for x in rows()])

# the QML against the real view model
engine = QQmlEngine()
view = QQuickView(engine, None)
view.setResizeMode(QQuickView.SizeRootObjectToView)
view.setInitialProperties({"viewModel": vm, "fixedFont": "Courier New"})
view.setSource(qml_url("RecordingEvents.qml"))
if view.status() == QQuickView.Error:
    check("RecordingEvents.qml loads", False, "\n".join(e.toString() for e in view.errors()))
else:
    view.resize(900, 300)
    view.show()
    app.processEvents()
    lst = view.rootObject().findChild(QObject, "event_list")
    check("RecordingEvents.qml loads and lists the rows", lst is not None and lst.property("count") == 1,
          "count=%s" % (lst.property("count") if lst is not None else None))
    for h in (160, 400):
        view.resize(900, h)
        app.processEvents()
        root = view.rootObject()
        content = root.childItems()[0]
        check("fits a %d px dock" % h, content.property("height") <= root.property("height") + 1)

shutil.rmtree(folder, ignore_errors=True)
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
