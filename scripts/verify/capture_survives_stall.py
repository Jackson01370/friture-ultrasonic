"""Does the capture survive a frozen GUI -- and if it is starved anyway, does it come back?

    python scripts\verify\capture_survives_stall.py

Needs the UltraMic and nothing else holding it.

1. THE FIX. The GUI thread sleeps 5 s. Before the capture thread existed,
   rtmixer's record action died for good once its 2.1 s ring filled, with
   no flag and no error. Now the frames reaching a raw sink must keep pace
   with the device clock straight through the freeze, with no break in the
   run.

2. THE SAFETY NET. The capture thread itself is starved, by holding the
   lock it drains under for 3 s, so the ring really does fill and the
   action really does die. It must be noticed and re-issued, the break
   reported as a new run, and samples must flow again. What was lost is
   lost -- the point is that it is a gap, not the end.
"""

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))

import logging
from PyQt5.QtCore import QCoreApplication

logging.basicConfig(level=logging.ERROR)
from friture.audiobackend import SAMPLING_RATE, AudioBackend

app = QCoreApplication(sys.argv)
be = AudioBackend()
if be.device is None:
    print("FAIL: no capture device")
    sys.exit(1)
be.restart()
ok = True


def check(label, cond, detail=""):
    global ok
    print("%-52s %s   %s" % (label, "ok " if cond else "FAIL", detail))
    ok = ok and cond


lock = threading.Lock()
st = {"frames": 0, "last": None, "breaks": 0, "runs": set()}


def sink(block, run_id, run_index, t_first, overflow):
    with lock:
        if st["last"] is not None and st["last"] != (run_id, run_index):
            st["breaks"] += 1
        st["frames"] += block.shape[0]
        st["runs"].add(run_id)
        st["last"] = (run_id, run_index + block.shape[0])


be.add_raw_sink(sink)


def pump(seconds):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        be.fetchAudioData()
        app.processEvents()
        time.sleep(0.01)


def frames():
    with lock:
        return st["frames"], st["breaks"], len(st["runs"])


# -- 1 -----------------------------------------------------------------------------
pump(1.0)
f0, b0, _ = frames()
s0 = be.get_stream_time()
time.sleep(5.0)                                   # the GUI thread, frozen
pump(2.0)
f1, b1, _ = frames()
clock = be.get_stream_time() - s0
got = (f1 - f0) / SAMPLING_RATE
check("GUI frozen 5 s: frames kept pace with the device", abs(clock - got) < 0.05,
      "device %.3f s, frames %.3f s" % (clock, got))
check("GUI frozen 5 s: no break in the run", b1 == b0, "%d break(s)" % (b1 - b0))
check("GUI frozen 5 s: the record action never died", be.capture_restarts == 0,
      "%d restart(s)" % be.capture_restarts)

# -- 2 -----------------------------------------------------------------------------
_, _, runs_before = frames()
with be._lock:                                    # starve the capture thread itself
    time.sleep(3.0)
pump(3.0)
f2, b2, runs_after = frames()
check("capture thread starved 3 s: the dead action was re-issued", be.capture_restarts == 1,
      "%d restart(s)" % be.capture_restarts)
check("...and the break was reported as a new run", runs_after == runs_before + 1,
      "%d run(s) before, %d after" % (runs_before, runs_after))
pump(2.0)
f3, _, _ = frames()
check("...and samples flow again afterwards", (f3 - f2) / SAMPLING_RATE > 1.5,
      "%.2f s in the next 2 s" % ((f3 - f2) / SAMPLING_RATE))

# -- 3 -----------------------------------------------------------------------------
# A stop must end the run where the audio ended and not one block later: the
# blocks still in the ring when Stop is pressed are the end of THIS run.
# Before the fix they came out as a run of their own -- measured, a 16 ms
# file recorded after "a 0.022 s gap" that never happened.
seen = []
lock2 = threading.Lock()


def order(block, run_id, run_index, t_first, overflow):
    with lock2:
        seen.append((run_id, run_index, block.shape[0]))


be.add_raw_sink(order)
pump(1.0)
with lock2:
    run_before = seen[-1][0]
be.pause()
with lock2:
    n_at_pause = len(seen)
time.sleep(1.0)
pump(0.5)
with lock2:
    n_after_pause = len(seen)
be.restart()
pump(1.5)
with lock2:
    tail = [s for s in seen if s[0] == run_before]
    later = sorted({s[0] for s in seen[n_at_pause:]})
contiguous = all(b[1] == a[1] + a[2] for a, b in zip(tail, tail[1:]))
check("stopped: nothing arrives while stopped", n_after_pause == n_at_pause,
      "%d block(s) after the stop" % (n_after_pause - n_at_pause))
check("the run before the stop is contiguous to its last block", contiguous,
      "%d blocks, last one %d frames" % (len(tail), tail[-1][2] if tail else 0))
check("after Start, exactly one new run and no stray one", later == [run_before + 2],
      "runs after the stop: %s (the stop itself uses %d, which must stay empty)"
      % (later, run_before + 1))

be.close()
print("\n%s" % ("ALL OK" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
