"""Run the REAL application and watch its Digital Decode dock decode the speakers.

    python scripts\verify\app_live_decode.py [DIR] [--mode fsk|fm|am]   the application, driven
    python scripts\verify\app_live_decode.py --play --mode ...          (spawned by the above) the transmitter

The main window, the dock manager, the audio backend, the QML and the
timers are all the real ones -- this is Friture.analyzer.main() with a
timer added that reads the dock's view model once a second and prints it,
and a QQuickView.grabWindow() of the QML scene saved as PNG at the end
(desktop screen grabs of a Qt Quick window come back blank on this PC).

The dock is taken from the saved layout if there is one, else added. The
band is set to the tones the speakers can reproduce (see
decode_over_the_air.py): 11 / 13.5 kHz FSK at 25 Bd, the 13.5 kHz tone
sent 6 dB louder to arrive level with the other.

Needs the UltraMic free and the speakers on. Exit status 1 if the dock never
shows bits that are a run of the sent ones.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

PLAY_FS = 48000
BAUD = 25.0
SECONDS = 12.0
BITS = np.random.default_rng(2026).integers(0, 2, int(SECONDS * BAUD))
# --mode fsk (default): 11 / 13.5 kHz FSK at 25 Bd, bits recovered
# --mode fm : a 12 kHz carrier swung +-500 Hz at 20 Hz, the readout names both
# --mode am : a 12 kHz carrier 50% amplitude-modulated at 20 Hz
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "fsk"
FM_DEVIATION_HZ, FM_RATE_HZ = 500.0, 20.0
AM_DEPTH, AM_RATE_HZ = 0.5, 20.0


def transmitter() -> None:
    import sounddevice as sd
    if MODE == "fsk":
        T = PLAY_FS / BAUD
        n = int(len(BITS) * T)
        idx = np.clip((np.arange(n) / T).astype(int), 0, len(BITS) - 1)
        f = np.where(BITS[idx] == 1, 13500.0, 11000.0)
        g = np.where(BITS[idx] == 1, 2.0, 1.0)
        x = 0.4 * g * np.cos(2 * np.pi * np.cumsum(f) / PLAY_FS)
    else:
        t = np.arange(int(SECONDS * PLAY_FS)) / PLAY_FS
        if MODE == "fm":
            x = 0.4 * np.cos(2 * np.pi * 12000.0 * t
                             + (FM_DEVIATION_HZ / FM_RATE_HZ) * np.sin(2 * np.pi * FM_RATE_HZ * t))
        else:
            x = 0.3 * (1.0 + AM_DEPTH * np.cos(2 * np.pi * AM_RATE_HZ * t)) * np.cos(2 * np.pi * 12000.0 * t)
    r = np.linspace(0, 1, 480)
    x[:480] *= r
    x[-480:] *= r[::-1]
    sd.play(np.zeros(PLAY_FS, np.float32), PLAY_FS, blocking=True)     # warm-up, see decode_over_the_air.py
    sd.play(x.astype(np.float32), PLAY_FS, blocking=True)


if "--play" in sys.argv:
    transmitter()
    sys.exit(0)

# This PyQt5 does not register Qt's own paths -- see friture.bat.
import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("FRITURE_RECORDING_DIR", "off")   # never the user's recording folder
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")

import logging

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

logging.basicConfig(level=logging.WARNING)

app = QApplication(sys.argv)

from friture.analyzer import Friture
from friture.digital_decode import DigitalDecode_Widget
from friture.listen.listen_band_view_model import GetListenBand

positional = [a for a in sys.argv[1:] if not a.startswith("-") and a not in ("fsk", "fm", "am")]
out_dir = Path(positional[0]) if positional else ROOT / "sandbox"
out_dir.mkdir(parents=True, exist_ok=True)

window = Friture()
window.show()

decode_dock = None
for dock in window.dockmanager.docks:
    if isinstance(dock.audiowidget, DigitalDecode_Widget):
        decode_dock = dock
        break
if decode_dock is None:
    window.dockmanager.new_dock()
    decode_dock = window.dockmanager.docks[-1]
    decode_dock.widget_select(9)
    print("added a Digital Decode dock", flush=True)
else:
    print("using the Digital Decode dock from the saved layout: %s" % decode_dock.objectName(), flush=True)
widget = decode_dock.audiowidget
widget.set_mode(MODE)
band = GetListenBand()
band.set_width_hz(6000)
band.set_centre_hz(12250)
vm = widget.view_model()

truth = "".join(str(b) for b in BITS)
state = {"play": None, "t0": time.time(), "best": 0, "done": False, "good_readouts": 0}


def analog_readout_is_right() -> bool:
    """The dock's own numbers against what the transmitter sends.

    The rate must be exact; the deviation and the depth are allowed to read
    up to 40% low, which is what this room does to them (its reflections
    blur the modulation: measured on recordings, FM +-500 Hz read
    +-320..380 Hz, AM 50% read 23..37%).
    """
    dec = widget.decoder
    if not dec.analog_carrier or abs(dec.last_carrier_hz - 12000.0) > 100.0:
        return False
    if MODE == "fm":
        return (0.6 * FM_DEVIATION_HZ <= dec.last_deviation <= 1.3 * FM_DEVIATION_HZ
                and abs(dec.last_mod_rate_hz - FM_RATE_HZ) <= 2.0)
    return 0.4 * AM_DEPTH <= dec.last_deviation <= 1.3 * AM_DEPTH and abs(dec.last_mod_rate_hz - AM_RATE_HZ) <= 2.0


def tick() -> None:
    t = time.time() - state["t0"]
    if MODE == "fsk":
        shown = vm.bits.replace(" ", "")
        matched = max((k for k in range(len(shown) + 1) if shown[:k] in truth), default=0)
        state["best"] = max(state["best"], matched)
        print("%5.1f s | %-58s | %-54s | %s%s" % (
            t, vm.signal_text[:58], vm.decode_text[:54], vm.bits,
            ("  (%d bits match the sent ones)" % matched) if shown else ""), flush=True)
    else:
        right = state["play"] is not None and analog_readout_is_right()
        state["good_readouts"] += int(right)
        print("%5.1f s | %-78s | %d points%s" % (
            t, vm.signal_text[:78], len(vm.trace), "  (matches the transmitter)" if right else ""), flush=True)
    if state["play"] is None and t >= 3.0:
        state["play"] = subprocess.Popen([sys.executable, __file__, "--play", "--mode", MODE])
        print("   transmitter started", flush=True)
    if t >= 3.0 + SECONDS + 1.0 and not state["done"]:
        state["done"] = True
        image = window.quick_view.grabWindow()
        png = out_dir / ("app_live_decode_%s.png" % MODE)
        image.save(str(png))
        print("saved %s (%dx%d)" % (png, image.width(), image.height()), flush=True)
        if MODE == "fsk":
            state["ok"] = state["best"] >= 32
            print("RESULT: %s (best run of sent bits shown: %d)" % ("ok" if state["ok"] else "FAIL", state["best"]), flush=True)
        else:
            state["ok"] = state["good_readouts"] >= 5
            print("RESULT: %s (%d one-second readouts matched the transmitter)"
                  % ("ok" if state["ok"] else "FAIL", state["good_readouts"]), flush=True)
        QTimer.singleShot(300, window.close)


timer = QTimer()
timer.timeout.connect(tick)
timer.start(1000)
code = app.exec_()
if state["play"] is not None:
    state["play"].wait(timeout=20)
sys.exit(0 if state.get("ok") else 1)
