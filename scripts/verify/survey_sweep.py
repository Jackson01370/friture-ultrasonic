"""Walk the Digital Decode dock across a list of bands, on screen, and report.

    python scripts\verify\survey_sweep.py OUT_DIR [--seconds N] [--bands lo-hi,lo-hi,...]

Made to be watched. The real application starts, its window is brought to
the front, and the dock is pointed at one band after another -- so the band
overlay slides across the spectrum and the spectrogram while the dock's
readout follows. Each band is shown in FM first (carrier, deviation,
modulation rate, and the demodulated waveform) and then in FSK (whether it
is keyed), which is the pair that answers "is there anything here".

Every capture block is kept, so afterwards each band is written as a WAV to
listen to and the verdicts are printed from the same audio the screen was
showing.

The microphone is exclusive: if another Friture is holding it this waits,
saying so, rather than failing.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("FRITURE_RECORDING_DIR", "off")   # never the user's recording folder
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")

import ctypes
import logging

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

logging.basicConfig(level=logging.WARNING)

args = sys.argv[1:]


def option(name, default):
    return args[args.index(name) + 1] if name in args else default


OUT = Path(args[0] if args and not args[0].startswith("-") else ROOT / "sandbox")
OUT.mkdir(parents=True, exist_ok=True)
PER_BAND = float(option("--seconds", "16"))
BANDS = [tuple(float(v) for v in b.split("-")) for b in option(
    "--bands", "100-1000,1000-2000,2000-3000,3000-4500,4500-6000,6000-8000,8000-10000").split(",")]


def mic_is_free():
    """The UltraMic, opened the way the app opens it. -9996 means held."""
    dev = None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] and "UltraMic" in d["name"] \
                and "WASAPI" in sd.query_hostapis(d["hostapi"])["name"]:
            dev = i
            break
    if dev is None:
        return False
    try:
        with sd.InputStream(device=dev, channels=1, samplerate=250000, blocksize=1024,
                            extra_settings=sd.WasapiSettings(exclusive=True)):
            pass
        return True
    except Exception:
        return False


print("waiting for the microphone (close any running Friture) ...", flush=True)
deadline = time.time() + 300
while not mic_is_free():
    if time.time() > deadline:
        print("still held after 5 minutes; nothing recorded.", flush=True)
        sys.exit(1)
    time.sleep(2.0)
print("the microphone is free.\n", flush=True)

app = QApplication(sys.argv)

from friture.analyzer import Friture
from friture.audiobackend import SAMPLING_RATE
from friture.demod.decoder import DIGITAL_MODES, FM, FSK, MODE_LABELS, BandDecoder
from friture.digital_decode import DigitalDecode_Widget
from friture.listen.listen_band_view_model import GetListenBand

FS = float(SAMPLING_RATE)
window = Friture()
window.show()
try:                                   # bring it to the front for the camera
    hwnd = int(window.winId())
    ctypes.windll.user32.ShowWindow(hwnd, 3)          # maximised
    ctypes.windll.user32.SetForegroundWindow(hwnd)
except Exception:
    pass

dock = None
for d in window.dockmanager.docks:
    if isinstance(d.audiowidget, DigitalDecode_Widget):
        dock = d
        break
if dock is None:
    window.dockmanager.new_dock()
    dock = window.dockmanager.docks[-1]
    dock.widget_select(9)
widget = dock.audiowidget
vm = widget.view_model()
band = GetListenBand()

captured = []
window.audiobuffer.new_data_available.connect(lambda fd: captured.append(fd[0, :].astype(np.float32)))

state = {"i": -1, "phase": "", "t": 0.0, "started": time.time(), "marks": []}


def set_band(lo, hi):
    band.set_width_hz(int(hi - lo))
    band.set_centre_hz(int((lo + hi) / 2))


def step():
    now = time.time() - state["started"]
    if state["i"] >= 0 and now - state["t"] < PER_BAND:
        print("   %5.1f s  %-9s %-62s" % (now, state["phase"], vm.signal_text[:62]), flush=True)
        return
    # move on: FM then FSK for each band, then the next band
    if state["phase"] == "FM":
        state["phase"] = "FSK"
        widget.set_mode(FSK)
    else:
        state["i"] += 1
        if state["i"] >= len(BANDS):
            finish()
            return
        lo, hi = BANDS[state["i"]]
        set_band(lo, hi)
        widget.set_mode(FM)
        state["phase"] = "FM"
        state["marks"].append((now, lo, hi))
        print("\n=== %.0f-%.0f Hz ===" % (lo, hi), flush=True)
    state["t"] = now


def finish():
    timer.stop()
    x = np.concatenate(captured).astype(np.float64) if captured else np.zeros(0)
    np.savez_compressed(OUT / "sweep.npz", x=x.astype(np.float32), fs=FS)
    print("\nsaved %s (%.1f s)\n" % (OUT / "sweep.npz", x.size / FS), flush=True)
    image = window.quick_view.grabWindow()
    image.save(str(OUT / "sweep_screen.png"))
    print("saved %s\n" % (OUT / "sweep_screen.png"), flush=True)

    print("verdict per band, from the same audio (every mode, whole capture):", flush=True)
    for lo, hi in BANDS:
        row = []
        for mode in DIGITAL_MODES:
            dec = BandDecoder(FS, mode=mode)
            dec.configure(lo, hi - lo)
            locked = 0
            for i in range(0, x.size, 2048):
                dec.process(x[i:i + 2048])
                locked += int(dec.decode_locked)
            row.append("%s %s" % (MODE_LABELS[mode], ("BITS %d" % locked) if locked else "-"))
        dec = BandDecoder(FS, mode=FM)
        dec.configure(lo, hi - lo)
        for i in range(0, x.size, 2048):
            dec.process(x[i:i + 2048])
        print("   %5.0f-%5.0f Hz  %s" % (lo, hi, " | ".join(row)), flush=True)
        print("                   FM: %s" % dec.signal_text()[:96], flush=True)
    QTimer.singleShot(8000, window.close)


timer = QTimer()
timer.timeout.connect(step)
timer.start(2000)
app.exec_()
