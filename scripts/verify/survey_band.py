"""Investigate one band in the running application, and say what it is.

    python scripts\verify\survey_band.py CENTRE_HZ [WIDTH_HZ] [--seconds N] [--out DIR]

Starts the real Friture -- window, dock manager, audio backend, QML -- with a
Digital Decode dock pointed at the band, and for the duration:

  * the dock runs in FM, so the screen shows the demodulated waveform and,
    if there is a carrier, its frequency, deviation and modulation rate;
  * a second set of decoders runs on the same blocks in every other mode, so
    the log says whether the band is keyed (OOK / FSK / 4-FSK / DBPSK /
    DQPSK) as well as whether it is analog-modulated;
  * the raw capture is kept and saved, so the verdict can be re-checked
    without the microphone.

At the end it grabs the QML scene as a PNG (desktop screen grabs of a Qt
Quick window come back blank on this machine) and prints a verdict: what the
band's level does, whether any mode ever locked, and the strongest lines in
the band's envelope and instantaneous frequency -- the two places a
modulation would show.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# This PyQt5 does not register Qt's own paths -- see friture.bat.
import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")

import logging

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

logging.basicConfig(level=logging.WARNING)

args = sys.argv[1:]


def option(name, default):
    return args[args.index(name) + 1] if name in args else default


CENTRE = float(args[0]) if args and not args[0].startswith("-") else 5800.0
WIDTH = float(args[1]) if len(args) > 1 and not args[1].startswith("-") else 1000.0
SECONDS = float(option("--seconds", "40"))
OUT = Path(option("--out", str(ROOT / "sandbox")))
OUT.mkdir(parents=True, exist_ok=True)

app = QApplication(sys.argv)

from friture.audiobackend import SAMPLING_RATE
from friture.analyzer import Friture
from friture.demod.decoder import DIGITAL_MODES, FM, MODE_LABELS, MODES, BandDecoder
from friture.demod.ddc import ComplexDdc
from friture.demod.detectors import AmDemodulator, FmDemodulator
from friture.digital_decode import DigitalDecode_Widget
from friture.listen.listen_band_view_model import GetListenBand

FS = float(SAMPLING_RATE)
F_LO = CENTRE - WIDTH / 2.0
print("band %.0f - %.0f Hz (centre %.0f, width %.0f), %.0f s"
      % (F_LO, F_LO + WIDTH, CENTRE, WIDTH, SECONDS), flush=True)

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
widget = decode_dock.audiowidget
widget.set_mode(FM)
band = GetListenBand()
band.set_width_hz(int(WIDTH))
band.set_centre_hz(int(CENTRE))
vm = widget.view_model()

# the other modes, on the same blocks the dock gets
others = {}
for mode in MODES:
    if mode == FM:
        continue
    dec = BandDecoder(FS, mode=mode)
    dec.configure(F_LO, WIDTH)
    others[mode] = dec
locked_ever = {mode: 0 for mode in MODES}
best_conf = {mode: 0.0 for mode in MODES}
carrier_blocks = 0
n_blocks = 0
captured = []

state = {"t0": time.time(), "done": False}


def on_block(floatdata):
    global n_blocks, carrier_blocks
    x = floatdata[0, :]
    captured.append(x.astype(np.float32))
    n_blocks += 1
    for mode, dec in others.items():
        dec.process(x)
        if dec.decode_locked:
            locked_ever[mode] += 1
        best_conf[mode] = max(best_conf[mode], dec.last_baud_conf_db)
    if widget.decoder.analog_carrier:
        carrier_blocks += 1
    best_conf[FM] = max(best_conf[FM], 0.0)


window.audiobuffer.new_data_available.connect(on_block)


def lines(sig, fs_bb, name, unit, count=4):
    """The strongest periodic components of a demodulated waveform."""
    s = np.asarray(sig, dtype=np.float64)
    s = s - float(np.mean(s))
    n = 1 << int(np.floor(np.log2(max(s.size, 2))))
    if n < 1024:
        return
    s = s[-n:]
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(s * w))
    f = np.fft.rfftfreq(n, 1.0 / fs_bb)
    sel = (f >= 0.5) & (f <= 2000.0)
    ss, ff = spec[sel], f[sel]
    floor = float(np.median(ss))
    print("  %s, strongest lines 0.5-2000 Hz:" % name, flush=True)
    picked = []
    for j in np.argsort(ss)[::-1]:
        f0 = float(ff[j])
        if any(abs(f0 - p) < 2.0 for p in picked):
            continue
        picked.append(f0)
        amp = np.sqrt(4.0 * float(ss[j]) ** 2 / (n * float(np.sum(w ** 2))))
        print("     %8.2f Hz  %5.1f dB over the floor   amplitude ~%.4g %s"
              % (f0, 20 * np.log10(float(ss[j]) / max(floor, 1e-30)), amp, unit), flush=True)
        if len(picked) >= count:
            break


def verdict():
    x = np.concatenate(captured).astype(np.float64) if captured else np.zeros(0)
    npz = OUT / ("survey_%.0fHz.npz" % CENTRE)
    np.savez_compressed(npz, x=x.astype(np.float32), fs=FS, f_lo=F_LO, width=WIDTH)
    print("\nsaved the capture: %s (%.1f s)" % (npz, x.size / FS), flush=True)

    print("\nthe band, second by second (level of %.0f-%.0f Hz, and its median frequency):"
          % (F_LO, F_LO + WIDTH), flush=True)
    ddc = ComplexDdc(F_LO, WIDTH, FS, 5)
    z = ddc.process(x)
    fs_bb = ddc.fs_out
    env = AmDemodulator(smooth_taps=50).process(z).astype(np.float64)
    freq = FmDemodulator(fs_bb).process(z)[0].astype(np.float64) + F_LO
    per = int(fs_bb)
    levels = []
    for s in range(int(env.size // per)):
        e = env[s * per:(s + 1) * per]
        f = freq[s * per:(s + 1) * per]
        lv = 20 * np.log10(max(float(np.sqrt(np.mean(e ** 2))), 1e-12))
        levels.append(lv)
        if s < 30:
            print("   %2d s  %7.1f dBFS   median %8.1f Hz   1-99%% spread %7.1f Hz"
                  % (s, lv, float(np.median(f)), float(np.percentile(f, 99) - np.percentile(f, 1))), flush=True)
    if levels:
        lv = np.array(levels)
        print("   level over the run: mean %.1f dBFS, spread %.1f dB (min %.1f, max %.1f)"
              % (lv.mean(), lv.max() - lv.min(), lv.min(), lv.max()), flush=True)

    print("\nis anything modulating it?", flush=True)
    lines(env, fs_bb, "envelope (an AM or keyed signal shows here)", "")
    lines(freq, fs_bb, "instantaneous frequency (an FM or FSK signal shows here)", "Hz")

    print("\ndid any mode ever produce bits, in %d blocks?" % n_blocks, flush=True)
    for mode in DIGITAL_MODES:
        dec = others[mode]
        print("   %-6s locked %4d blocks, best clock confidence %4.1f dB   %s"
              % (MODE_LABELS[mode], locked_ever[mode], best_conf[mode], dec.decode_text()), flush=True)
    for mode in MODES:
        if mode in DIGITAL_MODES:
            continue
        dec = widget.decoder if mode == FM else others[mode]
        print("   %-6s %s" % (MODE_LABELS[mode], dec.signal_text()), flush=True)
    print("   FM carrier present in %d of %d blocks" % (carrier_blocks, n_blocks), flush=True)

    image = window.quick_view.grabWindow()
    png = OUT / ("survey_%.0fHz.png" % CENTRE)
    image.save(str(png))
    print("\nsaved the screen: %s (%dx%d)" % (png, image.width(), image.height()), flush=True)


def tick():
    t = time.time() - state["t0"]
    print("%5.1f s | %-72s | %d points" % (t, vm.signal_text[:72], len(vm.trace)), flush=True)
    if t >= SECONDS and not state["done"]:
        state["done"] = True
        verdict()
        QTimer.singleShot(300, window.close)


timer = QTimer()
timer.timeout.connect(tick)
timer.start(2000)
app.exec_()
