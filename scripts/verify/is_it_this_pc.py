"""Does a band come from this PC's own fans? Load the CPU and see if it follows.

    python scripts\verify\is_it_this_pc.py OUT_DIR [--idle N] [--load N]

A mystery signal that is really the machine you are measuring with is the
easiest one to chase for hours. The test takes two minutes: record with the
CPU idle, record again with every core busy, and compare. Fan noise tracks
the load within seconds and changes its blade tones; anything outside the
room does not care what this PC is doing.

Both captures are saved, and the comparison is printed per band and for the
strongest individual lines.
"""

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
import logging
logging.basicConfig(level=logging.CRITICAL)

import numpy as np
from PyQt5.QtCore import QCoreApplication

BURN = (
    "import time, sys\n"
    "end = time.time() + float(sys.argv[1])\n"
    "x = 0.0\n"
    "while time.time() < end:\n"
    "    for i in range(200000):\n"
    "        x += i * i\n"
)

args = sys.argv[1:]
out = Path(args[0] if args and not args[0].startswith("-") else ".")
out.mkdir(parents=True, exist_ok=True)


def option(name, default):
    return float(args[args.index(name) + 1]) if name in args else default


idle_s = option("--idle", 70.0)
load_s = option("--load", 45.0)
KEEP = 20.0                      # the last this many seconds of each are compared

app = QCoreApplication(sys.argv)
from friture.audiobackend import AudioBackend, SAMPLING_RATE
FS = float(SAMPLING_RATE)

backend = AudioBackend()
assert backend.device is not None, "no capture device"
backend.restart()


def record(seconds, note):
    blocks = []
    handler = lambda fd: blocks.append(fd[0, :].astype(np.float32))
    backend.new_data_available.connect(handler)
    print("%s: %.0f s ..." % (note, seconds), flush=True)
    t0 = time.time()
    while time.time() - t0 < seconds:
        backend.fetchAudioData()
        app.processEvents()
        time.sleep(0.004)
    backend.new_data_available.disconnect(handler)
    x = np.concatenate(blocks).astype(np.float64) if blocks else np.zeros(0)
    return x[-int(KEEP * FS):]


# idle first: the capture loop itself is light, so the fans settle during it
idle = record(idle_s, "idle (nothing running but the capture)")
n_cpu = os.cpu_count() or 4
burners = [subprocess.Popen([sys.executable, "-c", BURN, str(load_s + 8)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
           for _ in range(n_cpu)]
print("started %d CPU burners" % n_cpu, flush=True)
loaded = record(load_s, "loaded (every core busy)")
for b in burners:
    b.terminate()
backend.close()

np.savez_compressed(out / "idle.npz", x=idle.astype(np.float32), fs=FS)
np.savez_compressed(out / "loaded.npz", x=loaded.astype(np.float32), fs=FS)
print("saved %s and %s (%.0f s each)\n" % (out / "idle.npz", out / "loaded.npz", KEEP), flush=True)


def psd(x, nf=1 << 16):
    acc = np.zeros(nf // 2 + 1)
    c = 0
    for i in range(0, x.size - nf, nf // 2):
        acc += np.abs(np.fft.rfft(x[i:i + nf] * np.hanning(nf))) ** 2
        c += 1
    return acc / max(c, 1)


pi, pl = psd(idle), psd(loaded)
f = np.fft.rfftfreq(1 << 16, 1.0 / FS)
print("band level, idle -> loaded:")
for a, b in ((60, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 4500),
             (4500, 10000), (10000, 30000), (30000, 120000)):
    s = (f >= a) & (f < b)
    x0 = 10 * np.log10(max(float(pi[s].mean()), 1e-30))
    x1 = 10 * np.log10(max(float(pl[s].mean()), 1e-30))
    flag = "  <- follows the CPU" if abs(x1 - x0) > 6 else ""
    print("   %6d-%6d Hz  %6.1f -> %6.1f dB  (%+5.1f)%s" % (a, b, x0, x1, x1 - x0, flag))

print("\nthe strongest lines under 2 kHz, idle -> loaded:")
db_l = 10 * np.log10(np.maximum(pl, 1e-30))
db_i = 10 * np.log10(np.maximum(pi, 1e-30))
sel = (f >= 100) & (f <= 2000)
idx = np.flatnonzero(sel)
half = max(1, int(250.0 / f[1]))
local = np.array([np.median(db_l[max(i - half, 0):min(i + half + 1, db_l.size)]) for i in idx])
excess = db_l[idx] - local
picked = []
for j in np.argsort(excess)[::-1]:
    f0 = float(f[idx][j])
    if any(abs(f0 - p) < 40.0 for p in picked):
        continue
    picked.append(f0)
    print("   %7.1f Hz  %6.1f -> %6.1f dB  (%+5.1f)" % (f0, db_i[idx][j], db_l[idx][j], db_l[idx][j] - db_i[idx][j]))
    if len(picked) >= 10:
        break
