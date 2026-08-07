"""What a narrower band would cost: tap count vs CPU vs latency.

Two ways to run the FIR:
  direct    np.convolve, what the code does today -- O(N * block)
  overlap   FFT overlap-save                      -- O(M log M), M ~ N + block

Realtime budget: 1024 samples at 48 kHz is 21.3 ms of audio, so anything
near that per block cannot keep up. Two filters run at once (the live
monitor and the player), and the rest of Friture needs the CPU too.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# This PyQt5 does not register Qt's own paths -- see friture.bat.
import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
import sys
import time

import numpy as np

FS = 48000.0
BLOCK = 1024
BLOCK_MS = 1000.0 * BLOCK / FS


def transition_width(n_taps):
    return 3.3 * FS / n_taps


def bench_direct(taps, blocks=40):
    tail = np.zeros(taps.size - 1, dtype=taps.dtype)
    x = np.random.default_rng(1).standard_normal(BLOCK)
    if np.iscomplexobj(taps):
        x = x + 0j
    start = time.perf_counter()
    for _ in range(blocks):
        buf = np.concatenate((tail, x))
        np.convolve(buf, taps, mode="valid")
        tail = buf[buf.size - tail.size:]
    return 1000.0 * (time.perf_counter() - start) / blocks


def bench_overlap(taps, blocks=40):
    n = taps.size
    size = 1
    while size < n + BLOCK - 1:
        size *= 2
    spectrum = np.fft.fft(taps, size)
    history = np.zeros(size, dtype=np.complex128 if np.iscomplexobj(taps) else np.float64)
    x = np.random.default_rng(1).standard_normal(BLOCK)
    if np.iscomplexobj(taps):
        x = x + 0j
    start = time.perf_counter()
    for _ in range(blocks):
        history = np.concatenate((history[BLOCK:], x))
        np.fft.ifft(np.fft.fft(history) * spectrum)[-BLOCK:]
    return 1000.0 * (time.perf_counter() - start) / blocks


print("block = %d samples = %.1f ms of audio at %.0f kHz\n" % (BLOCK, BLOCK_MS, FS / 1000))
print("%7s %11s %10s %11s %11s" % ("taps", "min band", "latency", "direct/blk", "overlap/blk"))
for n in (255, 511, 1023, 2047, 4095, 8191, 16383):
    taps = np.random.default_rng(0).standard_normal(n) + 1j * np.random.default_rng(2).standard_normal(n)
    latency_ms = 1000.0 * ((n - 1) / 2) / FS
    print("%7d %8.0f Hz %8.1f ms %8.2f ms %8.2f ms"
          % (n, transition_width(n), latency_ms, bench_direct(taps), bench_overlap(taps)))

print("\n(complex taps: the heterodyne path, which is the expensive one)")
print("anything at or above %.1f ms per block cannot keep up" % BLOCK_MS)
