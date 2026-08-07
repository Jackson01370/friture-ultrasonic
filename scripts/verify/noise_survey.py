"""What is actually in the noise, so the fix can be chosen rather than guessed.

Three questions decide which kind of noise reduction is worth building:

  is it flat?      broadband hiss is the microphone's own floor. Spectral
                   subtraction can trade it for artefacts; nothing removes it.

  are there tones? a narrow spike is interference -- ultrasonic motion
                   sensors, switching supplies, pest repellers. A notch takes
                   those out completely and costs nothing.

  is it steady?    stationary noise can be profiled and subtracted. Noise that
                   moves cannot, and only a gate helps.
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
import sounddevice as sd


FS = 250000
SECONDS = 4.0

index = [i for i, d in enumerate(sd.query_devices())
         if d["max_input_channels"] > 0 and "Ultra" in d["name"]
         and "WASAPI" in sd.query_hostapis(d["hostapi"])["name"]][0]

print("recording %.0f s of room tone at %d Hz -- stay quiet" % (SECONDS, FS))
data = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1, device=index,
              dtype="float32", blocking=True,
              extra_settings=sd.WasapiSettings(exclusive=True))
x = data[:, 0].astype(np.float64)
print("captured %d samples, rms %.6f (%.1f dBFS)\n"
      % (x.size, np.sqrt(np.mean(x ** 2)), 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-20)))

# Welch-ish: average many frames so the floor is smooth and tones stand out
frame = 8192
hop = frame // 2
window = np.hanning(frame)
frames = [np.abs(np.fft.rfft(x[i:i + frame] * window)) ** 2
          for i in range(0, x.size - frame, hop)]
psd = np.mean(frames, axis=0)
freqs = np.fft.rfftfreq(frame, 1.0 / FS)
db = 10 * np.log10(psd + 1e-30)
db -= db.max()

print("noise floor shape (relative to its own peak):")
for lo, hi in ((0, 5), (5, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 125)):
    band = (freqs >= lo * 1000) & (freqs < hi * 1000)
    print("  %3d-%3d kHz  median %6.1f dB   max %6.1f dB"
          % (lo, hi, np.median(db[band]), db[band].max()))

# tones: bins standing well above a running median of their neighbourhood
width = 201
padded = np.pad(db, width // 2, mode="edge")
local = np.array([np.median(padded[i:i + width]) for i in range(db.size)])
excess = db - local

print("\nnarrowband interference (>10 dB above the local floor):")
peaks = []
i = 1
while i < excess.size - 1:
    if excess[i] > 10 and excess[i] >= excess[i - 1] and excess[i] > excess[i + 1]:
        peaks.append((freqs[i], excess[i], db[i]))
        i += 20
    i += 1
if peaks:
    for f, over, level in sorted(peaks, key=lambda p: -p[1])[:12]:
        print("  %8.0f Hz   +%5.1f dB above floor   (%.1f dB)" % (f, over, level))
else:
    print("  none -- the floor is smooth, so there is nothing a notch would fix")

# stationarity: how much the per-band level wanders between halves
half = len(frames) // 2
first = 10 * np.log10(np.mean(frames[:half], axis=0) + 1e-30)
second = 10 * np.log10(np.mean(frames[half:], axis=0) + 1e-30)
print("\nis it steady? (level difference between the two halves of the recording)")
for lo, hi in ((20, 40), (40, 60), (60, 100)):
    band = (freqs >= lo * 1000) & (freqs < hi * 1000)
    drift = float(np.median(second[band] - first[band]))
    print("  %3d-%3d kHz  %+.2f dB %s" % (lo, hi, drift,
                                          "(steady)" if abs(drift) < 1.0 else "(moving)"))
