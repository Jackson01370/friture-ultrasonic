"""Does the UltraMic really deliver 250 kHz, and through which host API?

check_input_settings() is not evidence: MME and DirectSound accept almost any
rate and quietly resample. The test that counts is to open the stream, take
real samples, and look at what is in them:

  - the sample count over a known wall-clock interval says whether the rate
    the device reports is the rate it is running at
  - energy above 24 kHz says the samples are genuinely wideband rather than
    48 kHz content stretched to look like 250 kHz. Upsampled audio has a
    brick wall where the original Nyquist was; real capture does not.
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

RATES = (48000, 96000, 192000, 250000)
SECONDS = 1.5


def band_energy(x, fs, lo, hi):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    inside = (freqs >= lo) & (freqs < hi)
    return float(np.sum(spectrum[inside] ** 2))


def try_rate(index, name, channels, fs):
    try:
        start = time.perf_counter()
        data = sd.rec(int(SECONDS * fs), samplerate=fs, channels=channels,
                      device=index, dtype="float32", blocking=True)
        elapsed = time.perf_counter() - start
    except Exception as exc:
        return "open failed: %s" % str(exc).splitlines()[0][:58]

    x = data[:, 0].astype(np.float64)
    if not np.any(x):
        return "opened, but every sample is zero"

    effective = x.size / elapsed
    rms = float(np.sqrt(np.mean(x ** 2)))

    note = "%7.0f Hz asked, %7.0f Hz of samples arrived, rms %.5f" % (fs, effective, rms)
    if fs > 48000:
        low = band_energy(x, fs, 1000.0, 20000.0)
        high = band_energy(x, fs, 30000.0, min(fs / 2.0, 120000.0) - 1000.0)
        ratio = high / low if low > 0 else 0.0
        note += " | >30kHz vs 1-20kHz = %.4f %s" % (
            ratio, "(wideband)" if ratio > 1e-3 else "(BRICK WALL -> resampled 48k)")
    return note


targets = []
for index, device in enumerate(sd.query_devices()):
    if device["max_input_channels"] > 0 and "Ultra" in device["name"]:
        api = sd.query_hostapis(device["hostapi"])["name"]
        targets.append((index, device["name"], api, device["max_input_channels"]))

if not targets:
    print("UltraMic not found")
    sys.exit(1)

for index, name, api, channels in targets:
    print("\n#%d [%s] ch=%d" % (index, api, channels))
    for fs in RATES:
        print("   %s" % try_rate(index, name, channels, fs))
