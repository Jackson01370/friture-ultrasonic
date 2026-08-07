"""Capture through Friture's own path and check what actually arrives.

The point is not that a stream opens -- MME opens too. It is that the samples
Friture's ring buffer ends up holding contain energy above 24 kHz, which
upsampled 48 kHz audio cannot. Then the same samples go out through the
listening chain to confirm the audible end still produces something.
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
import logging
import sys
import time

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QCoreApplication

logging.basicConfig(level=logging.WARNING)

from friture.audiobackend import AudioBackend, OUTPUT_SAMPLING_RATE, SAMPLING_RATE
from friture.audiobuffer import AudioBuffer
from friture.listen.band_dsp import BANDPASS, HETERODYNE
from friture.listen.listen_band_view_model import GetListenBand
from friture.listen.playout import Playout
from friture.listen.processor import BandProcessor

app = QCoreApplication(sys.argv)

backend = AudioBackend()
if backend.device is None:
    print("FAIL: no capture device opened")
    sys.exit(1)

api = sd.query_hostapis(backend.device["hostapi"])["name"]
print("capturing from %s [%s] at %d Hz"
      % (backend.device["name"][:34], api, SAMPLING_RATE))

buffer = AudioBuffer()
AudioBackend().new_data_available.connect(buffer.handle_new_data)

# What the app does when Start is pressed: __init__ opens the stream, but only
# restart() sets up the read bookkeeping fetchAudioData needs.
backend.restart()

print("\nrecording 3 s -- make some broadband noise (jingle keys, hiss, a clap)")
deadline = time.time() + 3.0
while time.time() < deadline:
    backend.fetchAudioData()
    app.processEvents()
    time.sleep(0.005)

captured = buffer.data(int(2.0 * SAMPLING_RATE))[0].astype(np.float64)
print("ring buffer holds %d samples (%.2f s)" % (captured.size, captured.size / SAMPLING_RATE))

ok = True


def check(label, condition, detail):
    global ok
    print("%-44s %s   %s" % (label, "ok  " if condition else "FAIL", detail))
    ok = ok and condition


spectrum = np.abs(np.fft.rfft(captured * np.hanning(captured.size)))
freqs = np.fft.rfftfreq(captured.size, 1.0 / SAMPLING_RATE)


def energy(lo, hi):
    band = (freqs >= lo) & (freqs < hi)
    return float(np.sum(spectrum[band] ** 2))


audible = energy(1000.0, 20000.0)
ultrasonic = energy(30000.0, 120000.0)
ratio = ultrasonic / audible if audible > 0 else 0.0

check("something was captured at all", float(np.sqrt(np.mean(captured ** 2))) > 1e-6,
      "rms %.5f" % np.sqrt(np.mean(captured ** 2)))
check("the band really extends past 24 kHz", ratio > 1e-3,
      ">30 kHz vs 1-20 kHz = %.5f  (a brick wall would read 0.00000)" % ratio)

# where the energy actually is, decade by decade -- a resampled stream would
# be empty in the top two rows
print("\n  energy by band:")
for lo, hi in ((1000, 20000), (20000, 40000), (40000, 60000), (60000, 90000), (90000, 120000)):
    share = energy(lo, hi) / (audible + 1e-30)
    print("    %3d-%3d kHz  %8.5f %s" % (lo // 1000, hi // 1000, share, "#" * min(40, int(share * 200))))

# now the listening chain: heterodyne a 10 kHz slice down and resample out
band = GetListenBand()
band.enabled = True
band.mode = HETERODYNE
band.width_hz = 10000
band.click_center(45000.0)

processor = BandProcessor(None, band)
playout = Playout(1 << 16)
for i in range(0, captured.size - 2048, 2048):
    playout.push(processor.process(captured[i:i + 2048]))

heard = np.zeros(playout.available, dtype=np.float32)
if heard.size:
    playout.pop_into(heard)
check("the listening chain produced playable audio", heard.size > 0.5 * OUTPUT_SAMPLING_RATE,
      "%d samples at %d Hz (%.2f s)" % (heard.size, OUTPUT_SAMPLING_RATE, heard.size / OUTPUT_SAMPLING_RATE))
check("what comes out is inside the audible range", heard.size > 0 and float(np.max(np.abs(heard))) <= 1.001,
      "peak %.4f, rms %.5f" % (np.max(np.abs(heard)) if heard.size else 0,
                               np.sqrt(np.mean(heard ** 2)) if heard.size else 0))

# and the honest negative: band-pass up there has to come out silent
band.mode = BANDPASS
silent_processor = BandProcessor(None, band)
silent_playout = Playout(1 << 16)
for i in range(0, captured.size - 2048, 2048):
    silent_playout.push(silent_processor.process(captured[i:i + 2048]))
silent = np.zeros(silent_playout.available, dtype=np.float32)
if silent.size:
    silent_playout.pop_into(silent)
loud = float(np.sqrt(np.mean(heard ** 2))) if heard.size else 0.0
quiet = float(np.sqrt(np.mean(silent ** 2))) if silent.size else 0.0
check("band-pass at 45 kHz is silent, as warned", quiet < 0.05 * max(loud, 1e-9),
      "heterodyne rms %.5f vs band-pass rms %.5f" % (loud, quiet))

backend.close()
sys.exit(0 if ok else 1)
