"""Measure the acoustic channel speakers -> room -> UltraMic before blaming the decoder.

1. a 14 kHz tone switched off: how long the envelope takes to fall 10/20/30 dB
   (the room's reverberation at that frequency -- an OOK or DBPSK symbol has
   to be much longer than this)
2. 12 kHz then 16 kHz: does each tone arrive, and does the FM demodulator
   follow the switch

The first playback after the capture starts is late or silent on this
machine's MME output, so a silent warm-up is played first. Samples are taken
from the backend's signal, block by block, as the docks get them.
"""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins")); os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))
import logging; logging.basicConfig(level=logging.CRITICAL)
import numpy as np, sounddevice as sd
from PyQt5.QtCore import QCoreApplication
from friture.audiobackend import AudioBackend, SAMPLING_RATE
from friture.demod.ddc import ComplexDdc
from friture.demod.detectors import AmDemodulator, FmDemodulator

PLAY_FS = 48000; AMP = 0.5
app = QCoreApplication(sys.argv)
backend = AudioBackend(); assert backend.device is not None, "no capture device"
backend.restart()

def capture(seconds):
    blocks = []
    def feed(fd): blocks.append(fd[0, :].astype(np.float64))
    backend.new_data_available.connect(feed)
    t0 = time.time()
    while time.time() - t0 < seconds:
        backend.fetchAudioData(); app.processEvents(); time.sleep(0.004)
    backend.new_data_available.disconnect(feed)
    return np.concatenate(blocks) if blocks else np.zeros(0)

def tone(f, seconds, fs=PLAY_FS):
    t = np.arange(int(seconds * fs)) / fs
    x = np.cos(2 * np.pi * f * t); r = np.linspace(0, 1, int(0.005 * fs)); x[:r.size] *= r; x[-r.size:] *= r[::-1]
    return (AMP * x).astype(np.float32)

print("warm-up: 1 s of silence through the output, then 1 s of capture", flush=True)
sd.play(np.zeros(PLAY_FS, np.float32), PLAY_FS, blocking=True)
capture(1.0)

# 1. decay after a 14 kHz tone stops
sig = np.concatenate((tone(14000.0, 1.0), np.zeros(int(1.5 * PLAY_FS), np.float32)))
sd.play(sig, PLAY_FS, blocking=False); cap = capture(2.8); sd.stop()
print("captured %.2f s" % (cap.size / SAMPLING_RATE), flush=True)
ddc = ComplexDdc(10000.0, 8000.0, SAMPLING_RATE, 5); env = AmDemodulator(smooth_taps=25).process(ddc.process(cap))
fs_bb = ddc.fs_out
env_db = 20 * np.log10(np.maximum(env, 1e-9))
peak = float(np.max(env_db)); on = np.flatnonzero(env_db > peak - 3.0)
off_edge = int(on[-1]) if on.size else 0
floor_db = float(np.median(env_db[-int(0.3 * fs_bb):]))
print("14 kHz tone: peak %.1f dBFS at %.2f s, tone ends at %.2f s, floor afterwards %.1f dB (%.1f dB of range)"
      % (peak, np.argmax(env_db) / fs_bb, off_edge / fs_bb, floor_db, peak - floor_db), flush=True)
for drop in (10, 20, 30, 40):
    below = np.flatnonzero(env_db[off_edge:] < peak - drop)
    print("   -%2d dB after the tone stops: %s" % (drop, ("%.1f ms" % (1e3 * below[0] / fs_bb)) if below.size else "never (floor too high)"), flush=True)

# 2. 12 kHz then 16 kHz, twice
sig = np.concatenate([tone(12000.0, 0.5), tone(16000.0, 0.5)] * 2)
sd.play(sig, PLAY_FS, blocking=False); cap = capture(2.4); sd.stop()
spec = np.abs(np.fft.rfft(cap * np.hanning(cap.size))) ** 2; freqs = np.fft.rfftfreq(cap.size, 1 / SAMPLING_RATE)
def lvl(lo, hi): return 10 * np.log10(np.mean(spec[(freqs >= lo) & (freqs < hi)]))
ref = lvl(30000, 60000)
print("levels over the 30-60 kHz floor: 12 kHz %.1f dB, 16 kHz %.1f dB (14 kHz, not played: %.1f dB)" % (lvl(11900, 12100) - ref, lvl(15900, 16100) - ref, lvl(13900, 14100) - ref), flush=True)
ddc = ComplexDdc(10000.0, 8000.0, SAMPLING_RATE, 5); z = ddc.process(cap); f, _ = FmDemodulator(ddc.fs_out).process(z)
win = int(0.05 * ddc.fs_out)
print("FM trace, median per 50 ms (kHz):", " ".join("%.1f" % ((np.median(f[k:k + win]) + 10000) / 1e3) for k in range(0, f.size - win, win)), flush=True)
# how fast the FM trace settles on the new tone after a switch: samples within +-500 Hz of 16 kHz after the first 12->16 edge
f_abs = f + 10000.0
sw = np.flatnonzero((np.abs(f_abs[:-1] - 12000) < 500) & (np.abs(f_abs[1:] - 16000) < 500))
if sw.size:
    k = int(sw[0]); seg = f_abs[k:k + int(0.02 * ddc.fs_out)]
    good = np.abs(seg - 16000) < 1000
    print("after a 12->16 kHz switch: %d%% of the next 20 ms is within 1 kHz of 16 kHz; first 5 ms: %d%%" % (round(100 * good.mean()), round(100 * good[:int(0.005 * ddc.fs_out)].mean())), flush=True)
backend.close()
print("done", flush=True)
