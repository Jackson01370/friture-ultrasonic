"""The same sensitivity question, but through real air instead of arithmetic.

    this PC speakers -> the room -> UltraMic 250K -> Friture capture path
    -> VoiceDetector

    python scripts\verify\voice_over_the_air.py SPEECH.wav [--seconds S]
                                                [--levels DB,DB,...] [--save DIR]

Mixing speech into a saved capture measures the DETECTOR. It does not
measure the microphone, the room, the speakers, or what reverberation does
to a pitch track -- and in this project every real defect turned up over the
air and none of them in simulation. So the same speech is played into the
room at a series of known attenuations and captured for real.

The silence run comes first and its required answer is "nothing": if the
room alone scores like a voice, no number after it means anything.

Needs the speakers on, the UltraMic free, and a quiet room. Exit status 1 if
silence scores above the loudest speech run.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
os.environ.setdefault("QML2_IMPORT_PATH", str(_qt / "qml"))

import logging
import time

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QCoreApplication, QTimer
from scipy.io import wavfile
from scipy.signal import resample_poly

from friture.audiobackend import SAMPLING_RATE, AudioBackend
from friture.demod.speech import VoiceDetector

logging.basicConfig(level=logging.WARNING)

ANALYSIS_FS = 16000.0
PLAY_FS = 48000

args = sys.argv[1:]
speech_path = args[0]


def option(name, default):
    return args[args.index(name) + 1] if name in args else default


seconds = float(option("--seconds", 25.0))
levels = [float(v) for v in option("--levels", "0,-10,-20,-30,-40").split(",")]
save_dir = Path(option("--save", "")) if "--save" in args else None
if save_dir:
    save_dir.mkdir(parents=True, exist_ok=True)

fs_w, wav = wavfile.read(speech_path)
if wav.ndim > 1:
    wav = wav.mean(axis=1)
wav = wav.astype(np.float64) / 32768.0
if abs(fs_w - 22050) < 1:
    play = resample_poly(wav, 320, 147)      # 22050 * 320/147 = 48000 exactly
elif abs(fs_w - PLAY_FS) < 1:
    play = wav
else:
    raise SystemExit("no exact ratio from %g Hz to %d Hz" % (fs_w, PLAY_FS))
play = play / max(np.abs(play).max(), 1e-12) * 0.9

app = QCoreApplication(sys.argv)
backend = AudioBackend()
if backend.device is None:
    print("FAIL: no capture device (is another process holding the UltraMic?)")
    raise SystemExit(1)
api = sd.query_hostapis(backend.device["hostapi"])["name"]
print("capturing from %s [%s] at %d Hz" % (backend.device["name"][:34], api, SAMPLING_RATE))
print("playing through %s at %d Hz" % (sd.query_devices(sd.default.device[1])["name"][:40], PLAY_FS))
print("speech %.1f s available, runs of %.0f s\n" % (play.size / PLAY_FS, seconds))

detector = VoiceDetector(ANALYSIS_FS)
chunks = []


def on_data(floatdata, *_):
    chunks.append(np.array(floatdata[0, :], dtype=np.float32))


backend.new_data_available.connect(on_data)
backend.restart()


def capture(duration, signal=None):
    """Record for duration seconds, optionally playing signal at the same time."""
    chunks.clear()
    if signal is not None:
        sd.play(signal.astype(np.float32), PLAY_FS, blocking=False)
    t_end = time.time() + duration
    while time.time() < t_end:
        # the backend is PULLED: nothing arrives unless it is asked for,
        # which is what the display timer does inside the app
        backend.fetchAudioData()
        app.processEvents()
        time.sleep(0.002)
    sd.stop()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def analyse(x):
    if x.size < SAMPLING_RATE:
        return None, 0.0
    band = resample_poly(x.astype(np.float64), 16, 250)
    return detector.score(band), float(20 * np.log10(max(band.std(), 1e-12)))


# the speaker only wakes up properly after something has been sent to it
sd.play(np.zeros(PLAY_FS // 2, dtype=np.float32), PLAY_FS, blocking=True)
capture(1.0)                      # drain whatever queued up while starting

# CAPTURE EVERYTHING FIRST, ANALYSE AFTERWARDS. Scoring between runs let the
# ring buffer fall a second behind and the next run came back empty.
SEG = 5.0                         # seconds per scored segment
print("recording %.0f s of the room alone, then %d playback levels ..."
      % (2 * seconds, len(levels)), flush=True)
silence = capture(2 * seconds)
runs = []
for db in levels:
    take = play[:int(seconds * PLAY_FS)]
    runs.append((db, capture(seconds, take * (10.0 ** (db / 20.0)))))
backend.close()
if silence.size < SAMPLING_RATE or any(r.size < SAMPLING_RATE for _, r in runs):
    print("FAIL: a run captured nothing")
    raise SystemExit(1)
if save_dir:
    np.savez_compressed(save_dir / "silence.npz", x=silence, fs=float(SAMPLING_RATE))
    for db, rec in runs:
        np.savez_compressed(save_dir / ("speech_%+03d.npz" % db), x=rec, fs=float(SAMPLING_RATE))


def segments(x):
    band = resample_poly(x.astype(np.float64), 16, 250)
    n = int(SEG * ANALYSIS_FS)
    return [band[i:i + n] for i in range(0, band.size - n + 1, n)]


def power(x):
    return float(np.mean(resample_poly(x.astype(np.float64), 16, 250) ** 2))


# the null: every 5 s of silence, and the threshold is the worst of them --
# a run segment counts only if it beats everything the room did on its own
null = np.array([detector.score(s).score for s in segments(silence)])
threshold = float(null.max())
p_room = power(silence)
print()
print("the room alone: %d segments of %.0f s, scores %.2f .. %.2f  (threshold = the worst, %.2f)"
      % (null.size, SEG, null.min(), null.max(), threshold))
print()
print("%-16s %-13s %-17s %s" % ("playback", "SNR in air", "segments found", "pitch it heard"))
results = []
for db, rec in runs:
    evs = [detector.score(s) for s in segments(rec)]
    hits = sum(e.score > threshold for e in evs)
    ratio = power(rec) / p_room - 1.0
    snr = 10 * np.log10(ratio) if ratio > 1e-3 else float("-inf")
    f0 = [e.median_f0_hz for e in evs if e.score > threshold and np.isfinite(e.median_f0_hz)]
    results.append((db, snr, hits, len(evs)))
    print("%-16s %-13s %2d of %-2d  %4.0f%%   %s"
          % ("%+.0f dB" % db, "%+.1f dB" % snr if np.isfinite(snr) else "unmeasurable",
             hits, len(evs), 100.0 * hits / len(evs),
             "%.0f Hz" % np.median(f0) if f0 else "-"))

print()
print("SNR in air = the speech band during playback over the same band in silence,")
print("so it is on the same axis as voice_sensitivity.py (the synthetic test).")
ok = max(h / n for _, _, h, n in results) >= 0.8
print()
print("OK" if ok else "FAIL: even the loudest playback was not found")
raise SystemExit(0 if ok else 1)
