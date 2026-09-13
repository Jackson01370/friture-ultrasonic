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


def on_data(floatdata):
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

print("%-22s %-11s %-9s %s" % ("run", "level dBFS", "score", "what the detector says"))
silence = capture(seconds)
ev, silence_level = analyse(silence)
if ev is None:
    print("FAIL: captured nothing")
    raise SystemExit(1)
floor = ev.score
lvl = silence_level
print("%-22s %-11.1f %-9.2f %s" % ("the room alone", lvl, ev.score, ev.describe()))
if save_dir:
    np.savez_compressed(save_dir / "silence.npz", x=silence, fs=float(SAMPLING_RATE))

results = []
for db in levels:
    take = play[:int(seconds * PLAY_FS)]
    rec = capture(seconds, take * (10.0 ** (db / 20.0)))
    ev, lvl = analyse(rec)
    results.append((db, ev.score, lvl))
    verdict = "FOUND" if ev.score > floor * 1.5 else "not found"
    print("%-22s %-11.1f %-9.2f %-10s %s"
          % ("speech at %+.0f dB" % db, lvl, ev.score, verdict,
             "pitch %.0f Hz" % ev.median_f0_hz if np.isfinite(ev.median_f0_hz) else ""))
    if save_dir:
        np.savez_compressed(save_dir / ("speech_%+03d.npz" % db), x=rec, fs=float(SAMPLING_RATE))

backend.close()
print("\nthe room alone scored %.2f; a run counts as found above %.2f" % (floor, floor * 1.5))
found = [db for db, sc, _ in results if sc > floor * 1.5]
if found:
    print("quietest playback still found: %+.0f dB, captured at %.1f dBFS"
          % (min(found), [l for d, s, l in results if d == min(found)][0]))
    quietest = [l for d, s, l in results if d == min(found)][0]
    print("the room alone sat at %.1f dBFS, the capture at %.1f dBFS"
          % (silence_level, quietest))
else:
    print("nothing was found at any level -- check the speakers are on and unmuted")
ok = results and max(sc for _, sc, _ in results) > floor
print("\n%s" % ("OK" if ok else "FAIL: silence scored as high as the loudest speech"))
raise SystemExit(0 if ok else 1)
