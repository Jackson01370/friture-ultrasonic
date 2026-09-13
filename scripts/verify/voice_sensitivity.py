"""How far below the noise can a voice be and still be found? Measure it.

    python scripts\verify\voice_sensitivity.py CAPTURE.npz SPEECH.wav [--trials N]
                                               [--seconds S] [--band LO,HI]

A detector that has not been characterised cannot support a negative
result: "nothing was found" only means something once you know what WOULD
have been found. So this buries KNOWN speech in the REAL noise of a real
capture at a series of known levels, and reports the level at which the
detector stops finding it.

Two things make the number honest.

  THE NOISE IS THIS ROOM, NOT WHITE NOISE. Every false positive this
  project has ever hit came from real room noise and none from synthetic
  noise, so a sensitivity measured against white noise would be a
  measurement of nothing.

  THE THRESHOLD IS SET BY THE NULL, NOT BY TASTE. Noise-only segments of
  the same capture are scored first; the threshold is the level that lets
  through FALSE_POSITIVE_RATE of them. Detection rate is then counted at
  that threshold, so the sensitivity quoted is at a stated false-alarm
  rate rather than at a flattering one.

SNR is defined over speech-ACTIVE frames only -- the pauses between
sentences are part of speech, but counting their silence as signal would
report a number several dB better than the truth.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from friture.demod.speech import VoiceDetector

FS = 16000.0                 # the rate the detector runs at
FALSE_POSITIVE_RATE = 0.05   # what the threshold is set to allow through
NULL_TRIALS = 120            # noise-only segments used to set that threshold
SNR_DB = [10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40]

args = sys.argv[1:]
capture_path = args[0]
speech_path = args[1]


def option(name, default):
    return args[args.index(name) + 1] if name in args else default


trials = int(option("--trials", 24))
seconds = float(option("--seconds", 20.0))
band = [float(v) for v in option("--band", "80,7800").split(",")]

rng = np.random.default_rng(20260913)
detector = VoiceDetector(FS)


def per_frame_score(x):
    """The same evidence with the TIME thrown away, as a control.

    Best lag in each frame, averaged over frames: exactly the harmonicity a
    per-frame test measures, with no path, no continuity and no duration.
    Running it beside the real detector is the only way to say what the
    Markov part is worth rather than assuming it is worth something.
    """
    r = detector.periodicity(x)
    return float(r.max(axis=1).mean()) if r.shape[0] else 0.0


def to_analysis_rate(x, fs_in):
    if abs(fs_in - 250000.0) < 1:
        return resample_poly(x, 16, 250)
    if abs(fs_in - 22050.0) < 1:
        return resample_poly(x, 320, 441)
    if abs(fs_in - FS) < 1:
        return x
    raise SystemExit("no exact ratio from %g Hz to %g Hz" % (fs_in, FS))


def bandpass(x, lo, hi):
    n = 2 ** int(np.ceil(np.log2(x.size)))
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, n)[:x.size]


rec = np.load(capture_path)
room = to_analysis_rate(rec["x"].astype(np.float64), float(rec["fs"]))
room = bandpass(room, band[0], band[1])

fs_w, wav = wavfile.read(speech_path)
if wav.ndim > 1:
    wav = wav.mean(axis=1)
speech = to_analysis_rate(wav.astype(np.float64) / 32768.0, float(fs_w))
speech = bandpass(speech, band[0], band[1])

need = int(seconds * FS)
print("capture %.1f s of room, speech %.1f s, band %.0f-%.0f Hz, segments of %.0f s"
      % (room.size / FS, speech.size / FS, band[0], band[1], seconds))
if room.size < need * 2 or speech.size < need:
    raise SystemExit("not enough material for %.0f s segments" % seconds)


def active_rms(x):
    """RMS over the loud half of the frames: the level while someone is talking."""
    f = int(0.02 * FS)
    n = x.size // f
    e = (x[:n * f].reshape(n, f) ** 2).mean(axis=1)
    loud = e >= np.percentile(e, 50.0)
    return float(np.sqrt(e[loud].mean()))


def take(x, n):
    i = int(rng.integers(0, x.size - n))
    return x[i:i + n]


# -- the null: what the room alone scores ------------------------------------
print("\nscoring %d noise-only segments to set the thresholds ..." % NULL_TRIALS)
null_path, null_frame = [], []
for _ in range(NULL_TRIALS):
    nz = take(room, need)
    null_path.append(detector.score(nz).score)
    null_frame.append(per_frame_score(nz))
null_path = np.array(null_path)
null_frame = np.array(null_frame)
thr_path = float(np.quantile(null_path, 1.0 - FALSE_POSITIVE_RATE))
thr_frame = float(np.quantile(null_frame, 1.0 - FALSE_POSITIVE_RATE))
print("   room alone, path score : median %.4f, 95th %.4f, worst %.4f"
      % (np.median(null_path), thr_path, null_path.max()))
print("   room alone, per-frame  : median %.4f, 95th %.4f, worst %.4f"
      % (np.median(null_frame), thr_frame, null_frame.max()))
print("   both thresholds let through %.0f%% of noise-only segments."
      % (100 * FALSE_POSITIVE_RATE))

# -- the sweep ----------------------------------------------------------------
print("\nburying speech in that same room at known levels, %d trials each:" % trials)
print("   %6s  %9s  %9s   %9s  %s"
      % ("SNR", "PATH", "per-frame", "path score", "what the path found"))
print("   %6s  %9s  %9s   %9s" % ("(dB)", "detected", "detected", "(median)"))
results = []
for snr in SNR_DB:
    scores, hits, hits_f, evid = [], 0, 0, None
    for _ in range(trials):
        sp = take(speech, need)
        nz = take(room, need)
        a = active_rms(nz) * (10.0 ** (snr / 20.0)) / max(active_rms(sp), 1e-12)
        mixed = a * sp + nz
        e = detector.score(mixed)
        scores.append(e.score)
        hits += e.score > thr_path
        hits_f += per_frame_score(mixed) > thr_frame
        if evid is None or e.score > evid.score:
            evid = e
    rate, rate_f = hits / trials, hits_f / trials
    results.append((snr, rate, rate_f))
    note = ""
    if evid is not None and rate >= 0.5:
        note = "pitch %.0f Hz, voiced %.0f%%" % (evid.median_f0_hz, 100 * evid.voiced_fraction)
    print("   %+5d     %5.0f%%     %5.0f%%    %9.3f  %s"
          % (snr, 100 * rate, 100 * rate_f, np.median(scores), note))

# -- where each gives out ------------------------------------------------------


def reach(index):
    got = [snr for row in results for snr in [row[0]] if row[index] >= 0.9]
    return min(got) if got else None


print("\nthe answer, over %.0f s segments at a %.0f%% false-alarm rate"
      % (seconds, 100 * FALSE_POSITIVE_RATE))
for label, idx in (("path (Markov)", 1), ("per-frame control", 2)):
    r90 = reach(idx)
    half = [row[0] for row in results if row[idx] >= 0.5]
    print("   %-18s 9 times in 10 down to %s;  half the time down to %s"
          % (label,
             "%+d dB" % r90 if r90 is not None else "never",
             "%+d dB" % min(half) if half else "never"))
a, b = reach(1), reach(2)
if a is not None and b is not None:
    print("   the Markov part is worth %+d dB" % (b - a))

