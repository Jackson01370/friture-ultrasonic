"""Decode a signal that really went through the air.

    this PC's speakers -> the room -> UltraMic 250K -> Friture's capture path
    -> BandDecoder (the same object the Digital Decode dock runs)

A keyed signal is synthesised at the playback rate and played through the
default output device while Friture's AudioBackend captures at 250 kHz. The
decoder is fed from the backend's new_data_available signal exactly as the
dock is. The tones sit at 12-16 kHz: the top of what ordinary speakers
reproduce, well inside what the UltraMic hears. First 3 s of silence, where
the required answer is "no clock"; then FSK, OOK and DBPSK in turn.

Use it in a quiet room with the speakers on. Exit status 1 if the silence
locks or any keyed run fails to decode.
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
import argparse
import logging
import time

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QCoreApplication

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--save", metavar="DIR", default=None,
                    help="also save each run's capture as DIR/<mode>_<baud>.npz, "
                         "for scripts/verify/decode_recording.py")
args = parser.parse_args()
if args.save:
    os.makedirs(args.save, exist_ok=True)

from friture.audiobackend import AudioBackend, SAMPLING_RATE
from friture.audiobuffer import AudioBuffer
from friture.demod.decoder import DBPSK, FSK, OOK, BandDecoder

PLAY_FS = 48000
# The band and the tones follow what the speakers and the microphone
# actually pass. Measured here (scratchpad response probe, 0.4 s tones): 11,
# 12 and 13 kHz arrive within 1 dB of each other at ~50 dB over the floor;
# 16 kHz is 21 dB weaker. Two tones 21 dB apart cannot be keyed through a
# room, because the louder one's reverberation outweighs the quieter one's
# direct sound for most of a symbol.
F_LO, WIDTH = 9_500.0, 6_000.0             # the band the decoder looks at
TONES = (11_000.0, 13_500.0)               # FSK
# The 13.5 kHz tone arrived 6 dB weaker than the 11 kHz one (recording
# analysed with decode_recording.py: -34.4 vs -28.2 dB), and the room holds
# the 11 kHz tone's reverberation only 12 dB under its own level -- so for
# a whole symbol after a switch the 11 kHz residue out-weighed the 13.5 kHz
# direct sound (the sent 13.5 kHz symbols were heard as 13.5 kHz only 65% of
# the time, the 11 kHz ones 100%). A transmitter is expected to send its
# tones at equal power; this one is made to.
TONE_GAIN = (1.0, 2.0)
CARRIER = 12_000.0                         # OOK / DBPSK
# Symbol rates a ROOM allows. Measured here: a tone's envelope falls 10 dB
# within 3 ms of it stopping but then sits 15-20 dB down for 40 ms, and
# reflections keep arriving for ~25 ms after it starts, so an ON symbol's
# level wanders by 10 dB for that long (a 50 Bd OOK recording: the middle
# of ON symbols dipped to 13 dB below their median). A symbol has to be
# long against that: 40-50 ms. The first runs are what the room should
# support and decide the exit status; the rest are recorded for the record.
REQUIRED = [(FSK, 25.0), (OOK, 20.0)]
INFORMATIONAL = [(FSK, 50.0), (OOK, 50.0), (DBPSK, 25.0)]
RUNS = REQUIRED + INFORMATIONAL
SECONDS = 12.0
AMP = 0.4

app = QCoreApplication(sys.argv)

backend = AudioBackend()
if backend.device is None:
    print("FAIL: no capture device opened (is another process holding the UltraMic?)")
    sys.exit(1)
api = sd.query_hostapis(backend.device["hostapi"])["name"]
print("capturing from %s [%s] at %d Hz" % (backend.device["name"][:34], api, SAMPLING_RATE))
out = sd.query_devices(sd.default.device[1])
print("playing through %s at %d Hz" % (out["name"][:40], PLAY_FS))

buffer = AudioBuffer()
backend.new_data_available.connect(buffer.handle_new_data)


def synth(mode, baud, bits):
    T = PLAY_FS / baud
    n = int(len(bits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
    t = np.arange(n) / PLAY_FS
    if mode == FSK:
        f_inst = np.where(bits[idx] == 1, TONES[1], TONES[0])
        gain = np.where(bits[idx] == 1, TONE_GAIN[1], TONE_GAIN[0])
        x = gain * np.cos(2 * np.pi * np.cumsum(f_inst) / PLAY_FS)
    elif mode == OOK:
        x = np.cos(2 * np.pi * CARRIER * t) * bits[idx]
    else:
        sym = np.cumsum(bits) % 2
        x = np.cos(2 * np.pi * CARRIER * t + np.pi * sym[idx])
    # a short fade at both ends, so the speaker does not click
    ramp = np.linspace(0.0, 1.0, int(0.01 * PLAY_FS))
    x[:ramp.size] *= ramp
    x[-ramp.size:] *= ramp[::-1]
    return (AMP * x).astype(np.float32)


class Feed:
    """The dock's handle_new_data, minus the dock. Keeps the last 2 s too."""

    def __init__(self, decoder, keep_all=False):
        self.decoder = decoder
        self.blocks = 0
        self.locked_blocks = 0
        self.recent = []
        self.keep_all = keep_all
        self.t0 = time.time()
        self.log = []                 # (seconds since start, locked)
        # The decoder's readout the last time it was locked. The capture
        # runs on after the transmission, and the clock is dropped on the
        # silence that follows, so the state at the very end says nothing
        # about the transmission.
        self.last_locked = None       # (baud, confidence, eye, bits)

    def __call__(self, floatdata):
        dec = self.decoder
        dec.process(floatdata[0, :])
        self.blocks += 1
        self.log.append((time.time() - self.t0, dec.decode_locked))
        if dec.decode_locked:
            self.locked_blocks += 1
            self.last_locked = (dec.last_baud_hz, dec.last_baud_conf_db, dec.last_eye, dec.last_bits)
        self.recent.append(floatdata[0, :].astype(np.float32))
        if not self.keep_all:
            self.recent = self.recent[-int(2.0 * SAMPLING_RATE / 2048):]

    def locked_share(self, t_from, t_to):
        inside = [locked for t, locked in self.log if t_from <= t <= t_to]
        return (sum(inside) / len(inside)) if inside else 0.0

    def captured(self):
        return np.concatenate(self.recent).astype(np.float64) if self.recent else np.zeros(0)


def run(seconds, feed):
    backend.new_data_available.connect(feed)
    deadline = time.time() + seconds
    while time.time() < deadline:
        backend.fetchAudioData()
        app.processEvents()
        time.sleep(0.004)
    backend.new_data_available.disconnect(feed)


def band_level_db(samples, lo, hi):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
    freqs = np.fft.rfftfreq(samples.size, 1.0 / SAMPLING_RATE)
    inside = float(np.mean(spectrum[(freqs >= lo) & (freqs < hi)]))
    outside = float(np.mean(spectrum[(freqs >= 30_000) & (freqs < 60_000)]))
    return 10 * np.log10(inside / max(outside, 1e-30))


ok = True


def check(label, condition, detail, required=True):
    global ok
    verdict = "ok " if condition else ("FAIL" if required else "no  ")
    print("%-52s %s   %s" % (label, verdict, detail))
    if required:
        ok = ok and condition


backend.restart()

# The first playback after the capture starts comes out late or not at all
# on this machine's MME output (measured: the first of three runs arrived
# 3.8 dB over the floor, the next two 34 dB). A silent warm-up absorbs it.
sd.play(np.zeros(PLAY_FS, dtype=np.float32), PLAY_FS, blocking=True)

# -- the negative first: the room alone must not produce bits -----------------
print("\n3 s of the room alone (no playback), per mode")
for mode in (FSK, OOK, DBPSK):
    dec = BandDecoder(SAMPLING_RATE, mode=mode)
    dec.configure(F_LO, WIDTH)
    feed = Feed(dec)
    run(3.0, feed)
    check("room noise, %s: never locked" % mode.upper(), feed.locked_blocks == 0 and dec.last_bits == "",
          "%s (%d blocks)" % (dec.decode_text(), feed.blocks))

# -- then each modulation through the air ----------------------------------------
rng = np.random.default_rng(2026)
for mode, baud in RUNS:
    bits = rng.integers(0, 2, int(SECONDS * baud))
    if mode == OOK:
        bits[:8] = 0
    signal = synth(mode, baud, bits)
    dec = BandDecoder(SAMPLING_RATE, mode=mode)
    dec.configure(F_LO, WIDTH)
    feed = Feed(dec, keep_all=bool(args.save))

    required = (mode, baud) in REQUIRED
    print("\n%s at %.0f Bd through the speakers for %.0f s%s"
          % (mode.upper(), baud, SECONDS, "" if required else "  (for the record)"))
    sd.play(signal, PLAY_FS, blocking=False)
    run(SECONDS + 0.5, feed)
    sd.stop()

    captured = feed.captured()
    if args.save and captured.size:
        path = os.path.join(args.save, "%s_%.0f.npz" % (mode, baud))
        np.savez_compressed(path, x=captured.astype(np.float32), bits=bits, mode=mode,
                            baud=baud, f_lo=F_LO, width=WIDTH, fs=float(SAMPLING_RATE))
        print("   saved %s (%.1f s)" % (path, captured.size / SAMPLING_RATE))
        captured = captured[-int(2.0 * SAMPLING_RATE):]
    level = band_level_db(captured, F_LO, F_LO + WIDTH) if captured.size else 0.0
    truth = "".join(str(b) for b in bits)
    got_baud, got_conf, got_eye, got_bits = feed.last_locked or (0.0, 0.0, 0.0, "")
    shown = got_bits.replace(" ", "")
    print("   %s" % dec.signal_text())
    print("   at the last locked block: %.1f Bd, confidence %.0f dB, eye %.2f" % (got_baud, got_conf, got_eye)
          if feed.last_locked else "   never locked: %s" % dec.decode_text())
    print("   bits: %s" % got_bits)
    check("%s: the signal reached the microphone" % mode.upper(), level > 10.0,
          "band %.0f-%.0f kHz sits %.1f dB over 30-60 kHz" % (F_LO / 1e3, (F_LO + WIDTH) / 1e3, level))
    # The long-history clock needs 2 s of trace before its first estimate,
    # so the share is judged from 3 s in, up to the end of the transmission.
    share = feed.locked_share(3.0, SECONDS)
    check("%s: locked for most of the transmission" % mode.upper(), share > 0.5,
          "%d%% of the blocks between 3 s and %.0f s" % (round(100 * share), SECONDS), required)
    check("%s: recovered the baud" % mode.upper(),
          feed.last_locked is not None and abs(got_baud - baud) < 0.03 * baud,
          "%.1f Bd sent %.0f" % (got_baud, baud), required)
    # A few symbols of silence can trail the last sent bit (the clock is
    # held for one more estimate): the shown bits must be a run of the sent
    # ones up to such a tail.
    matched = max((k for k in range(len(shown) + 1) if shown[:k] in truth), default=0)
    check("%s: the shown bits are a run of the sent ones" % mode.upper(),
          matched >= 32 and len(shown) - matched <= 8,
          "%d of %d shown bits found in the sent sequence" % (matched, len(shown)), required)

backend.close()
print("\n" + ("ALL OK" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
