"""Run the digital decoder on synthetic keyed signals at the capture rate.

No GUI, no microphone: OOK, FSK and DBPSK signals are built at 250 kHz and
fed to friture.demod.decoder.BandDecoder in 2048-sample blocks, exactly as
the audio backend delivers them. For each, the recovered baud, the clock's
confidence, the eye, and whether the shown bits are a run of the sent ones.
Then the same for noise, where the required answer is "no clock".

The last column is the processing time per block, against the 8.2 ms a block
lasts.

Exit status 1 if any keyed signal fails to decode or any noise run shows bits.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from friture.demod.decoder import (
    AM, ANALOG_MODES, DBPSK, DIGITAL_MODES, DQPSK, FM, FSK, FSK4, MODE_LABELS, OOK, PM, BandDecoder,
)
from friture.demod.symbols import symbols_to_bits

FS = 250_000.0
BLOCK = 2048
LABELS = MODE_LABELS
MODES = DIGITAL_MODES
F_LO, WIDTH = 40_000.0, 10_000.0
SECONDS = 3.0


def fsk(f0, f1, baud, bits, amp=0.5):
    T = FS / baud
    n = int(len(bits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
    f_inst = np.where(bits[idx] == 1, f1, f0)
    return amp * np.cos(2 * np.pi * np.cumsum(f_inst) / FS)


def dbpsk(fc, baud, bits, amp=0.5):
    sym = np.cumsum(bits) % 2
    T = FS / baud
    n = int(len(bits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
    t = np.arange(n) / FS
    return amp * np.cos(2 * np.pi * fc * t + np.pi * sym[idx])


def ook(fc, baud, bits, amp=0.5, seed=0):
    T = FS / baud
    n = int(len(bits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
    t = np.arange(n) / FS
    return (amp * np.cos(2 * np.pi * fc * t) * bits[idx]
            + 1e-4 * np.random.default_rng(seed).normal(size=n))


def fsk4(tones, baud, dibits, amp=0.5):
    T = FS / baud
    n = int(len(dibits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(dibits) - 1)
    return amp * np.cos(2 * np.pi * np.cumsum(np.asarray(tones)[dibits[idx]]) / FS)


def dqpsk(fc, baud, dibits, amp=0.5):
    sym = np.cumsum(dibits) % 4
    T = FS / baud
    n = int(len(dibits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(dibits) - 1)
    t = np.arange(n) / FS
    return amp * np.cos(2 * np.pi * fc * t + (np.pi / 2) * sym[idx])


def synth(mode, baud, seed):
    """(signal, the sent BITS)."""
    rng = np.random.default_rng(seed)
    if mode in (FSK4, DQPSK):
        dibits = rng.integers(0, 4, int(SECONDS * baud))
        bits = symbols_to_bits(dibits, 2)
        if mode == FSK4:
            return fsk4((42_000.0, 44_000.0, 46_000.0, 48_000.0), baud, dibits), bits
        return dqpsk(45_000.0, baud, dibits), bits
    bits = rng.integers(0, 2, int(SECONDS * baud))
    if mode == OOK:
        bits[:8] = 0
        return ook(45_000.0, baud, bits), bits
    if mode == FSK:
        return fsk(43_000.0, 47_000.0, baud, bits), bits
    return dbpsk(45_000.0, baud, bits), bits


def run(mode, x):
    dec = BandDecoder(FS, mode=mode)
    dec.configure(F_LO, WIDTH)
    ever_locked = False
    peak_conf = 0.0
    t0 = time.perf_counter()
    n_blocks = 0
    for i in range(0, x.size, BLOCK):
        dec.process(x[i:i + BLOCK])
        n_blocks += 1
        ever_locked = ever_locked or dec.decode_locked
        peak_conf = max(peak_conf, dec.last_baud_conf_db)
    per_block_ms = 1e3 * (time.perf_counter() - t0) / max(n_blocks, 1)
    return dec, ever_locked, peak_conf, per_block_ms


ok = True
print("%-6s %8s %9s %6s %5s  %-7s %-30s %s" % (
    "mode", "sent Bd", "found Bd", "conf", "eye", "bits", "signal", "ms/block"))
print("-" * 96)

SWEEP = [(OOK, 100.0), (OOK, 200.0), (OOK, 400.0), (OOK, 700.0),
         (FSK, 100.0), (FSK, 300.0), (FSK, 1500.0), (FSK, 4000.0),
         (FSK4, 300.0), (FSK4, 1500.0),
         (DBPSK, 200.0), (DBPSK, 1000.0), (DBPSK, 3000.0),
         (DQPSK, 500.0), (DQPSK, 2000.0)]

for k, (mode, baud) in enumerate(SWEEP):
    x, bits = synth(mode, baud, seed=100 + k)
    dec, _, _, ms = run(mode, x)
    shown = dec.last_bits.replace(" ", "")
    truth = "".join(str(b) for b in bits)
    bits_ok = dec.decode_locked and len(shown) >= 32 and shown in truth
    baud_ok = dec.decode_locked and abs(dec.last_baud_hz - baud) < 0.02 * baud
    good = bits_ok and baud_ok
    ok = ok and good
    if mode in (FSK, FSK4):
        signal = ("tones " + "/".join("%.1f" % (t / 1e3) for t in dec.last_fsk_tones) + " kHz"
                  if dec.last_fsk_tones else "no tones")
    elif mode in (DBPSK, DQPSK):
        signal = "carrier %.2f kHz lock %.2f" % (dec.last_carrier_hz / 1e3, dec.last_lock)
    else:
        signal = "on %d%%" % round(100 * dec.last_on_fraction)
    print("%-6s %8.0f %9.1f %5.1f %5.2f  %-7s %-30s %6.2f   %s" % (
        LABELS[mode], baud, dec.last_baud_hz, dec.last_baud_conf_db, dec.last_eye,
        "match" if bits_ok else ("NONE" if not dec.decode_locked else "WRONG"),
        signal, ms, "ok" if good else "FAIL"))

print()
print("noise only (0.01 rms white), the required answer is no clock:")
noise = 0.01 * np.random.default_rng(7).normal(size=int(SECONDS * FS))
for mode in MODES:
    dec, ever_locked, peak_conf, ms = run(mode, noise)
    good = not ever_locked and dec.last_bits == ""
    ok = ok and good
    print("  %-6s peak confidence %5.1f dB, %s   %6.2f ms/block   %s" % (
        LABELS[mode], peak_conf,
        "never locked" if not ever_locked else "LOCKED ON NOISE", ms,
        "ok" if good else "FAIL"))

print()
print("analog modes, 2.5 s each (the readout must name the carrier, the deviation and the rate):")
t = np.arange(int(2.5 * FS)) / FS
analog = [
    (FM, 0.5 * np.cos(2 * np.pi * 45_000.0 * t + (1_500.0 / 200.0) * np.sin(2 * np.pi * 200.0 * t)),
     ("carrier", 45_000.0, 30.0), ("deviation", 1_500.0, 225.0), ("rate", 200.0, 10.0)),
    (AM, 0.3 * (1.0 + 0.5 * np.cos(2 * np.pi * 150.0 * t)) * np.cos(2 * np.pi * 45_000.0 * t),
     ("carrier", 45_000.0, 30.0), ("depth", 0.5, 0.08), ("rate", 150.0, 8.0)),
    (PM, 0.5 * np.cos(2 * np.pi * 45_000.0 * t + 1.0 * np.sin(2 * np.pi * 100.0 * t)),
     ("carrier", 45_000.0, 30.0), ("deviation", 1.0, 0.25), ("rate", 100.0, 5.0)),
]
for mode, x, *expect in analog:
    dec, _, _, ms = run(mode, x)
    got = {"carrier": dec.last_carrier_hz, "deviation": dec.last_deviation, "depth": dec.last_deviation,
           "rate": dec.last_mod_rate_hz}
    good = dec.analog_carrier and all(abs(got[name] - want) <= tol for name, want, tol in expect)
    ok = ok and good
    print("  %-6s %-70s %6.2f ms/block   %s" % (LABELS[mode], dec.signal_text()[:70], ms, "ok" if good else "FAIL"))
noise = 0.01 * np.random.default_rng(8).normal(size=int(2.5 * FS))
for mode in ANALOG_MODES:
    dec, _, _, ms = run(mode, noise)
    good = not dec.analog_carrier
    ok = ok and good
    print("  %-6s noise: %-56s %6.2f ms/block   %s" % (LABELS[mode], dec.signal_text()[:56], ms, "ok" if good else "FAIL"))

print()
print("a block lasts %.1f ms" % (1e3 * BLOCK / FS))
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
