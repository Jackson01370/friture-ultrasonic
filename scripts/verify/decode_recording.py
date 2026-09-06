"""Replay a capture saved by decode_over_the_air.py --save through the decoder.

    python scripts\verify\decode_recording.py DIR\fsk_50.npz [more.npz ...]

No microphone needed: the same 2048-sample blocks the dock would have seen,
the same BandDecoder. Prints what the decoder said at the end, how much of
the run it trusted, and whether the shown bits are a run of the sent ones.
Optionally --trace prints the clock's verdict every 0.5 s, and for OOK the
envelope level, floor and ON share per half second.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from friture.demod.decoder import OOK, BandDecoder

BLOCK = 2048

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+")
parser.add_argument("--trace", action="store_true", help="print the readout every 0.5 s")
parser.add_argument("--mode", default=None, help="decode as this mode instead of the recorded one")
args = parser.parse_args()

ok = True
for name in args.files:
    rec = np.load(name)
    x = rec["x"].astype(np.float64)
    bits = rec["bits"]
    mode = args.mode or str(rec["mode"])
    baud, f_lo, width, fs = float(rec["baud"]), float(rec["f_lo"]), float(rec["width"]), float(rec["fs"])

    dec = BandDecoder(fs, mode=mode)
    dec.configure(f_lo, width)
    blocks = locked = 0
    next_report = 0.5
    last_locked = None            # the readout the last time the decoder was locked
    log = []                      # (t, locked)
    print("%s: %s %.0f Bd, band %.1f-%.1f kHz, %.1f s" % (name, mode.upper(), baud, f_lo / 1e3, (f_lo + width) / 1e3, x.size / fs))
    for i in range(0, x.size, BLOCK):
        dec.process(x[i:i + BLOCK])
        blocks += 1
        locked += int(dec.decode_locked)
        t = (i + BLOCK) / fs
        log.append((t, dec.decode_locked))
        if dec.decode_locked:
            last_locked = (dec.last_baud_hz, dec.last_baud_conf_db, dec.last_eye, dec.last_bits, t)
        if args.trace and t >= next_report:
            next_report += 0.5
            extra = ""
            if mode == OOK:
                extra = " | level %.2e floor %.2e on %d%%" % (dec._ook_level, dec.amp_floor, round(100 * dec.last_on_fraction))
            print("   %5.1f s  %-70s%s" % (t, dec.decode_text()[:70], extra))
    truth = "".join(str(b) for b in bits)
    # The capture runs on after the transmission ends and the clock is then
    # dropped on silence, so the state at the very end says nothing: judge
    # the readout at the LAST LOCKED block, and the locked share from 3 s
    # (the long history's first estimate) up to that block.
    got_baud, got_conf, got_eye, got_bits, t_last = last_locked or (0.0, 0.0, 0.0, "", 0.0)
    shown = got_bits.replace(" ", "")
    inside = [l for t, l in log if 3.0 <= t <= t_last]
    share = (sum(inside) / len(inside)) if inside else 0.0
    # A few symbols of silence can trail the last sent bit (the clock is
    # held for one more estimate): the shown bits must be a run of the sent
    # ones up to such a tail.
    matched = max((k for k in range(len(shown) + 1) if shown[:k] in truth), default=0)
    match = matched >= 32 and len(shown) - matched <= 8
    good = (last_locked is not None and abs(got_baud - baud) < 0.03 * baud
            and share > 0.5 and match)
    ok = ok and good
    print("   %s" % dec.signal_text())
    if last_locked is None:
        print("   never locked: %s" % dec.decode_text())
    else:
        print("   at the last locked block (%.1f s): %.1f Bd, confidence %.0f dB, eye %.2f; locked %d%% of the blocks from 3 s to there"
              % (t_last, got_baud, got_conf, got_eye, round(100 * share)))
    print("   bits %s" % (got_bits or "(none)"))
    if shown:
        print("   %d of %d shown bits are a run of the sent ones%s" % (
            matched, len(shown), "" if match else "  <- NOT the sent bits"))
    print("   %s" % ("ok" if good else "FAIL"))
sys.exit(0 if ok else 1)
