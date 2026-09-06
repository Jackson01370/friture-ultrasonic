"""List the steady lines in a saved capture -- the Band Survey dock, offline.

    python scripts\verify\band_anomalies.py CAPTURE.npz [--top N] [--from HZ] [--to HZ]

Runs friture.demod.survey, the same analysis the dock runs live, over a
whole capture instead of a rolling window. Every candidate is judged against
a LOCAL floor -- the median of the spectrum a few hundred hertz either side
-- because the room's noise and this microphone's own response both slope,
so "10 dB over the wideband floor" says nothing about whether something is a
bump on the display. A line is then measured over time: one that is always
there is a device, one that comes and goes is an event, and one that appears
only in the average is ripple.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from friture.demod.survey import SpectrumSurvey

args = sys.argv[1:]
path = args[0]


def option(name, default):
    return float(args[args.index(name) + 1]) if name in args else default


top_n = int(option("--top", 12))
f_from = option("--from", 100.0)
f_to = option("--to", 125000.0)

rec = np.load(path)
x = rec["x"].astype(np.float64)
fs = float(rec["fs"])
print("capture %.1f s at %.0f Hz, looking at %.0f - %.0f Hz\n" % (x.size / fs, fs, f_from, f_to))

# the whole capture is the history here: nothing is thrown away, which is
# the one difference from the live dock
survey = SpectrumSurvey(fs, history=100000)
for i in range(0, x.size, 1 << 16):
    survey.process(x[i:i + (1 << 16)])
if not survey.ready:
    print("capture too short (needs about %.1f s)" % (2 * survey.hop / fs))
    sys.exit(1)
print("%d windows of %.2f s, %.2f Hz per bin\n"
      % (survey.n_windows_total, survey.nfft / fs, survey.bin_hz))

print("lines standing out from their own neighbourhood, strongest first:")
print("  %11s %10s %10s   %s" % ("frequency", "over its", "absolute", "over time"))
print("  %11s %10s %10s" % ("", "floor", "level"))
for line in survey.lines(f_from, f_to, top=top_n):
    print("  %9.1f Hz %+9.1f %+10.1f   %s"
          % (line.frequency_hz, line.excess_db, line.level_db,
             "steady" if line.steady else "comes and goes (%.0f dB)" % line.spread_db))

print("\nthe shape of the noise itself (median level per band):")
edges = [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000,
         15000, 20000, 30000, 50000, 80000, 125000]
for lo, hi, median, peak, f_peak in survey.band_shape(edges):
    if hi <= f_to:
        print("   %6d-%6d Hz  median %6.1f dB   peak %6.1f dB at %8.1f Hz"
              % (lo, hi, median, peak, f_peak))
