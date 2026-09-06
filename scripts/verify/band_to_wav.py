"""Cut a band out of a capture and write it as a WAV you can listen to.

    python scripts\verify\band_to_wav.py CAPTURE.npz OUT.wav LO_HZ HI_HZ
                                        [--gain-db N] [--decim N] [--from S] [--seconds S]
                                        [--heterodyne]

--heterodyne moves the band down so that LO_HZ becomes 0 Hz, which is the
only way to hear anything above about 20 kHz. It changes the pitch: a tone
at LO_HZ + 3 kHz comes out at 3 kHz whatever LO_HZ was. This is what the
app's Listen feature does live.

For a band inside hearing there is nothing to demodulate: it is already
sound, and the honest way to find out what it is is to listen to it. The
capture runs at 250 kHz, so the output is decimated by 5 to 50 kHz -- an
exact factor and a rate every player accepts -- after the band-pass, which
has already removed everything that decimation could fold back in.

The output is normalised, and the gain applied is printed, because a band
40 dB under full scale is inaudible otherwise and how much it was lifted is
part of the measurement.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

args = sys.argv[1:]
cap, out = args[0], args[1]
lo, hi = float(args[2]), float(args[3])
gain_db = float(args[args.index("--gain-db") + 1]) if "--gain-db" in args else None
# 250 kHz / 5 = 50 kHz suits any band; a band under 10 kHz fits comfortably
# in 25 kHz and the file is then half the size.
decim = int(args[args.index("--decim") + 1]) if "--decim" in args else 5
start_s = float(args[args.index("--from") + 1]) if "--from" in args else 0.0
take_s = float(args[args.index("--seconds") + 1]) if "--seconds" in args else None

heterodyne = "--heterodyne" in args

rec = np.load(cap)
x = rec["x"].astype(np.float64)
fs = float(rec["fs"])
a0 = int(start_s * fs)
x = x[a0:a0 + int(take_s * fs)] if take_s else x[a0:]
fs_out = fs / decim
if heterodyne:
    assert hi - lo < fs_out / 2, "the band's WIDTH must fit under the output's Nyquist"
else:
    assert hi < fs_out / 2, "the band must fit under the output's Nyquist"

# band-pass in the frequency domain: exact, and this is offline
n = 1 << int(np.ceil(np.log2(x.size)))
X = np.fft.rfft(x, n)
f = np.fft.rfftfreq(n, 1.0 / fs)
keep = (f >= lo) & (f <= hi)
# a raised-cosine edge, so the cut does not ring
edge = max((hi - lo) * 0.05, 20.0)
taper = np.clip(np.minimum((f - (lo - edge)) / edge, ((hi + edge) - f) / edge), 0.0, 1.0)
X *= taper
band = np.fft.irfft(X, n)[:x.size]

if heterodyne:
    # mix lo down to DC; the band-pass above has already isolated the band,
    # so nothing folds in from the negative side
    band = band * np.exp(-2j * np.pi * lo * np.arange(band.size) / fs)
    band = 2.0 * np.real(band)
y = band[::decim]
peak = float(np.max(np.abs(y))) if y.size else 0.0
rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
if gain_db is None:
    gain = 0.7 / max(peak, 1e-12)          # normalise to -3 dBFS
else:
    gain = 10.0 ** (gain_db / 20.0)
y = np.clip(y * gain, -1.0, 1.0)

wavfile.write(out, int(fs_out), (y * 32767).astype(np.int16))
print("%s: %.0f-%.0f Hz of %.1f s, written at %.0f Hz" % (Path(out).name, lo, hi, x.size / fs, fs_out))
print("   the band measured %.1f dBFS peak / %.1f dBFS rms in the capture;"
      % (20 * np.log10(max(peak, 1e-12)), 20 * np.log10(max(rms, 1e-12))))
print("   the file is that lifted by %+.1f dB, so it is audible." % (20 * np.log10(gain)))
