"""Does a band carry speech, a machine, or just noise? Judged against a reference band.

    python scripts\verify\is_it_speech.py CAPTURE.npz LO_HZ HI_HZ [--reference LO,HI]

A band inside hearing needs no demodulation -- it is already sound. What is
worth measuring is whether anything STRUCTURED is in it, and the honest way
to do that is to run the same tests on a band of the SAME capture that is
known to hold nothing, so the numbers are compared with this microphone in
this room rather than with remembered thresholds.

  1. PAUSES. Speech is intermittent: its loudest and quietest seconds
     differ far more than steady noise does. This is the strongest test and
     it needs no calibration beyond the reference band. It is measured as
     the 10th-to-90th PERCENTILE spread, not the full range: a capture with
     one door slam in it has a 30 dB range and is otherwise dead steady,
     and the range alone calls that speech. Speech has many loud seconds
     and many quiet ones, so its percentile spread is wide too.
  2. SYLLABLE RHYTHM. Speech switches on and off at 2-8 Hz, so its envelope
     spectrum is concentrated there.
  3. HARMONIC COMB. Voiced speech has a pitch whose harmonics ride through
     the band, so the band's spectrum repeats at the pitch. NOTE the
     reference: a noisy spectrum autocorrelates to 0.3-0.4 on its own in
     this room, so only a score well clear of the reference band's means
     anything.
  4. WHERE THE ENERGY IS. Speech is loudest below 1 kHz and falls above
     4 kHz. A hump that sits only at 3-4 kHz, with a valley below it, is
     not a voice.
"""

import sys

import numpy as np

args = sys.argv[1:]
path = args[0]
lo = float(args[1]) if len(args) > 1 else 3000.0
hi = float(args[2]) if len(args) > 2 else 4500.0
ref = args[args.index("--reference") + 1] if "--reference" in args else "10000,11500"
ref_lo, ref_hi = (float(v) for v in ref.split(","))

rec = np.load(path)
x = rec["x"].astype(np.float64)
fs = float(rec["fs"])
print("capture %.1f s at %.0f Hz" % (x.size / fs, fs))
print("band under test %.0f-%.0f Hz, reference band %.0f-%.0f Hz\n" % (lo, hi, ref_lo, ref_hi))

_n = 1 << int(np.ceil(np.log2(x.size)))
_X = np.fft.rfft(x, _n)
_f = np.fft.rfftfreq(_n, 1.0 / fs)


def bandpass(a, b):
    X = _X.copy()
    X[(_f < a) | (_f > b)] = 0.0
    return np.fft.irfft(X, _n)[:x.size]


def measures(a, b):
    band = bandpass(a, b)
    env = np.abs(band)
    taps = int(fs / 100)
    env_s = np.convolve(env, np.ones(taps) / taps, mode="same")
    per = int(fs)
    levels = np.array([20 * np.log10(max(float(np.sqrt(np.mean(env[i * per:(i + 1) * per] ** 2))), 1e-12))
                       for i in range(int(env.size // per))])
    e = env_s - env_s.mean()
    m = 1 << int(np.floor(np.log2(e.size)))
    spec = np.abs(np.fft.rfft(e[:m] * np.hanning(m)))
    fm = np.fft.rfftfreq(m, 1.0 / fs)
    tot = spec[(fm >= 0.5) & (fm < 200)].sum()
    syl = spec[(fm >= 2.0) & (fm < 8.0)].sum()
    seg = 1 << 15
    comb = 0.0
    for i in range(0, x.size - seg, seg // 2):
        sp = np.abs(np.fft.rfft(x[i:i + seg] * np.hanning(seg)))
        ff = np.fft.rfftfreq(seg, 1.0 / fs)
        s = (ff >= a) & (ff <= b)
        v = sp[s] - sp[s].mean()
        ac = np.correlate(v, v, mode="full")[v.size - 1:]
        ac = ac / max(ac[0], 1e-30)
        k0, k1 = int(80 / ff[1]), int(400 / ff[1])
        if k1 < ac.size:
            comb = max(comb, float(ac[k0:k1].max()))
    return levels, 100 * syl / max(tot, 1e-30), comb


lv, syl, comb = measures(lo, hi)
rlv, rsyl, rcomb = measures(ref_lo, ref_hi)

def spread(v):
    return float(np.percentile(v, 90) - np.percentile(v, 10))


print("1. pauses (speech is intermittent)")
print("   band under test: 10-90%% spread %.1f dB   (full range %.1f dB)"
      % (spread(lv), lv.max() - lv.min()))
print("   reference band : 10-90%% spread %.1f dB   (full range %.1f dB)"
      % (spread(rlv), rlv.max() - rlv.min()))
print("   per second: " + " ".join("%.0f" % v for v in lv[:60])
      + (" ..." if lv.size > 60 else ""))

print("\n2. syllable rhythm (share of the envelope's 0.5-200 Hz energy at 2-8 Hz)")
print("   band under test: %.1f%%      reference band: %.1f%%" % (syl, rsyl))

print("\n3. harmonic comb (spectrum repeating at a pitch of 80-400 Hz)")
print("   band under test: %.2f        reference band: %.2f" % (comb, rcomb))

print("\n4. where the energy is")
nf = 1 << 16
acc = np.zeros(nf // 2 + 1)
c = 0
for i in range(0, x.size - nf, nf // 2):
    acc += np.abs(np.fft.rfft(x[i:i + nf] * np.hanning(nf))) ** 2
    c += 1
psd = acc / max(c, 1)
ff = np.fft.rfftfreq(nf, 1.0 / fs)
for a, b in ((100, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4500), (4500, 8000)):
    s = (ff >= a) & (ff < b)
    bar = "#" * max(0, int((10 * np.log10(max(float(psd[s].mean()), 1e-30)) + 20)))
    print("   %5d-%5d Hz  %6.1f dB  %s" % (a, b, 10 * np.log10(max(float(psd[s].mean()), 1e-30)), bar))

print("\nverdict")
speechy = spread(lv) > spread(rlv) + 8.0
print("   intermittency %s the reference band's -> %s"
      % ("clearly exceeds" if speechy else "matches or is below",
         "worth listening to for speech" if speechy else "steady, so not speech"))
loud = int(np.sum(lv > np.percentile(lv, 10) + 10.0))
print("   %d of %d seconds sit more than 10 dB above the quiet level%s"
      % (loud, lv.size, " -- a handful of events, not a conversation" if 0 < loud <= max(3, lv.size // 20) else ""))
