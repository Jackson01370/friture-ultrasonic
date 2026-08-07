"""Push synthetic audio through Friture's non-GUI pipeline.

The GUI cannot be started here, so this exercises the parts the fallbacks
actually sit on: decimation, octave filtering, the FFT stage, the
spectrogram's resample+colour pipeline, and the new band listening.
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
import sys


import numpy as np

FS = 48000
BLOCK = 1024
t = np.arange(4 * BLOCK) / FS
audio = (np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 9000 * t)
         + 0.1 * np.sin(2 * np.pi * 20500 * t))

ok = True


def check(label, fn):
    global ok
    try:
        detail = fn()
    except Exception as exc:
        ok = False
        print("FAIL %-42s %s: %s" % (label, type(exc).__name__, exc))
    else:
        print("ok   %-42s %s" % (label, detail))


def decimation():
    from friture.signal.decimate import decimate_multiple
    from friture.generated_filters import PARAMS
    bdec, adec = PARAMS['dec']
    zis = [np.zeros(max(len(bdec), len(adec)) - 1) for _ in range(3)]
    out, zfs = decimate_multiple(3, np.array(bdec), np.array(adec), audio, zis)
    assert np.all(np.isfinite(out)), "decimation produced non-finite samples"
    return "%d -> %d samples, finite" % (audio.size, out.size)


def fft_stage():
    from friture.audioproc import audioproc
    proc = audioproc()
    proc.set_fftsize(1024)
    proc.set_maxfreq(FS // 2)
    spectrum = proc.analyzelive(audio[:1024])
    peak_hz = proc.get_freq_scale()[int(np.argmax(spectrum))]
    assert abs(peak_hz - 440) < 60, "peak at %s Hz, expected 440" % peak_hz
    return "peak at %.0f Hz (fed 440 Hz)" % peak_hz


def octave_filters():
    from friture.octavefilters import Octave_Filters
    bank = Octave_Filters(3)
    levels, _ = bank.filter(audio)
    # each band comes out at its own decimated rate, so the rows are ragged
    for i, band in enumerate(levels):
        assert np.all(np.isfinite(band)), "band %d has non-finite samples" % i
    return "%d bands, all finite (%d..%d samples)" % (
        len(levels), min(len(b) for b in levels), max(len(b) for b in levels))


def exp_smoothing():
    from friture_extensions.exp_smoothing_conv import pyx_exp_smoothed_value_numpy
    alpha = 0.02
    kernel = (1. - alpha) ** np.arange(512)[::-1]
    data = np.abs(audio[:300]).reshape(1, -1) ** 2
    value = pyx_exp_smoothed_value_numpy(kernel, alpha, data, np.zeros(1))
    assert np.isfinite(value).all() and value[0] > 0, "smoothed value %s" % value
    return "value %.4f" % value[0]


def spectrogram_pipeline():
    from friture.signal.color_tranform import Color_Transform
    from friture.signal.frequency_resampler import Frequency_Resampler
    from friture.signal.online_linear_2D_resampler import Online_Linear_2D_resampler
    from friture.signal.transform_pipeline import Transform_Pipeline
    from fractions import Fraction
    import friture.plotting.frequency_scales as fscales

    freq_resampler = Frequency_Resampler()
    freq_resampler.setfreq(np.linspace(0, FS / 2, 513))
    freq_resampler.setfreqscale(fscales.Mel)
    freq_resampler.setfreqrange(20, FS / 2)
    freq_resampler.setnsamples(200)

    screen_resampler = Online_Linear_2D_resampler()
    screen_resampler.set_height(200)
    screen_resampler.set_ratio(Fraction(46), Fraction(30))

    pipeline = Transform_Pipeline([freq_resampler, screen_resampler, Color_Transform()])

    columns = 0
    for _ in range(20):
        spn = np.random.default_rng(1).random((513, 4))
        out = pipeline.push(spn)
        columns += out.shape[1] if out.size else 0
    assert columns > 0, "the pipeline never produced a column"
    return "%d colour columns produced" % columns


def band_listening():
    from friture.listen.band_dsp import BANDPASS, HETERODYNE, make_filter
    results = []
    for mode, name in ((BANDPASS, "band-pass"), (HETERODYNE, "heterodyne")):
        f = make_filter(mode)
        f.configure(19500.0, 2000.0, FS)  # around the 20.5 kHz tone
        out = np.concatenate([f.process(audio[i:i + BLOCK])
                              for i in range(0, audio.size, BLOCK)])
        assert out.size == audio.size, "%s changed the block length" % name
        results.append("%s rms %.4f" % (name, np.sqrt(np.mean(out ** 2))))
    return ", ".join(results)


check("decimation (lfilter fallback)", decimation)
check("FFT analysis", fft_stage)
check("octave filter bank (lfilter fallback)", octave_filters)
check("exponential smoothing fallback", exp_smoothing)
check("spectrogram resample + colour", spectrogram_pipeline)
check("band listening, both modes", band_listening)

sys.exit(0 if ok else 1)
