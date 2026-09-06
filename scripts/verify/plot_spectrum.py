"""Draw a capture's average spectrum as a PNG, with the lines that stand out marked.

    python scripts\verify\plot_spectrum.py CAPTURE.npz OUT.png [--from HZ] [--to HZ] [--mark HZ,HZ,...]

Qt does the drawing (matplotlib is not installed in this environment). The
curve is the average power spectrum; the grey band behind it is the LOCAL
floor -- the median of the spectrum a few hundred hertz either side -- so
what reads as a bump on screen is what rises above the grey, not what is
merely high up the slope.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import PyQt5
_qt = Path(PyQt5.__file__).parent / "Qt5"
os.environ.setdefault("QT_PLUGIN_PATH", str(_qt / "plugins"))
# NOT the offscreen platform: it comes up with no font database on this
# machine, so every drawText is silently a no-op and the picture arrives
# with its curves and no labels. A QImage needs no window, so the normal
# platform plugin is fine and it brings the fonts with it.

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QImage, QPainter, QPen

args = sys.argv[1:]
cap, out = args[0], args[1]


def option(name, default):
    return args[args.index(name) + 1] if name in args else default


f_from = float(option("--from", "300"))
f_to = float(option("--to", "30000"))
marks = [float(v) for v in option("--mark", "").split(",") if v]
title = option("--title", Path(cap).stem)

app = QGuiApplication([])
FAMILY = QFontDatabase().families()[0] if QFontDatabase().families() else "Arial"
for wanted in ("Segoe UI", "Arial", "Tahoma"):
    if wanted in QFontDatabase().families():
        FAMILY = wanted
        break

rec = np.load(cap)
x = rec["x"].astype(np.float64)
fs = float(rec["fs"])

nfft = 1 << 16
win = np.hanning(nfft)
acc = np.zeros(nfft // 2 + 1)
n = 0
for i in range(0, x.size - nfft, nfft // 2):
    acc += np.abs(np.fft.rfft(x[i:i + nfft] * win)) ** 2
    n += 1
psd = acc / max(n, 1)
freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
db = 10 * np.log10(np.maximum(psd, 1e-30))
bin_hz = float(freqs[1])
half = max(1, int(250.0 / bin_hz))
sel = (freqs >= f_from) & (freqs <= f_to)
idx = np.flatnonzero(sel)
local = np.array([np.median(db[max(i - half, 0):min(i + half + 1, db.size)]) for i in idx])
f_sel, d_sel = freqs[idx], db[idx]

W, H = 1400, 620
L, R, T, B = 90, 30, 60, 70
img = QImage(W, H, QImage.Format_RGB32)
img.fill(QColor(20, 20, 26))
p = QPainter(img)
p.setRenderHint(QPainter.Antialiasing)

lo_db = float(np.percentile(d_sel, 1)) - 6
hi_db = float(d_sel.max()) + 8


def X(f):
    return L + (np.log10(f) - np.log10(f_from)) / (np.log10(f_to) - np.log10(f_from)) * (W - L - R)


def Y(v):
    return T + (hi_db - v) / (hi_db - lo_db) * (H - T - B)


# grid
p.setFont(QFont(FAMILY, 9))
decade = 10 ** int(np.floor(np.log10(f_from)))
f = decade
while f <= f_to:
    for m in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        fv = f * m
        if f_from <= fv <= f_to:
            major = m == 1 or fv in (1000.0, 10000.0)
            p.setPen(QPen(QColor(60, 60, 72) if major else QColor(38, 38, 46), 1))
            p.drawLine(QPointF(X(fv), T), QPointF(X(fv), H - B))
            if major or m in (2, 5):
                p.setPen(QColor(150, 150, 160))
                label = "%.0f k" % (fv / 1000) if fv >= 1000 else "%.0f" % fv
                p.drawText(QRectF(X(fv) - 30, H - B + 4, 60, 16), Qt.AlignHCenter, label)
    f *= 10
for v in np.arange(np.ceil(lo_db / 10) * 10, hi_db, 10):
    p.setPen(QPen(QColor(45, 45, 55), 1))
    p.drawLine(QPointF(L, Y(v)), QPointF(W - R, Y(v)))
    p.setPen(QColor(150, 150, 160))
    p.drawText(QRectF(4, Y(v) - 8, L - 12, 16), Qt.AlignRight | Qt.AlignVCenter, "%.0f dB" % v)

# the local floor, then the spectrum
p.setPen(QPen(QColor(120, 120, 140), 2))
pts = [QPointF(X(f_sel[i]), Y(local[i])) for i in range(0, f_sel.size, 4)]
for a, b in zip(pts[:-1], pts[1:]):
    p.drawLine(a, b)
p.setPen(QPen(QColor(110, 220, 140), 1))
pts = [QPointF(X(f_sel[i]), Y(d_sel[i])) for i in range(f_sel.size)]
for a, b in zip(pts[:-1], pts[1:]):
    p.drawLine(a, b)

# the marks
p.setFont(QFont(FAMILY, 10, QFont.Bold))
for k, mf in enumerate(marks):
    if not (f_from <= mf <= f_to):
        continue
    j = int(np.argmin(np.abs(f_sel - mf)))
    over = d_sel[j] - local[j]
    colour = QColor(255, 170, 70) if over > 8 else QColor(120, 190, 255)
    # stack the labels down the top of the plot so neighbouring marks on a
    # log axis do not overprint each other
    row = T + 8 + 20 * k
    p.setPen(QPen(colour, 1, Qt.DashLine))
    p.drawLine(QPointF(X(mf), Y(d_sel[j]) - 10), QPointF(X(mf), row + 14))
    p.setPen(colour)
    txt = "%.2f kHz  %+.0f dB" % (mf / 1000, over)
    p.drawText(QRectF(X(mf) - 75, row, 150, 18), Qt.AlignHCenter, txt)

p.setPen(QColor(230, 230, 235))
p.setFont(QFont(FAMILY, 12, QFont.Bold))
p.drawText(QRectF(L, 8, W - L - R, 20), Qt.AlignLeft, title)
p.setPen(QColor(160, 160, 170))
p.setFont(QFont(FAMILY, 9))
p.drawText(QRectF(L, 28, W - L - R, 20), Qt.AlignLeft,
           "green: average spectrum of %.0f s   grey: the local floor (median +-250 Hz)   "
           "a bump is green above grey" % (x.size / fs))
p.drawText(QRectF(L, H - 22, W - L - R, 18), Qt.AlignHCenter, "Frequency (Hz)")
p.end()
img.save(out)
print("saved %s (%dx%d)" % (out, W, H))
