"""Drive the Band Survey dock offscreen, without the microphone.

Same shape as digital_decode_dock.py: the real widget, fed synthetic blocks
through handle_new_data, its view model read back after canvasUpdate, its
settings dialog driven, its state saved and restored, and its QML loaded
against the real view model. The one thing this dock does that the others do
not is send the Listen band somewhere when a line is clicked, so that is
checked too.
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
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")

import tempfile

import numpy as np
from PyQt5.QtCore import QObject, QSettings
from PyQt5.QtQml import QQmlEngine
from PyQt5.QtQuick import QQuickView
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)

from friture.audiobackend import SAMPLING_RATE
from friture.band_survey import BandSurvey_Widget
from friture.listen.listen_band_view_model import GetListenBand
from friture.qml_tools import qml_url

FS = float(SAMPLING_RATE)
BLOCK = 2048
ok = True


def check(label, condition, detail=""):
    global ok
    print("%-54s %s   %s" % (label, "ok " if condition else "FAIL", detail))
    ok = ok and condition


def feed(widget, x):
    for i in range(0, x.size, BLOCK):
        widget.handle_new_data(x[i:i + BLOCK].reshape(1, -1))


# a room with one steady line, one that comes and goes, and a broad hump
n = int(4.0 * FS)
t = np.arange(n) / FS
rng = np.random.default_rng(3)
hump = np.fft.irfft(np.fft.rfft(rng.normal(size=n))
                    * (1.0 + 6.0 * np.exp(-0.5 * ((np.fft.rfftfreq(n, 1.0 / FS) - 25_000.0) / 9_000.0) ** 2)), n)
x = (0.01 * hump
     + 0.02 * np.cos(2 * np.pi * 41_000.0 * t)
     + 0.02 * np.cos(2 * np.pi * 12_000.0 * t) * ((t % 1.0) < 0.35))

widget = BandSurvey_Widget(None)
check("qml file name", widget.qml_file_name() == "BandSurvey.qml", widget.qml_file_name())
vm = widget.view_model()
widget.set_buffer(None)
for _ in range(5):
    widget.canvasUpdate()
check("says it is still listening before it can answer", "listening" in vm.status_text, vm.status_text)

feed(widget, x)
for _ in range(5):
    widget.canvasUpdate()
lines = [vm.lines.get(i) for i in range(vm.lines.count)]
check("found lines", len(lines) >= 2, "%d lines" % len(lines))
check("sorted by frequency, lowest first",
      [row["frequency"] for row in lines] == sorted(row["frequency"] for row in lines),
      "%s" % [round(row["frequency"] / 1000, 1) for row in lines])
freqs = [round(row["frequency"] / 1000) for row in lines]
check("found the steady line at 41 kHz", 41 in freqs, "%s" % freqs)
check("found the intermittent line at 12 kHz", 12 in freqs, "%s" % freqs)
steady = {round(row["frequency"] / 1000): row["steady"] for row in lines}
check("told the steady one from the intermittent one",
      steady.get(41) is True and steady.get(12) is False, "%s" % steady)
check("the hump itself is not listed as a line",
      not any(20 <= round(row["frequency"] / 1000) <= 30 for row in lines), "%s" % freqs)
check("the shape readout shows the hump", len(vm.shape) > 4, "%d bands" % len(vm.shape))

# -- clicking a line moves the shared Listen band -----------------------------
band = GetListenBand()
band.click_center(1000.0)
row41 = next(r for r in lines if round(r["frequency"] / 1000) == 41)
vm.tune(row41["frequency"], row41["width"])
check("clicking a line points the Listen band at it",
      abs(band.get_centre_hz() - 41_000) < 1, "centre %s width %s" % (band.get_centre_hz(), band.get_width_hz()))
check("the band is sized to the line, not a fixed 2 kHz",
      400 <= band.get_width_hz() <= 4000 and abs(band.get_width_hz() - 2000) > 1,
      "width %s Hz for a bare tone" % band.get_width_hz())

# THE CLICK MUST LAND, from wherever the band happened to be. Both setters on
# the shared band keep it inside the capture, so each can move the other's
# value: measured before the fix, 8 kHz asked for at 120 kHz came back 2 kHz
# wide (clamped against the OLD centre), and 124 kHz asked for 20 kHz wide
# came back centred at 115 kHz.
missed = []
for start_c in (300, 25_000, 124_000):
    for start_w in (400, 20_000):
        for f in (120.0, 1_000.0, 25_000.0, 95_000.0, 124_500.0):
            for wanted in (400.0, 8_000.0, 20_000.0):
                band.click_center(float(start_c))
                band.set_width_hz(start_w)
                widget._on_tune(f, wanted)
                if abs(band.get_centre_hz() - f) > 1:
                    missed.append((start_c, start_w, f, wanted, band.get_centre_hz()))
check("the click lands on the line from any starting band",
      not missed, "%d of 90 combinations missed%s" % (len(missed), (": %s" % missed[:2]) if missed else ""))

# -- the settings dialog drives it -------------------------------------------
dlg = widget.settings_dialog
dlg.spin_from.setValue(30000)
dlg.spin_to.setValue(60000)
for _ in range(5):
    widget.canvasUpdate()
freqs = [round(vm.lines.get(i)["frequency"] / 1000) for i in range(vm.lines.count)]
check("the range setting narrows the search", 41 in freqs and 12 not in freqs, "%s" % freqs)
check("the range is shown", "30.0" in vm.range_text, vm.range_text)
dlg.spin_top.setValue(1)
for _ in range(5):
    widget.canvasUpdate()
check("the count setting is honoured", vm.lines.count == 1, "%d lines" % vm.lines.count)
dlg.spin_excess.setValue(40.0)
for _ in range(5):
    widget.canvasUpdate()
check("a high threshold empties the list", vm.lines.count == 0, "%d lines" % vm.lines.count)
dlg.spin_excess.setValue(4.0)

ini = Path(tempfile.mkdtemp()) / "survey.ini"
settings = QSettings(str(ini), QSettings.IniFormat)
widget.saveState(settings)
settings.sync()
other = BandSurvey_Widget(None)
other.restoreState(QSettings(str(ini), QSettings.IniFormat))
check("state round-trips",
      other.settings_dialog.spin_from.value() == 30000 and other.settings_dialog.spin_top.value() == 1,
      "%d / %d" % (other.settings_dialog.spin_from.value(), other.settings_dialog.spin_top.value()))

# -- pause / restart ----------------------------------------------------------
widget.pause()
before = widget.survey.n_windows_total
widget.handle_new_data(np.zeros((1, BLOCK)))
paused_ok = widget.survey.n_windows_total == before
widget.restart()
check("paused: blocks ignored; restart starts a fresh survey",
      paused_ok and widget.survey.n_windows_total == 0, "")

# -- the QML against the real view model --------------------------------------
feed(widget, x)
for _ in range(5):
    widget.canvasUpdate()
engine = QQmlEngine()
view = QQuickView(engine, None)
view.setResizeMode(QQuickView.SizeRootObjectToView)
view.setInitialProperties({"viewModel": vm, "fixedFont": "Courier New"})
view.setSource(qml_url("BandSurvey.qml"))
if view.status() == QQuickView.Error:
    check("BandSurvey.qml loads with the real view model", False,
          "\n".join(e.toString() for e in view.errors()))
else:
    view.resize(900, 500)
    view.show()
    app.processEvents()
    check("BandSurvey.qml loads with the real view model", True, "")
    lst = view.rootObject().findChild(QObject, "survey_lines")
    check("the line list is populated in the view", lst is not None and lst.property("count") > 0,
          "count=%s" % (lst.property("count") if lst is not None else "no list"))

    # THE CLICK TEST. The rows used to be a QVariantList, replaced whole on
    # every refresh, so the delegate under the cursor was destroyed between
    # press and release and most clicks did nothing. A row must survive a
    # refresh that changes only its numbers.
    first = view.rootObject().findChild(QObject, "survey_lines").childItems()
    before_rows = [vm.lines.get(i)["frequency"] for i in range(vm.lines.count)]
    resets = []
    vm.lines.modelAboutToBeReset.connect(lambda: resets.append(1))
    inserted = []
    vm.lines.rowsInserted.connect(lambda *a: inserted.append(1))
    removed = []
    vm.lines.rowsRemoved.connect(lambda *a: removed.append(1))
    for _ in range(30):
        feed(widget, x[:BLOCK * 40])
        widget.canvasUpdate()
        app.processEvents()
    after_rows = [vm.lines.get(i)["frequency"] for i in range(vm.lines.count)]
    check("rows survive refreshes (no rebuild under the cursor)",
          not resets and not inserted and not removed and before_rows == after_rows,
          "%d resets, %d inserts, %d removes" % (len(resets), len(inserted), len(removed)))

    # and the dock fits whatever height it is given
    for h in (200, 300, 500):
        view.resize(900, h)
        app.processEvents()
        root_item = view.rootObject()
        content = root_item.childItems()[0]
        fits = content.property("height") <= root_item.property("height") + 1
        check("fits a %d px dock" % h, fits,
              "content %.0f px in %.0f px" % (content.property("height"), root_item.property("height")))

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
