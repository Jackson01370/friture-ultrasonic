"""Load the touched QML offscreen and check it actually binds.

The Cython extensions are not built here so Friture itself will not start,
but the QML surface can be loaded on its own: that is where the risk is
(unresolved types, broken bindings, the click -> overlay chain).
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
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Fusion")


from PyQt5.QtCore import QObject, QUrl
from PyQt5.QtWidgets import QApplication
from PyQt5.QtQml import QQmlEngine, qmlRegisterType
from PyQt5.QtQuick import QQuickView

from friture.axis import Axis
from friture.colorBar import ColorBar
from friture.curve import Curve
from friture.filled_curve import FilledCurve
from friture.plotCurve import PlotCurve
from friture.plotFilledCurve import PlotFilledCurve
from friture.spectrogram_item import SpectrogramItem
from friture.spectrogram_item_data import SpectrogramImageData
from friture.listen.listen_band_view_model import GetListenBand, ListenBandViewModel
from friture.plotting.coordinateTransform import CoordinateTransform
from friture.plotting.scaleDivision import ScaleDivision, Tick
from friture.scope_data import Scope_Data
from friture.spectrum_data import Spectrum_Data
from friture.qml_tools import qml_url

app = QApplication(sys.argv)

for cls, name in [
    (ScaleDivision, 'ScaleDivision'),
    (CoordinateTransform, 'CoordinateTransform'),
    (Scope_Data, 'ScopeData'),
    (Spectrum_Data, 'SpectrumData'),
    (Axis, 'Axis'),
    (Curve, 'Curve'),
    (FilledCurve, 'FilledCurve'),
    (Tick, 'Tick'),
    (ColorBar, 'ColorBar'),
    (PlotCurve, 'PlotCurve'),
    (PlotFilledCurve, 'PlotFilledCurve'),
    (SpectrogramItem, 'SpectrogramItem'),
    (SpectrogramImageData, 'SpectrogramImageData'),
    (ListenBandViewModel, 'ListenBandViewModel'),
]:
    qmlRegisterType(cls, 'Friture', 1, 0, name)

engine = QQmlEngine()
failures = []


def load(file_name, initial_properties):
    view = QQuickView(engine, None)
    view.setResizeMode(QQuickView.SizeRootObjectToView)
    view.setInitialProperties(initial_properties)
    view.setSource(qml_url(file_name))
    if view.status() == QQuickView.Error:
        failures.append("%s: %s" % (file_name, '\n'.join(e.toString() for e in view.errors())))
        return None
    view.resize(800, 400)
    view.show()
    app.processEvents()
    return view


band = GetListenBand()

# 1. the control row
load("listen/ListenControl.qml", {"viewModel": band})

# 1b. the Digital Decode dock's view, with a readout already in it
from friture.digital_decode_view_model import DigitalDecodeViewModel
decode_vm = DigitalDecodeViewModel()
decode_vm.mode_text = "FSK"
decode_vm.band_text = "40.0 - 50.0 kHz"
decode_vm.decode_text = "300 Bd, confidence 23 dB, eye 1.00, 525 symbols"
decode_vm.locked = True
decode_vm.bits = "01000010 10001010 10100011 10111110"
decode_vm.set_symbols([0, 1, 1, -1, 0, 1], [1.0, 0.9, 0.6, 0.0, 1.0, 0.75])
decode_view = load("DigitalDecode.qml", {"viewModel": decode_vm, "fixedFont": "Courier New"})

# 1c. the Band Survey dock's view, with a couple of findings in it
from friture.band_survey_view_model import BandSurveyViewModel
survey_vm = BandSurveyViewModel()
survey_vm.status_text = "2 lines standing at least 4 dB over its own neighbourhood, from 8.4 s of spectrum"
survey_vm.range_text = "0.1 - 125.0 kHz"
survey_vm.set_lines([
    {"frequency": 25000.0, "frequency_text": "25.000 kHz", "excess": 23.3,
     "excess_text": "+23.3 dB", "level_text": "19.1 dB", "steady": True, "steadiness_text": "steady"},
    {"frequency": 16000.7, "frequency_text": "16.001 kHz", "excess": 14.9,
     "excess_text": "+14.9 dB", "level_text": "6.2 dB", "steady": False,
     "steadiness_text": "comes and goes (7 dB)"},
])
survey_vm.set_shape([
    {"band_text": "20.0 - 26.0 kHz", "bar": 1.0, "detail_text": "median -3.4 dB   peak 25.6 dB at 25.000 kHz"},
    {"band_text": "26.0 - 32.0 kHz", "bar": 0.4, "detail_text": "median -11.1 dB  peak 4.6 dB at 31.000 kHz"},
])
survey_view = load("BandSurvey.qml", {"viewModel": survey_vm, "fixedFont": "Courier New"})

# 2. a plot with a vertical frequency axis, as the spectrogram has
spectrogram_data = Scope_Data()
spectrogram_data.vertical_axis.name = "Frequency (Hz)"
spectrogram_data.vertical_axis.setRange(20, 20000)
spectrogram_data.horizontal_axis.setRange(0, 10)
spectrogram_data.set_listen_band(band, "vertical")
view = load("Plot.qml", {"scopedata": spectrogram_data})

# 3. a plot with a horizontal frequency axis, as the spectrum has
spectrum_data = Spectrum_Data()
spectrum_data.horizontal_axis.setRange(0, 22000)
spectrum_data.vertical_axis.setRange(-140, 0)
spectrum_data.set_listen_band(band, "horizontal")
h_view = load("Plot.qml", {"scopedata": spectrum_data})

if failures:
    print("QML ERRORS:")
    print('\n'.join(failures))
    sys.exit(1)

print("all QML loaded without errors")


def overlay_of(v):
    return v.rootObject().findChild(QObject, "listen_band_overlay")


# -- the click -> overlay chain, on the real bindings --------------------
band.enabled = True
band.width_hz = 2000

vertical = overlay_of(view)
horizontal = overlay_of(h_view)
if vertical is None or horizontal is None:
    print("FAIL: overlay item not found")
    sys.exit(1)

plot_h = vertical.parentItem().height()
plot_w = horizontal.parentItem().width()
print("plot area: %.0f x %.0f (vertical), width %.0f (horizontal)"
      % (vertical.parentItem().width(), plot_h, plot_w))

ok = True


def check(label, condition, detail):
    global ok
    print("%-46s %s   %s" % (label, "ok " if condition else "FAIL", detail))
    ok = ok and condition


# -- the Band Survey view binds to its model, and clicking a line tunes -----
survey_lines = survey_view.rootObject().findChild(QObject, "survey_lines") if survey_view else None
check("survey view lists the lines", survey_lines is not None and survey_lines.property("count") == 2,
      "count=%s" % (survey_lines.property("count") if survey_lines is not None else "not found"))
tuned = []
survey_vm.tuneRequested.connect(lambda f: tuned.append(f))
survey_vm.tune(25000.0)
check("clicking a line asks for a tune", tuned == [25000.0], "%s" % tuned)

# -- the Digital Decode view binds to its model ----------------------------
strip = decode_view.rootObject().findChild(QObject, "symbol_strip") if decode_view else None
check("decode view has the symbol strip", strip is not None and strip.property("visible"),
      "" if strip is not None else "symbol_strip not found")


def strip_symbols():
    # a `property var` comes back as a QJSValue, not a list
    value = strip.property("symbols")
    return list(value.toVariant() if hasattr(value, "toVariant") else value)


if strip is not None:
    check("symbol strip sees the model's symbols", strip_symbols() == [0, 1, 1, -1, 0, 1],
          "symbols=%s" % strip_symbols())
    decode_vm.bits = ""
    app.processEvents()
    check("strip hides when there are no bits", not strip.property("visible"), "")
    decode_vm.bits = "1111"
    decode_vm.set_symbols([1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0])
    app.processEvents()
    check("strip follows a new readout", strip.property("visible") and len(strip_symbols()) == 4,
          "symbols=%s" % strip_symbols())


band.click_center(10000.0)
app.processEvents()
y1, h1 = vertical.property("y"), vertical.property("height")
check("vertical overlay follows a click", h1 > 0, "y=%.1f h=%.1f" % (y1, h1))

band.click_center(2000.0)
app.processEvents()
y2 = vertical.property("y")
check("lower frequency sits lower on screen", y2 > y1, "y %.1f -> %.1f" % (y1, y2))

x1 = horizontal.property("x")
band.click_center(18000.0)
app.processEvents()
x2, hw = horizontal.property("x"), horizontal.property("width")
check("horizontal overlay moves right with frequency", x2 > x1, "x %.1f -> %.1f" % (x1, x2))
check("horizontal overlay has a width", hw > 0, "w=%.1f" % hw)

band.width_hz = 8000
app.processEvents()
h2 = vertical.property("height")
check("a wider band draws a taller overlay", h2 > h1, "h %.1f -> %.1f" % (h1, h2))

# the axis range changing must move the overlay too: this is what the
# range_min/range_max dependency in the binding is for
spectrogram_data.vertical_axis.setRange(20, 5000)
app.processEvents()
h3 = vertical.property("height")
check("overlay re-lays-out when the axis range changes", h3 != h2, "h %.1f -> %.1f" % (h2, h3))

band.enabled = False
app.processEvents()
check("overlay hides when listening is off", not vertical.property("visible"), "")

# -- a real mouse click, through the QML MouseArea ------------------------
# Everything above drove the model directly. This drives the actual handler:
# the injected `mouse` parameter and the pixel -> Hz mapping.
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtTest import QTest

spectrogram_data.vertical_axis.setRange(20, 20000)
band.enabled = True
band.width_hz = 2000
app.processEvents()


def click_at_frequency(plot_view, overlay, frequency, axis, is_vertical):
    """Click where `frequency` is drawn, and report what the band became."""
    plot_area = overlay.parentItem()
    relative = axis.coordinate_transform.toScreen(frequency)
    if is_vertical:
        local = QPointF(plot_area.width() / 2., (1. - relative) * plot_area.height())
    else:
        local = QPointF(relative * plot_area.width(), plot_area.height() / 2.)
    scene_point = plot_area.mapToScene(local)
    QTest.mouseClick(plot_view, Qt.LeftButton, Qt.NoModifier,
                     scene_point.toPoint())
    app.processEvents()
    return band.centre_hz


got = click_at_frequency(view, vertical, 8000., spectrogram_data.vertical_axis, True)
check("clicking the spectrogram centres the band there",
      abs(got - 8000) < 200, "clicked 8000 Hz -> centre %d Hz" % got)

got = click_at_frequency(h_view, horizontal, 3000., spectrum_data.horizontal_axis, False)
check("clicking the spectrum centres the band there",
      abs(got - 3000) < 200, "clicked 3000 Hz -> centre %d Hz" % got)

# a click near Nyquist must keep the width and shift the centre instead
got = click_at_frequency(h_view, horizontal, 21800., spectrum_data.horizontal_axis, False)
check("a click near the top keeps the width",
      abs((band.f_hi - band.f_lo) - 2000) < 1, "width %.0f Hz, centre %d Hz" % (band.f_hi - band.f_lo, got))



# -- dragging the band, through the real MouseArea ------------------------
print()
print("--- drag ---")
spectrogram_data.vertical_axis.setRange(20, 20000)
band.enabled = True
band.width_hz = 2000
band.click_center(9000.0)
app.processEvents()

plot_area = vertical.parentItem()
v_axis = spectrogram_data.vertical_axis


def y_of(frequency):
    return (1. - v_axis.coordinate_transform.toScreen(frequency)) * plot_area.height()


def press_move_release(points):
    """A press, a run of moves, then a release -- as a real drag arrives."""
    first = plot_area.mapToScene(QPointF(plot_area.width() / 2., points[0])).toPoint()
    QTest.mousePress(view, Qt.LeftButton, Qt.NoModifier, first)
    app.processEvents()
    for py in points[1:]:
        point = plot_area.mapToScene(QPointF(plot_area.width() / 2., py)).toPoint()
        QTest.mouseMove(view, point)
        app.processEvents()
    QTest.mouseRelease(view, Qt.LeftButton, Qt.NoModifier,
                       plot_area.mapToScene(QPointF(plot_area.width() / 2., points[-1])).toPoint())
    app.processEvents()


# grab the middle of the band and walk it down the screen (= lower frequency)
centre_y = y_of(9000.0)
steps = [centre_y + i for i in range(0, 61, 5)]
press_move_release(steps)
moved = band.centre_hz
check("dragging the body moves the band", moved < 8500,
      "centre 9000 -> %d Hz" % moved)
check("dragging the body keeps the width", abs(band.width_hz - 2000) < 2,
      "width %d Hz" % band.width_hz)

# grab the top edge and pull it up (= wider, centre must not move)
band.click_center(9000.0)
band.width_hz = 2000
app.processEvents()
before_centre = band.centre_hz
top_y = y_of(band.f_hi)
press_move_release([top_y, top_y - 10, top_y - 20, top_y - 30])
check("dragging an edge widens the band", band.width_hz > 2400,
      "width 2000 -> %d Hz" % band.width_hz)
check("dragging an edge keeps the centre", abs(band.centre_hz - before_centre) <= 60,
      "centre %d -> %d Hz" % (before_centre, band.centre_hz))

# pushing the same edge inwards must narrow it again
wide = band.width_hz
top_y = y_of(band.f_hi)
press_move_release([top_y, top_y + 10, top_y + 20])
check("pushing the edge back narrows the band", band.width_hz < wide,
      "width %d -> %d Hz" % (wide, band.width_hz))

# a press outside the band must still be a plain click-to-centre
band.click_center(9000.0)
band.width_hz = 2000
app.processEvents()
QTest.mouseClick(view, Qt.LeftButton, Qt.NoModifier,
                 plot_area.mapToScene(QPointF(plot_area.width() / 2., y_of(3000.0))).toPoint())
app.processEvents()
check("a click outside the band still re-centres it", abs(band.centre_hz - 3000) < 200,
      "centre -> %d Hz" % band.centre_hz)

sys.exit(0 if ok else 1)
