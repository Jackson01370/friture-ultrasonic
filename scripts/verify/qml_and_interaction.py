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
