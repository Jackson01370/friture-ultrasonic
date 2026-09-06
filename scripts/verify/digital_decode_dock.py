"""Drive the Digital Decode dock widget offscreen, without the microphone.

The Dock loads a widget by duck typing (view_model, qml_file_name,
set_buffer, handle_new_data, canvasUpdate, pause/restart, settings_called,
saveState/restoreState). This exercises that surface on the real class:
synthetic FSK blocks go in through handle_new_data, the view model is read
back after canvasUpdate, the settings dialog is driven, the state is saved
and restored through QSettings, and the QML is loaded against the real view
model.
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
from friture.demod.decoder import DBPSK, FM, FSK, MODES, OOK
from friture.digital_decode import DigitalDecode_Widget
from friture.listen.listen_band_view_model import GetListenBand
from friture.qml_tools import qml_url

FS = float(SAMPLING_RATE)
BLOCK = 2048
ok = True


def check(label, condition, detail=""):
    global ok
    print("%-52s %s   %s" % (label, "ok " if condition else "FAIL", detail))
    ok = ok and condition


def fsk(f0, f1, baud, bits, amp=0.5):
    T = FS / baud
    n = int(len(bits) * T)
    idx = np.clip((np.arange(n) / T).astype(int), 0, len(bits) - 1)
    return amp * np.cos(2 * np.pi * np.cumsum(np.where(bits[idx] == 1, f1, f0)) / FS)


def feed(widget, x):
    for i in range(0, x.size, BLOCK):
        widget.handle_new_data(x[i:i + BLOCK].reshape(1, -1))


# -- construction, as the Dock does it -------------------------------------
widget = DigitalDecode_Widget(None)
check("qml file name", widget.qml_file_name() == "DigitalDecode.qml", widget.qml_file_name())
vm = widget.view_model()
widget.set_buffer(None)
widget.canvasUpdate()
check("starts in FSK with the band shown", vm.mode_text == "FSK" and "kHz" in vm.band_text,
      "%s / %s" % (vm.mode_text, vm.band_text))
check("says what is missing before any data", vm.decode_text != "" and not vm.locked, vm.decode_text)

# -- the band comes from the shared Listen band ----------------------------
band = GetListenBand()
band.click_center(45000.0)
band.set_width_hz(10000)
bits = np.random.default_rng(3).integers(0, 2, 750)          # 2.5 s at 300 Bd
x = fsk(43000.0, 47000.0, 300.0, bits)
feed(widget, x)
widget.canvasUpdate()
check("decodes the listen band", vm.locked and "Bd" in vm.decode_text, vm.decode_text)
check("shows the bits", len(vm.bits.replace(" ", "")) >= 32
      and vm.bits.replace(" ", "") in "".join(str(b) for b in bits), vm.bits)
check("symbol strip data is populated", len(vm.symbols) == len(vm.agreements) > 0,
      "%d symbols" % len(vm.symbols))
check("signal line names the tones", "43.0" in vm.signal_text, vm.signal_text)

# -- the settings dialog drives the decoder ---------------------------------
dlg = widget.settings_dialog
check("dialog mirrors the band", dlg.spin_centre.value() == 45000 and dlg.spin_width.value() == 10000,
      "%d / %d" % (dlg.spin_centre.value(), dlg.spin_width.value()))
dlg.spin_centre.setValue(60000)
check("dialog moves the shared band", abs(band.get_centre_hz() - 60000) < 1, "%s" % band.get_centre_hz())
band.click_center(45000.0)
check("a plot click comes back into the dialog", dlg.spin_centre.value() == 45000, "%d" % dlg.spin_centre.value())

dlg.combo_mode.setCurrentIndex(0)
widget.canvasUpdate()
check("mode combo switches the decoder", widget.decoder.mode == OOK and vm.mode_text == "OOK",
      "%s / %s" % (widget.decoder.mode, vm.mode_text))
check("a mode change clears the readout", vm.bits == "" and not vm.locked, vm.decode_text)
dlg.spin_threshold.setValue(20.0)
check("threshold spin reaches the decoder", widget.decoder.threshold_db == 20.0, "%s" % widget.decoder.threshold_db)
dlg.check_decode.setChecked(False)
widget.canvasUpdate()
check("decode checkbox switches the clock off", not widget.decoder.enabled and "off" in vm.decode_text, vm.decode_text)
dlg.check_decode.setChecked(True)
dlg.combo_mode.setCurrentIndex(MODES.index(DBPSK))
check("DBPSK selectable", widget.decoder.mode == DBPSK, widget.decoder.mode)
check("the mode combo lists all %d modes" % len(MODES), dlg.combo_mode.count() == len(MODES),
      "%d entries" % dlg.combo_mode.count())

# -- an analog mode shows the waveform instead of bits ----------------------
dlg.combo_mode.setCurrentIndex(MODES.index(FM))
check("analog mode disables the clock controls", not dlg.check_decode.isEnabled()
      and not dlg.spin_threshold.isEnabled(), "")
t = np.arange(int(2.0 * FS)) / FS
feed(widget, 0.5 * np.cos(2 * np.pi * 45000.0 * t + (1500.0 / 200.0) * np.sin(2 * np.pi * 200.0 * t)))
widget.canvasUpdate()
check("FM readout names carrier, deviation and rate",
      vm.analog and "FM carrier" in vm.signal_text and "modulated at 200" in vm.signal_text, vm.signal_text)
check("the waveform reaches the view model", len(vm.trace) == 400 and vm.trace_hi_text.endswith("kHz"),
      "%d points, %s .. %s" % (len(vm.trace), vm.trace_lo_text, vm.trace_hi_text))
check("no bits in an analog mode", vm.bits == "" and not vm.locked, vm.decode_text)
dlg.combo_mode.setCurrentIndex(MODES.index(DBPSK))
widget.canvasUpdate()
check("back to a keyed mode: clock controls return, trace cleared",
      dlg.check_decode.isEnabled() and not vm.analog and len(vm.trace) == 0, "")

# -- save / restore through QSettings ---------------------------------------
ini = Path(tempfile.mkdtemp()) / "decode.ini"
settings = QSettings(str(ini), QSettings.IniFormat)
widget.saveState(settings)
settings.sync()
band.click_center(60000.0)
band.set_width_hz(4000)
other = DigitalDecode_Widget(None)
other.restoreState(QSettings(str(ini), QSettings.IniFormat))
check("state round-trips (mode, threshold)", other.decoder.mode == DBPSK and other.decoder.threshold_db == 20.0,
      "%s / %s" % (other.decoder.mode, other.decoder.threshold_db))
check("the band comes back with the dock", abs(band.get_centre_hz() - 45000) < 1 and band.get_width_hz() == 10000,
      "centre %s width %s" % (band.get_centre_hz(), band.get_width_hz()))

# -- pause / restart -----------------------------------------------------------
dlg.combo_mode.setCurrentIndex(MODES.index(FSK))
feed(widget, x)
widget.canvasUpdate()
locked_before = vm.locked
widget.pause()
widget.handle_new_data(np.zeros((1, BLOCK)))
n_blocks_paused = widget.decoder.n_blocks
widget.restart()
check("paused: blocks are ignored, restart starts over", locked_before and n_blocks_paused > 0
      and widget.decoder.n_blocks == 0 and not widget.decoder.decode_locked, "")

# -- the QML against the real view model --------------------------------------
engine = QQmlEngine()
view = QQuickView(engine, None)
view.setResizeMode(QQuickView.SizeRootObjectToView)
view.setInitialProperties({"viewModel": vm, "fixedFont": "Courier New"})
view.setSource(qml_url("DigitalDecode.qml"))
if view.status() == QQuickView.Error:
    check("DigitalDecode.qml loads with the real view model", False,
          "\n".join(e.toString() for e in view.errors()))
else:
    view.resize(800, 400)
    view.show()
    feed(widget, x)
    widget.canvasUpdate()
    app.processEvents()
    check("DigitalDecode.qml loads with the real view model", True, "")
    check("live readout reaches the view", vm.locked, vm.decode_text)

# -- the dock type selector lists every widget, this one included -------------
from friture.controlbar_viewmodel import ControlBarViewModel
from friture.widgetdict import widgets

bar_vm = ControlBarViewModel(None)
bar = QQuickView(engine, None)
bar.setInitialProperties({"viewModel": bar_vm})
bar.setSource(qml_url("ControlBar.qml"))
if bar.status() == QQuickView.Error:
    check("ControlBar.qml loads", False, "\n".join(e.toString() for e in bar.errors()))
else:
    selector = bar.rootObject().findChild(QObject, "widget_selector")
    names = [w["Name"] for w in widgets]
    listed = selector.property("model") if selector is not None else None
    listed = list(listed.toVariant() if hasattr(listed, "toVariant") else (listed or []))
    check("every widget in widgetdict can be constructed",
      all(w["Class"](None) is not None for w in widgets), "%d widgets" % len(widgets))
check("the dock type selector lists every widget in widgetdict order", listed == names,
          "QML %s vs widgetdict %s" % (listed, names))

print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
