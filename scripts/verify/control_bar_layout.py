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
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FRITURE_RECORDING_DIR", "off")   # never the user's recording folder
os.environ.setdefault("QT_QUICK_BACKEND", "software")
import PyQt5
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject
from friture.analyzer import Friture
from friture.listen.listen_band_view_model import GetListenBand

app = QApplication(sys.argv)
w = Friture(); w.show()
band = GetListenBand()

def find(item, cls):
    if item.metaObject().className().startswith(cls):
        return item
    for c in item.childItems():
        r = find(c, cls)
        if r: return r

print("%-8s %-9s %-8s %-8s %s" % ("width", "listen", "bar h", "plot h", "plot w"))
for width in (1000, 1280, 1600):
    for on in (False, True):
        band.enabled = on
        w.resize(width, 700)
        for _ in range(40): app.processEvents()
        root = w.quick_view.rootObject()
        bar = find(root, "ListenControl")
        tile = find(root, "TileLayout")
        print("%-8d %-9s %-8.0f %-8.0f %.0f"
              % (width, "on" if on else "off", bar.property("height"),
                 tile.property("height"), tile.property("width")))
