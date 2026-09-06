#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2026 puppy

# This file is part of Friture.
#
# Friture is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as published by
# the Free Software Foundation.
#
# Friture is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Friture.  If not, see <http://www.gnu.org/licenses/>.

"""The Digital Decode dock: bits out of the listen band.

Decodes whatever band the Listen feature has selected -- click a peak on a
spectrum or spectrogram, or type it into this dock's settings -- as OOK, FSK
or DBPSK, and shows the recovered symbol rate, how much to believe it, and
the bits. The maths is friture.demod; this file is the dock plumbing that
every friture widget has: a view model for the QML, a settings dialog, and
the handle_new_data / canvasUpdate pair.

The decoder runs on the GUI thread, in handle_new_data, like every other
dock's analysis. Measured cost is well under a millisecond per 2048-sample
block for FSK, the heaviest mode; see scripts/verify/digital_decode.py.
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject

from friture.audiobackend import SAMPLING_RATE
from friture.demod.decoder import (
    AM,
    ANALOG_MODES,
    FM,
    FSK,
    MODE_DESCRIPTIONS,
    MODE_LABELS,
    MODES,
    PM,
    BandDecoder,
)
from friture.digital_decode_view_model import DigitalDecodeViewModel
from friture.listen.listen_band_view_model import (
    MIN_WIDTH_HZ,
    NYQUIST_HZ,
    GetListenBand,
)

DEFAULT_MODE = FSK
DEFAULT_DECODE = True
DEFAULT_THRESHOLD_DB = BandDecoder.DEFAULT_THRESHOLD_DB

HELP_TEXT = (
    "Demodulates the Listen band: click a peak on a spectrum or spectrogram to "
    "centre the band there, drag its edges to set the width, or type both into "
    "this dock's settings. Pick the modulation there too: a keyed mode (OOK, "
    "FSK, 4-FSK, DBPSK, DQPSK) gives bits, shown only once a symbol clock has "
    "been recovered with enough confidence; an analog mode (FM, AM, PM) shows "
    "the demodulated waveform with its carrier, deviation and modulation rate."
)


class DigitalDecode_Widget(QObject):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.audiobuffer = None
        self._band = GetListenBand()
        self._decoder = BandDecoder(SAMPLING_RATE, mode=DEFAULT_MODE,
                                    threshold_db=DEFAULT_THRESHOLD_DB)
        self._decoder.enabled = DEFAULT_DECODE
        self._paused = False
        # Point the decoder at the band now, not on the first block: the
        # readout then says "no clock found" from the start rather than "no
        # band selected", which would be untrue -- the Listen band always is.
        f_lo, width, _mode = self._band.band_snapshot()
        self._decoder.configure(f_lo, width)

        self._view_model = DigitalDecodeViewModel(self)
        self._view_model.help_text = HELP_TEXT
        self._refresh_mode_texts()
        self._refresh_band_text()

        self.settings_dialog = DigitalDecode_Settings_Dialog(parent, self, self._band)

    # -- the dock's duck-typed interface ----------------------------------

    def view_model(self):
        return self._view_model

    def qml_file_name(self):
        return "DigitalDecode.qml"

    def set_buffer(self, buffer):
        self.audiobuffer = buffer

    def handle_new_data(self, floatdata):
        if self._paused:
            return
        # The band is sampled here rather than pushed in on a signal, the
        # same way the listen path does it: a drag that moves it a hundred
        # times between blocks costs one rebuild, not a hundred.
        f_lo, width, _mode = self._band.band_snapshot()
        self._decoder.configure(f_lo, width)
        self._decoder.process(floatdata[0, :])

    def canvasUpdate(self):
        decoder = self._decoder
        vm = self._view_model
        self._refresh_band_text()
        vm.signal_text = decoder.signal_text()
        vm.decode_text = decoder.decode_text()
        vm.locked = decoder.decode_locked
        vm.bits = decoder.last_bits
        vm.levels = max(2, decoder.levels)
        vm.set_symbols(decoder.last_symbols, decoder.last_agreements)
        vm.analog = decoder.is_analog
        if decoder.is_analog:
            vm.set_trace(decoder.last_trace,
                         self._trace_label(decoder.last_trace_hi),
                         self._trace_label(decoder.last_trace_lo))
        else:
            vm.set_trace((), "", "")

    def pause(self):
        self._paused = True

    def restart(self):
        # A pause is a gap in the stream, and every stage here carries state
        # across block edges. Start over rather than stitch across the gap.
        self._decoder.reset()
        self._paused = False

    def settings_called(self, checked):
        self.settings_dialog.show()

    def saveState(self, settings):
        self.settings_dialog.saveState(settings)

    def restoreState(self, settings):
        self.settings_dialog.restoreState(settings)

    # -- what the settings dialog drives ------------------------------------

    @property
    def decoder(self) -> BandDecoder:
        return self._decoder

    def set_mode(self, mode: str) -> None:
        self._decoder.set_mode(mode)
        self._refresh_mode_texts()

    def set_decode_enabled(self, enabled: bool) -> None:
        self._decoder.enabled = bool(enabled)
        if not enabled:
            self._decoder.reset()

    def set_threshold_db(self, threshold_db: float) -> None:
        self._decoder.threshold_db = float(threshold_db)

    # -- text ------------------------------------------------------------------

    def _refresh_mode_texts(self) -> None:
        decoder = self._decoder
        vm = self._view_model
        vm.mode_text = MODE_LABELS[decoder.mode]
        vm.mode_hint = MODE_DESCRIPTIONS[decoder.mode]
        if decoder.is_analog:
            vm.range_text = ""
        else:
            lo, hi = decoder.baud_range
            vm.range_text = "%.0f - %.0f Bd" % (lo, hi)

    def _trace_label(self, value: float) -> str:
        """An axis label for the analog waveform, in the mode's unit."""
        mode = self._decoder.mode
        if mode == FM:
            return "%.3f kHz" % (value / 1e3)
        if mode == PM:
            return "%+.2f rad" % value
        if mode == AM:
            return "%.3f" % value
        return ""

    def _refresh_band_text(self) -> None:
        f_lo, width, _mode = self._band.band_snapshot()
        text = "%.1f - %.1f kHz" % (f_lo / 1e3, (f_lo + width) / 1e3)
        seen = self._decoder.decoded_width_hz
        if self._decoder.band is not None and 0.0 < seen < width:
            # The baseband is 50 kHz, so the decoder sees 25 kHz of band at
            # most. Silence about the rest would read as "nothing up there".
            text += " (decoding the lower %.1f kHz)" % (seen / 1e3)
        self._view_model.band_text = text


class DigitalDecode_Settings_Dialog(QtWidgets.QDialog):

    def __init__(self, parent, widget: DigitalDecode_Widget, band):
        super().__init__(parent)

        self._widget = widget
        self._band = band

        self.setWindowTitle("Digital decode settings")
        self.form_layout = QtWidgets.QFormLayout(self)

        self.combo_mode = QtWidgets.QComboBox(self)
        for i, mode in enumerate(MODES):
            self.combo_mode.addItem(MODE_LABELS[mode], mode)
            self.combo_mode.setItemData(i, MODE_DESCRIPTIONS[mode], 3)  # Qt.ToolTipRole
        self.combo_mode.setCurrentIndex(MODES.index(DEFAULT_MODE))
        self.combo_mode.setToolTip(
            "Keyed modes give bits:\n"
            "  OOK: a carrier switched on and off.\n"
            "  FSK / 4-FSK: two or four tones, found from the signal itself.\n"
            "  DBPSK / DQPSK: 180- or 90-degree phase steps; the bits are the changes.\n"
            "Analog modes show the demodulated waveform:\n"
            "  FM: instantaneous frequency.  AM: envelope.  PM: phase.\n"
            "Symbol rates, measured on synthetic signals: OOK 10-700 Bd, "
            "FSK 10-4000 Bd, DBPSK 100-4000 Bd. Through the air in a room only slow "
            "keying survives the reverberation: FSK at 25 Bd and OOK at 20 Bd were "
            "decoded from this PC's speakers, DBPSK was not.")
        self.form_layout.addRow("Modulation:", self.combo_mode)

        self.check_decode = QtWidgets.QCheckBox("Recover the symbol clock and show bits", self)
        self.check_decode.setChecked(DEFAULT_DECODE)
        self.check_decode.setToolTip(
            "Off leaves only the signal readout (tones, carrier, keying). "
            "The clock recovery is the expensive part.")
        self.form_layout.addRow("Decode:", self.check_decode)

        self.spin_threshold = QtWidgets.QDoubleSpinBox(self)
        self.spin_threshold.setDecimals(0)
        self.spin_threshold.setMinimum(0.0)
        self.spin_threshold.setMaximum(40.0)
        self.spin_threshold.setValue(DEFAULT_THRESHOLD_DB)
        self.spin_threshold.setSuffix(" dB")
        self.spin_threshold.setToolTip(
            "How far the symbol clock's spectral line must stand above the floor "
            "before bits are shown. Noise scores 8-11 dB, real keying 20-33 dB.")
        self.form_layout.addRow("Clock confidence needed:", self.spin_threshold)

        self.spin_centre = QtWidgets.QSpinBox(self)
        self.spin_centre.setMinimum(0)
        self.spin_centre.setMaximum(int(NYQUIST_HZ))
        self.spin_centre.setSingleStep(100)
        self.spin_centre.setSuffix(" Hz")
        self.form_layout.addRow("Band centre:", self.spin_centre)

        self.spin_width = QtWidgets.QSpinBox(self)
        self.spin_width.setMinimum(int(MIN_WIDTH_HZ))
        self.spin_width.setMaximum(int(NYQUIST_HZ))
        self.spin_width.setSingleStep(100)
        self.spin_width.setSuffix(" Hz")
        self.form_layout.addRow("Band width:", self.spin_width)

        note = QtWidgets.QLabel(
            "The band is the Listen band, shared with the plots: clicking a "
            "peak on a spectrum or spectrogram moves it too. The decoder sees "
            "at most 25 kHz of it.", self)
        note.setWordWrap(True)
        self.form_layout.addRow(note)

        self.setLayout(self.form_layout)

        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.check_decode.toggled.connect(self._on_decode_toggled)
        self.spin_threshold.valueChanged.connect(self._widget.set_threshold_db)
        self.spin_centre.valueChanged.connect(self._band.set_centre_hz)
        self.spin_width.valueChanged.connect(self._band.set_width_hz)
        self._band.centre_changed.connect(self._sync_band)
        self._band.width_changed.connect(self._sync_band)
        self._sync_band()

    def _on_mode_changed(self, index: int) -> None:
        mode = MODES[index]
        self._widget.set_mode(mode)
        # the clock and its threshold mean nothing to a waveform
        keyed = mode not in ANALOG_MODES
        self.check_decode.setEnabled(keyed)
        self.spin_threshold.setEnabled(keyed and self.check_decode.isChecked())

    def _on_decode_toggled(self, checked: bool) -> None:
        self._widget.set_decode_enabled(checked)
        self.spin_threshold.setEnabled(checked)

    def _sync_band(self) -> None:
        """Mirror the shared band into the spin boxes, without echoing it back."""
        for box, value in ((self.spin_centre, self._band.get_centre_hz()),
                           (self.spin_width, self._band.get_width_hz())):
            box.blockSignals(True)
            box.setValue(int(value))
            box.blockSignals(False)

    # method
    def saveState(self, settings):
        settings.setValue("mode", MODES[self.combo_mode.currentIndex()])
        settings.setValue("decode", self.check_decode.isChecked())
        settings.setValue("threshold_db", self.spin_threshold.value())
        # The band is the shared Listen band, which Listen itself does not
        # persist. A decode dock that comes back on a different band than it
        # was closed on is a decode dock showing "no clock" for no visible
        # reason, so the dock remembers it.
        settings.setValue("band_centre_hz", int(self._band.get_centre_hz()))
        settings.setValue("band_width_hz", int(self._band.get_width_hz()))

    # method
    def restoreState(self, settings):
        mode = settings.value("mode", DEFAULT_MODE, type=str)
        if mode not in MODES:
            mode = DEFAULT_MODE
        self.combo_mode.setCurrentIndex(MODES.index(mode))
        self.check_decode.setChecked(settings.value("decode", DEFAULT_DECODE, type=bool))
        self.spin_threshold.setValue(settings.value("threshold_db", DEFAULT_THRESHOLD_DB, type=float))
        if settings.contains("band_centre_hz") and settings.contains("band_width_hz"):
            self._band.set_width_hz(settings.value("band_width_hz", type=int))
            self._band.set_centre_hz(settings.value("band_centre_hz", type=int))
