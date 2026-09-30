#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2009 Timothée Lecomte

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

import os
import sys
import logging

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtProperty
from friture.audiobackend import AudioBackend, SAMPLING_RATE
from friture.main_toolbar_view_model import MainToolbarViewModel
from friture.recording.store import BYTES_PER_GB, default_directory, inside_git_worktree
from friture.ui_settings import Ui_Settings_Dialog

DEFAULT_RECORDING_CAP_GB = 50

no_input_device_title = "No audio input device found"

no_input_device_message = """No audio input device has been found.

Friture needs at least one input device. Please check your audio configuration.

Friture will now exit.
"""


class Settings_Dialog(QtWidgets.QDialog, Ui_Settings_Dialog):
    show_playback_changed = pyqtSignal(bool)
    history_length_changed = pyqtSignal(int)
    # enabled, folder, cap in GB
    continuous_recording_changed = pyqtSignal(bool, str, int)

    def __init__(self, parent, toolbar_view_model: MainToolbarViewModel):
        QtWidgets.QDialog.__init__(self, parent)
        Ui_Settings_Dialog.__init__(self)

        self.logger = logging.getLogger(__name__)

        self._toolbar_view_model = toolbar_view_model

        # Setup the user interface
        self.setupUi(self)

        devices = AudioBackend().get_readable_devices_list()

        if devices == []:
            # no audio input device: display a message and exit
            QtWidgets.QMessageBox.critical(self, no_input_device_title, no_input_device_message)
            QtCore.QTimer.singleShot(0, self.exitOnInit)
            sys.exit(1)
            return

        for device in devices:
            self.comboBox_inputDevice.addItem(device)

        channels = AudioBackend().get_readable_current_channels()
        for channel in channels:
            self.comboBox_firstChannel.addItem(channel)
            self.comboBox_secondChannel.addItem(channel)

        current_device = AudioBackend().get_readable_current_device()
        self.comboBox_inputDevice.setCurrentIndex(current_device)

        first_channel = AudioBackend().get_current_first_channel()
        self.comboBox_firstChannel.setCurrentIndex(first_channel)
        second_channel = AudioBackend().get_current_second_channel()
        self.comboBox_secondChannel.setCurrentIndex(second_channel)

        # signals
        self.comboBox_inputDevice.currentIndexChanged.connect(self.input_device_changed)
        self.comboBox_firstChannel.activated.connect(self.first_channel_changed)
        self.comboBox_secondChannel.activated.connect(self.second_channel_changed)
        self.radioButton_single.toggled.connect(self.single_input_type_selected)
        self.radioButton_duo.toggled.connect(self.duo_input_type_selected)
        self.checkbox_showPlayback.stateChanged.connect(self.show_playback_checkbox_changed)
        self.spinBox_historyLength.editingFinished.connect(self.history_length_edit_finished)

        self._build_recording_group()

    def _build_recording_group(self):
        """The continuous recording controls, added below Playback.

        Built here rather than in the .ui file so the generated
        ui_settings.py stays as upstream generated it.
        """
        group = QtWidgets.QGroupBox("Continuous recording (all frequencies)", self)
        form = QtWidgets.QFormLayout(group)

        self.checkbox_recording = QtWidgets.QCheckBox("Record everything the microphone hears", group)
        self.checkbox_recording.setToolTip(
            "Writes the raw capture -- every frequency, 16-bit -- to disk for as long as the "
            "capture runs, in 10-minute files. Any band can be listened to or analysed later.")
        form.addRow(self.checkbox_recording)

        row = QtWidgets.QHBoxLayout()
        self.lineEdit_recordingDir = QtWidgets.QLineEdit(str(default_directory()), group)
        self.lineEdit_recordingDir.setReadOnly(True)
        self.button_recordingDir = QtWidgets.QPushButton("Browse...", group)
        row.addWidget(self.lineEdit_recordingDir, 1)
        row.addWidget(self.button_recordingDir)
        form.addRow("Folder:", row)

        self.spinBox_recordingCap = QtWidgets.QSpinBox(group)
        self.spinBox_recordingCap.setRange(1, 5000)
        self.spinBox_recordingCap.setSuffix(" GB")
        self.spinBox_recordingCap.setValue(DEFAULT_RECORDING_CAP_GB)
        form.addRow("Keep at most:", self.spinBox_recordingCap)

        self.label_recordingInfo = QtWidgets.QLabel(group)
        self.label_recordingInfo.setWordWrap(True)
        form.addRow(self.label_recordingInfo)

        # below Playback, above the stretch at the bottom
        self.verticalLayout_5.insertWidget(2, group)

        self.checkbox_recording.setChecked(True)
        self.checkbox_recording.toggled.connect(self._recording_settings_changed)
        self.spinBox_recordingCap.editingFinished.connect(self._recording_settings_changed)
        self.button_recordingDir.clicked.connect(self._choose_recording_dir)
        self._refresh_recording_info()

    def _refresh_recording_info(self):
        bytes_per_hour = SAMPLING_RATE * 2 * 3600
        hours = self.spinBox_recordingCap.value() * BYTES_PER_GB / bytes_per_hour
        text = ("%.1f GB per hour at %d kHz; %d GB holds about %.0f hours. "
                "The oldest files are deleted first."
                % (bytes_per_hour / BYTES_PER_GB, SAMPLING_RATE // 1000,
                   self.spinBox_recordingCap.value(), hours))
        repo = inside_git_worktree(self.lineEdit_recordingDir.text())
        if repo is not None:
            text += ("\n⚠ This folder is inside the git repository %s. Recordings of a "
                     "room do not belong in version control -- choose a folder outside it." % repo)
            self.label_recordingInfo.setStyleSheet("color: #c0392b;")
        else:
            self.label_recordingInfo.setStyleSheet("")
        self.label_recordingInfo.setText(text)

    def _choose_recording_dir(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Folder for continuous recordings", self.lineEdit_recordingDir.text())
        if chosen:
            self.lineEdit_recordingDir.setText(chosen)
            self._recording_settings_changed()

    def _recording_settings_changed(self, *_):
        self._refresh_recording_info()
        self.continuous_recording_changed.emit(self.checkbox_recording.isChecked(),
                                               self.lineEdit_recordingDir.text(),
                                               self.spinBox_recordingCap.value())

    @pyqtProperty(bool, notify=show_playback_changed) # type: ignore
    def show_playback(self) -> bool:
        return bool(self.checkbox_showPlayback.checkState())

    # slot
    # used when no audio input device has been found, to exit immediately
    def exitOnInit(self):
        QtWidgets.QApplication.instance().quit()

    # slot
    def input_device_changed(self, index):
        self._toolbar_view_model.recording = False

        success, index = AudioBackend().select_input_device(index)

        self.comboBox_inputDevice.setCurrentIndex(index)

        if not success:
            # Note: the error message is a child of the settings dialog, so that
            # that dialog remains on top when the error message is closed
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Input device error")
            error_message.showMessage("Impossible to use the selected input device, reverting to the previous one")

        # reset the channels
        channels = AudioBackend().get_readable_current_channels()

        self.comboBox_firstChannel.clear()
        self.comboBox_secondChannel.clear()

        for channel in channels:
            self.comboBox_firstChannel.addItem(channel)
            self.comboBox_secondChannel.addItem(channel)

        first_channel = AudioBackend().get_current_first_channel()
        self.comboBox_firstChannel.setCurrentIndex(first_channel)
        second_channel = AudioBackend().get_current_second_channel()
        self.comboBox_secondChannel.setCurrentIndex(second_channel)

        self._toolbar_view_model.recording = True

    # slot
    def first_channel_changed(self, index):
        self._toolbar_view_model.recording = False

        success, index = AudioBackend().select_first_channel(index)

        self.comboBox_firstChannel.setCurrentIndex(index)

        if not success:
            # Note: the error message is a child of the settings dialog, so that
            # that dialog remains on top when the error message is closed
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Input device error")
            error_message.showMessage("Impossible to use the selected channel as the first channel, reverting to the previous one")

        self._toolbar_view_model.recording = True

    # slot
    def second_channel_changed(self, index):
        self._toolbar_view_model.recording = False

        success, index = AudioBackend().select_second_channel(index)

        self.comboBox_secondChannel.setCurrentIndex(index)

        if not success:
            # Note: the error message is a child of the settings dialog, so that
            # that dialog remains on top when the error message is closed
            error_message = QtWidgets.QErrorMessage(self)
            error_message.setWindowTitle("Input device error")
            error_message.showMessage("Impossible to use the selected channel as the second channel, reverting to the previous one")

        self._toolbar_view_model.recording = True

    # slot
    def single_input_type_selected(self, checked):
        if checked:
            self.groupBox_second.setEnabled(False)
            AudioBackend().set_single_input()
            self.logger.info("Switching to single input")

    # slot
    def duo_input_type_selected(self, checked):
        if checked:
            self.groupBox_second.setEnabled(True)
            AudioBackend().set_duo_input()
            self.logger.info("Switching to difference between two inputs")

    # slot
    def show_playback_checkbox_changed(self, state: int) -> None:
        self.show_playback_changed.emit(bool(state))

    # slot
    def history_length_edit_finished(self) -> None:
        self.history_length_changed.emit(self.spinBox_historyLength.value())

    # method
    def saveState(self, settings):
        # for the input device, we search by name instead of index, since
        # we do not know if the device order stays the same between sessions
        settings.setValue("deviceName", self.comboBox_inputDevice.currentText())
        settings.setValue("firstChannel", self.comboBox_firstChannel.currentIndex())
        settings.setValue("secondChannel", self.comboBox_secondChannel.currentIndex())
        settings.setValue("duoInput", self.inputTypeButtonGroup.checkedId())
        settings.setValue("showPlayback", self.checkbox_showPlayback.checkState())
        settings.setValue("historyLength", self.spinBox_historyLength.value())
        if getattr(self, "_recording_override", None) is not None:
            # FRITURE_RECORDING_DIR was in force: keep the user's own choice
            enabled, folder, cap = self._recording_override
        else:
            enabled = self.checkbox_recording.isChecked()
            folder = self.lineEdit_recordingDir.text()
            cap = self.spinBox_recordingCap.value()
        settings.setValue("continuousRecording", enabled)
        settings.setValue("recordingDir", folder)
        settings.setValue("recordingCapGB", cap)

    # method
    def restoreState(self, settings):
        device_name = settings.value("deviceName", "")
        device_index = self.comboBox_inputDevice.findText(device_name)
        # change the device only if it exists in the device list
        if device_index >= 0:
            self.comboBox_inputDevice.setCurrentIndex(device_index)
            channel = settings.value("firstChannel", 0, type=int)
            self.comboBox_firstChannel.setCurrentIndex(channel)
            channel = settings.value("secondChannel", 0, type=int)
            self.comboBox_secondChannel.setCurrentIndex(channel)
            duo_input_id = settings.value("duoInput", 0, type=int)
            self.inputTypeButtonGroup.button(duo_input_id).setChecked(True)
        self.checkbox_showPlayback.setCheckState(settings.value("showPlayback", 0, type=int))
        self.spinBox_historyLength.setValue(settings.value("historyLength", 30, type=int))
        # need to emit this because setValue doesn't emit editFinished
        self.history_length_changed.emit(self.spinBox_historyLength.value())

        enabled = settings.value("continuousRecording", True, type=bool)
        folder = settings.value("recordingDir", str(default_directory()), type=str)
        cap = settings.value("recordingCapGB", DEFAULT_RECORDING_CAP_GB, type=int)
        # FRITURE_RECORDING_DIR overrides the folder for this run only
        # ("off" disables recording), and is never saved. Anything that runs
        # the application for a test sets it: a layout check that opened the
        # window with the user's settings left five empty files in the
        # user's own recording folder.
        self._recording_override = None
        override = os.environ.get("FRITURE_RECORDING_DIR")
        if override:
            self._recording_override = (enabled, folder, cap)
            if override.lower() == "off":
                enabled = False
            else:
                enabled, folder = True, override
        self.checkbox_recording.blockSignals(True)
        self.checkbox_recording.setChecked(enabled)
        self.checkbox_recording.blockSignals(False)
        self.lineEdit_recordingDir.setText(folder)
        self.spinBox_recordingCap.setValue(cap)
        # always emitted, changed or not: this is what starts the recorder
        self._recording_settings_changed()
