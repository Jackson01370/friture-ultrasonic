import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.2
import Friture 1.0

Rectangle { // eventually move to ApplicationWindow
    id: mainWindow
    anchors.fill: parent
    // title: qsTr("Friture") // ApplicationWindow
    // icon.source: "qrc:/images-src/window-icon.svg" // ApplicationWindow

    required property MainWindowViewModel main_window_view_model
    required property string fixedFont

    ColumnLayout { // remove once we use ApplicationWindow
        anchors.fill: parent
        spacing: 0

        ToolBar {
            id: toolBar
            Layout.fillWidth: true // remove once we use ApplicationWindow

            RowLayout {
                spacing: 0

                ToolButton {
                    id: startButton
                    checkable: true
                    checked: mainWindow.main_window_view_model.toolbar_view_model.recording
                    icon.source: startButton.checked ? "qrc:/images-src/stop.svg" : "qrc:/images-src/start.svg"
                    text: startButton.checked ? qsTr("Stop") : qsTr("Start")
                    ToolTip.text: qsTr("Start/Stop")
                    icon.height: 32
                    icon.width: 32
                    //shortcut: "Space"
                    onClicked: {
                        mainWindow.main_window_view_model.toolbar_view_model.recording_toggle()
                    }
                }
                ToolButton {
                    id: newDockButton
                    icon.source: "qrc:/images-src/new-dock.svg"
                    text: qsTr("New dock")
                    ToolTip.text: qsTr("Add a new dock to Friture window")
                    icon.height: 32
                    icon.width: 32
                    onClicked: {
                        mainWindow.main_window_view_model.toolbar_view_model.new_dock()
                    }
                }
                ToolButton {
                    id: settingsButton
                    icon.source: "qrc:/images-src/tools.svg"
                    text: qsTr("Settings")
                    ToolTip.text: qsTr("Display settings dialog")
                    icon.height: 32
                    icon.width: 32
                    onClicked: {
                        mainWindow.main_window_view_model.toolbar_view_model.settings()
                    }
                }
                ToolButton {
                    id: aboutButton
                    icon.source: "qrc:/images-src/window-icon.svg"
                    text: qsTr("About Friture")
                    icon.height: 32
                    icon.width: 32
                    onClicked: {
                        mainWindow.main_window_view_model.toolbar_view_model.about()
                    }
                }

                // The continuous recorder. A drive recorder that has quietly
                // stopped is the failure worth guarding against, so the state
                // is always on screen: red while writing, amber when it cannot.
                Row {
                    id: recorderIndicator
                    objectName: "recorder_indicator"
                    Layout.leftMargin: 18
                    spacing: 6
                    readonly property string state_: mainWindow.main_window_view_model.toolbar_view_model.recorder_state
                    visible: state_ !== "off"

                    Rectangle {
                        width: 12
                        height: 12
                        radius: 6
                        anchors.verticalCenter: parent.verticalCenter
                        color: recorderIndicator.state_ === "recording" ? "#e0302a"
                             : recorderIndicator.state_ === "error" ? "#e0a020" : "#8a8a96"
                    }
                    Label {
                        objectName: "recorder_text"
                        anchors.verticalCenter: parent.verticalCenter
                        text: mainWindow.main_window_view_model.toolbar_view_model.recorder_text
                        font.family: mainWindow.fixedFont
                        color: recorderIndicator.state_ === "error" ? "#c07000" : palette.windowText
                    }
                }
            }
        }

        MainWindow {
            id: centralWidget
            Layout.fillWidth: true
            Layout.fillHeight: true
            fixedFont: mainWindow.fixedFont
            main_window_view_model: mainWindow.main_window_view_model
        }
    }
}
