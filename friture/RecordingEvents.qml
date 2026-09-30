import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// The Recording Events dock: what happened in the continuous recordings,
// newest first. Click a row to replay it from a few seconds before; the
// lock keeps the files around it from being deleted to make room.
Rectangle {
    id: root
    required property var viewModel
    required property string fixedFont

    SystemPalette { id: systemPalette; colorGroup: SystemPalette.Active }
    color: systemPalette.window
    anchors.fill: parent
    clip: true

    function kindColor(kind, brief) {
        if (kind === "loud") return "#e08040"
        if (kind === "voice") return brief ? "#a09060" : "#e0c040"
        return "#70a8e0"
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 5

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Label {
                text: qsTr("Recording Events")
                font.pointSize: 13
                font.bold: true
                color: systemPalette.windowText
            }
            Label {
                text: root.viewModel.summary_text
                color: systemPalette.mid
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            Button {
                text: qsTr("Refresh")
                onClicked: root.viewModel.refresh()
            }
        }

        Flow {
            Layout.fillWidth: true
            spacing: 4
            CheckBox {
                objectName: "filter_loud"
                text: qsTr("Loud")
                checked: root.viewModel.show_loud
                onToggled: root.viewModel.set_filter("show_loud", checked)
            }
            CheckBox {
                text: qsTr("Voice-like")
                checked: root.viewModel.show_voice
                onToggled: root.viewModel.set_filter("show_voice", checked)
            }
            CheckBox {
                text: qsTr("...including brief ones")
                checked: root.viewModel.show_brief
                onToggled: root.viewModel.set_filter("show_brief", checked)
            }
            CheckBox {
                text: qsTr("Lines on/off")
                checked: root.viewModel.show_lines
                onToggled: root.viewModel.set_filter("show_lines", checked)
            }
            CheckBox {
                text: qsTr("Protected only")
                checked: root.viewModel.protected_only
                onToggled: root.viewModel.set_filter("protected_only", checked)
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 60
            color: "#18181c"
            radius: 3
            clip: true

            ListView {
                id: list
                objectName: "event_list"
                anchors.fill: parent
                anchors.margins: 4
                clip: true
                model: root.viewModel.events
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                delegate: Item {
                    width: list.width
                    height: 24

                    MouseArea {
                        id: rowMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: root.viewModel.play(model.key)
                        ToolTip.visible: containsMouse
                        ToolTip.delay: 500
                        ToolTip.text: qsTr("Replay from a few seconds before this")
                    }
                    Rectangle {
                        anchors.fill: parent
                        radius: 2
                        color: rowMouse.containsMouse ? "#2c2c36" : "transparent"
                    }
                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 4
                        anchors.rightMargin: 4
                        spacing: 8
                        Label {
                            text: model.time_text
                            font.family: root.fixedFont
                            color: "#e6e6eb"
                            Layout.preferredWidth: 130
                        }
                        Label {
                            text: model.kind_label
                            color: root.kindColor(model.kind, model.brief)
                            font.bold: !model.brief
                            Layout.preferredWidth: 110
                        }
                        Label {
                            text: model.detail
                            color: "#b8b8c2"
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        // the lock: filled when the files around this event are kept
                        Rectangle {
                            objectName: "protect_button"
                            Layout.preferredWidth: 64
                            Layout.preferredHeight: 18
                            radius: 3
                            color: model.protected ? "#3a5a2a" : "transparent"
                            border.color: model.protected ? "#80c060" : "#5a5a66"
                            Label {
                                anchors.centerIn: parent
                                text: model.protected ? qsTr("kept") : qsTr("keep")
                                font.pointSize: 8
                                color: model.protected ? "#c8f0b0" : "#9a9aa6"
                            }
                            MouseArea {
                                anchors.fill: parent
                                onClicked: root.viewModel.toggle_protect(model.key)
                                ToolTip.visible: containsMouse
                                hoverEnabled: true
                                ToolTip.text: model.protected
                                    ? qsTr("Let these files be deleted again when space is needed")
                                    : qsTr("Keep the files around this event: they will not be deleted to make room")
                            }
                        }
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: list.count === 0
                text: root.viewModel.status_text
                color: "#8a8a96"
            }
        }

        Label {
            text: root.viewModel.protected_text
            color: systemPalette.mid
            font.pointSize: 8
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
    }
}
