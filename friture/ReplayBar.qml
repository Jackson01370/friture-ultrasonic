import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// The replay bar. Shown whenever the application is looking at a recording
// instead of the room -- in amber, with the recording's own clock time in
// large type, because mistaking a replay for what is happening now is the
// one confusion this mode must never allow.
Rectangle {
    id: bar
    objectName: "replay_bar"
    required property var viewModel
    required property string fixedFont

    visible: viewModel.active
    implicitHeight: visible ? column.implicitHeight + 12 : 0
    color: "#3a2c08"
    border.color: "#e0a020"
    border.width: 1

    ColumnLayout {
        id: column
        anchors.fill: parent
        anchors.margins: 6
        spacing: 4

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Label {
                text: qsTr("REPLAY")
                font.bold: true
                font.pointSize: 12
                color: "#ffc040"
            }
            Label {
                objectName: "replay_position"
                text: bar.viewModel.position_text
                font.family: bar.fixedFont
                font.pointSize: 13
                font.bold: true
                color: "#fff0c8"
            }
            Label {
                objectName: "replay_status"
                text: bar.viewModel.status_text
                color: "#e8d8a8"
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            Button {
                text: qsTr("Back to live")
                onClicked: bar.viewModel.toggle()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 4
            visible: bar.viewModel.position_text !== ""

            Button { text: "-60 s"; onClicked: bar.viewModel.step(-60) }
            Button { text: "-10 s"; onClicked: bar.viewModel.step(-10) }
            Button {
                objectName: "replay_play_pause"
                text: bar.viewModel.playing ? qsTr("Pause") : qsTr("Play")
                onClicked: bar.viewModel.play_pause()
            }
            Button { text: "+10 s"; onClicked: bar.viewModel.step(10) }
            Button { text: "+60 s"; onClicked: bar.viewModel.step(60) }
            Button {
                text: qsTr("Latest")
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Jump to a few seconds ago and follow the recording as it is written")
                onClicked: bar.viewModel.latest()
            }

            Label {
                text: bar.viewModel.start_text
                font.family: bar.fixedFont
                font.pointSize: 8
                color: "#c8b888"
            }

            // the span of all recordings; the lighter stretches hold audio,
            // the dark ones between them are gaps
            Item {
                Layout.fillWidth: true
                Layout.minimumWidth: 120
                implicitHeight: 26

                Rectangle {
                    id: track
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    height: 8
                    radius: 3
                    color: "#1c1606"
                    Repeater {
                        model: bar.viewModel.ranges
                        delegate: Rectangle {
                            x: modelData[0] * track.width
                            width: Math.max(2, (modelData[1] - modelData[0]) * track.width)
                            height: track.height
                            radius: 3
                            color: "#a07820"
                        }
                    }
                }

                Slider {
                    id: slider
                    objectName: "replay_slider"
                    anchors.fill: parent
                    from: 0
                    to: 1
                    background: Item {}
                    // follow the playing position except while being dragged
                    Binding on value {
                        value: bar.viewModel.fraction
                        when: !slider.pressed
                    }
                    onPressedChanged: if (!pressed) bar.viewModel.seek(value)
                }
            }

            Label {
                text: bar.viewModel.end_text
                font.family: bar.fixedFont
                font.pointSize: 8
                color: "#c8b888"
            }

            // 8x was offered and measured: 3.9x is what this machine reaches
            // with the usual docks open, so offering it would promise twice
            // what it delivers. 4x reached 3.8x, and the status line says so.
            ComboBox {
                objectName: "replay_speed"
                model: ["1x", "2x", "4x"]
                currentIndex: [1, 2, 4].indexOf(bar.viewModel.speed)
                onActivated: bar.viewModel.set_speed([1, 2, 4][index])
                implicitWidth: 70
            }
        }
    }
}
