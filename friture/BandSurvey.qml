import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// The Band Survey dock: what stands out in the whole spectrum, measured
// against each line's OWN neighbourhood rather than one wideband floor --
// which is what "a bump on the display" means when the noise itself slopes.
// Click a line and the Listen band moves there, so the Digital Decode dock
// and the plot overlays follow.
Rectangle {
    id: root
    required property var viewModel
    required property string fixedFont

    SystemPalette { id: systemPalette; colorGroup: SystemPalette.Active }
    color: systemPalette.window
    anchors.fill: parent

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 6

        RowLayout {
            Layout.fillWidth: true
            spacing: 12
            Label {
                text: qsTr("Band Survey")
                font.pointSize: 14
                font.bold: true
                color: systemPalette.windowText
            }
            Label {
                text: root.viewModel.range_text
                color: systemPalette.mid
            }
            Item { Layout.fillWidth: true }
        }

        Label {
            text: root.viewModel.status_text
            color: systemPalette.windowText
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        // the lines themselves, strongest first
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(260, 22 * Math.max(root.viewModel.lines.length, 1) + 26)
            color: "#18181c"
            radius: 3

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 0

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    Label { text: qsTr("frequency"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 110 }
                    Label { text: qsTr("over its floor"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 110 }
                    Label { text: qsTr("level"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 90 }
                    Label { text: qsTr("over time"); color: "#8a8a96"; font.pointSize: 8; Layout.fillWidth: true }
                }

                ListView {
                    id: lineList
                    objectName: "survey_lines"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: root.viewModel.lines

                    delegate: Rectangle {
                        width: lineList.width
                        height: 22
                        color: hover.hovered ? "#2a2a34" : "transparent"

                        HoverHandler { id: hover }
                        TapHandler {
                            onTapped: root.viewModel.tune(modelData.frequency)
                        }
                        ToolTip.visible: hover.hovered
                        ToolTip.text: qsTr("Click to point the Listen band here, so the other docks follow.")

                        RowLayout {
                            anchors.fill: parent
                            spacing: 0
                            Label {
                                text: modelData.frequency_text
                                font.family: root.fixedFont
                                color: "#e6e6eb"
                                Layout.preferredWidth: 110
                            }
                            Label {
                                text: modelData.excess_text
                                font.family: root.fixedFont
                                // orange for something well clear of its
                                // neighbourhood, blue for a mild rise
                                color: modelData.excess >= 10 ? "#ffaa46" : "#78bef0"
                                Layout.preferredWidth: 110
                            }
                            Label {
                                text: modelData.level_text
                                font.family: root.fixedFont
                                color: "#9a9aa6"
                                Layout.preferredWidth: 90
                            }
                            Label {
                                text: modelData.steadiness_text
                                color: modelData.steady ? "#9a9aa6" : "#e0c060"
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                        }
                    }
                }
            }
        }

        Label {
            text: qsTr("The shape of the noise itself, so a line can be told from a slope:")
            color: systemPalette.mid
            font.pointSize: 8
            Layout.fillWidth: true
            visible: root.viewModel.shape.length > 0
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(190, 18 * Math.max(root.viewModel.shape.length, 1) + 12)
            color: "#18181c"
            radius: 3
            visible: root.viewModel.shape.length > 0

            ListView {
                objectName: "survey_shape"
                anchors.fill: parent
                anchors.margins: 6
                clip: true
                model: root.viewModel.shape
                delegate: Row {
                    spacing: 8
                    Label {
                        text: modelData.band_text
                        font.family: root.fixedFont
                        font.pointSize: 8
                        color: "#9a9aa6"
                        width: 150
                    }
                    Rectangle {
                        width: Math.max(2, modelData.bar * 320)
                        height: 11
                        y: 2
                        color: "#4a8a6a"
                        radius: 2
                    }
                    Label {
                        text: modelData.detail_text
                        font.family: root.fixedFont
                        font.pointSize: 8
                        color: "#8a8a96"
                    }
                }
            }
        }

        Label {
            text: root.viewModel.help_text
            color: systemPalette.windowText
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        Item { Layout.fillHeight: true }
    }
}
