import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// The Band Survey dock: what stands out in the whole spectrum, measured
// against each line's OWN neighbourhood rather than one wideband floor --
// which is what "a bump on the display" means when the noise itself slopes.
// Click a line and the Listen band moves onto it, as wide as the signal
// actually is, so the Digital Decode dock and the plot overlays follow.
//
// EVERYTHING SIZES TO THE DOCK. A dock is whatever height the user's layout
// left it, and fixed heights simply fell off the bottom of the window. So
// the list takes the space that is left, and the parts that are nice to
// have -- the noise-shape table, the help text -- appear only when there is
// room for them.
Rectangle {
    id: root
    required property var viewModel
    required property string fixedFont

    // below these the section in question would be a sliver, so it is
    // dropped rather than shown uselessly
    readonly property bool roomForShape: height > 340
    readonly property bool roomForHelp: height > 250

    SystemPalette { id: systemPalette; colorGroup: SystemPalette.Active }
    color: systemPalette.window
    anchors.fill: parent
    clip: true

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 8
        spacing: 5

        RowLayout {
            Layout.fillWidth: true
            spacing: 10
            Label {
                text: qsTr("Band Survey")
                font.pointSize: 13
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
            elide: Text.ElideRight
        }

        // the lines themselves, lowest frequency first
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 70
            color: "#18181c"
            radius: 3
            clip: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 5
                spacing: 0

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 0
                    Label { text: qsTr("frequency"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 96 }
                    Label { text: qsTr("width"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 78 }
                    Label { text: qsTr("over its floor"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 84 }
                    Label { text: qsTr("level"); color: "#8a8a96"; font.pointSize: 8; Layout.preferredWidth: 74 }
                    Label { text: qsTr("over time"); color: "#8a8a96"; font.pointSize: 8; Layout.fillWidth: true }
                }

                ListView {
                    id: lineList
                    objectName: "survey_lines"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: root.viewModel.lines
                    boundsBehavior: Flickable.StopAtBounds
                    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

                    delegate: Item {
                        width: lineList.width
                        height: 21

                        // A plain MouseArea, not a TapHandler: the row must
                        // answer a click even while its neighbours are being
                        // inserted and removed around it.
                        MouseArea {
                            id: rowMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: root.viewModel.tune(model.frequency, model.width)
                            ToolTip.visible: containsMouse
                            ToolTip.delay: 400
                            ToolTip.text: qsTr("Listen to %1, %2 wide -- the other docks follow.")
                                .arg(model.frequency_text).arg(model.width_text)
                        }

                        Rectangle {
                            anchors.fill: parent
                            color: rowMouse.containsMouse ? "#33333f" : "transparent"
                            radius: 2
                        }

                        RowLayout {
                            anchors.fill: parent
                            spacing: 0
                            Label {
                                text: model.frequency_text
                                font.family: root.fixedFont
                                color: "#e6e6eb"
                                Layout.preferredWidth: 96
                            }
                            Label {
                                text: model.width_text
                                font.family: root.fixedFont
                                color: "#9a9aa6"
                                Layout.preferredWidth: 78
                            }
                            Label {
                                text: model.excess_text
                                font.family: root.fixedFont
                                // orange for something well clear of its
                                // neighbourhood, blue for a mild rise
                                color: model.excess >= 10 ? "#ffaa46" : "#78bef0"
                                Layout.preferredWidth: 84
                            }
                            Label {
                                text: model.level_text
                                font.family: root.fixedFont
                                color: "#9a9aa6"
                                Layout.preferredWidth: 74
                            }
                            Label {
                                text: model.steadiness_text
                                color: model.steady ? "#9a9aa6" : "#e0c060"
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
            visible: root.roomForShape && root.viewModel.shape.length > 0
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.min(150, 17 * Math.max(root.viewModel.shape.length, 1) + 10)
            color: "#18181c"
            radius: 3
            clip: true
            visible: root.roomForShape && root.viewModel.shape.length > 0

            ListView {
                objectName: "survey_shape"
                anchors.fill: parent
                anchors.margins: 5
                clip: true
                model: root.viewModel.shape
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
                delegate: Row {
                    spacing: 7
                    Label {
                        text: modelData.band_text
                        font.family: root.fixedFont
                        font.pointSize: 8
                        color: "#9a9aa6"
                        width: 132
                    }
                    Rectangle {
                        width: Math.max(2, modelData.bar * 260)
                        height: 10
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
            color: systemPalette.mid
            font.pointSize: 8
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            maximumLineCount: root.height > 420 ? 4 : 2
            elide: Text.ElideRight
            visible: root.roomForHelp
        }
    }
}
