import QtQuick 2.15
import QtQuick.Layouts 1.15
import QtQuick.Shapes 1.15
import Friture 1.0

Item {
    id: scopePlotArea

    SystemPalette { id: systemPalette; colorGroup: SystemPalette.Active }

    required property Axis vertical_axis
    required property Axis horizontal_axis

    // Band listening. Plots without a frequency axis leave these at their
    // defaults and neither the overlay nor the click handler does anything.
    property var listen_band: null
    property string freq_axis: "" // "vertical", "horizontal", or "" for none

    readonly property bool has_freq_axis: listen_band !== null && freq_axis !== ""
    readonly property Axis freq_axis_object: freq_axis === "vertical" ? vertical_axis : horizontal_axis
    readonly property bool freq_is_vertical: freq_axis === "vertical"

    // How far either side of an edge counts as grabbing it. The band can be a
    // couple of pixels tall at high frequencies on a mel axis, so this is
    // sized for a mouse rather than for the drawing.
    readonly property real band_grab_margin: 9

    // The frequency drawn at a position in this item, in Hz.
    function frequency_at(px, py) {
        var relative = freq_is_vertical
            ? 1. - Math.min(Math.max(py, 0), height) / height
            : Math.min(Math.max(px, 0), width) / width
        return freq_axis_object.coordinate_transform.toPlot(relative)
    }

    default property alias content: plotItemPlaceholder.children

    PlotBackground {
        anchors.fill: parent
    }

    PlotGrid {
        anchors.fill: parent

        vertical_scale_division: scopePlotArea.vertical_axis.scale_division
        show_minor_vertical: scopePlotArea.vertical_axis.show_minor_grid_lines
        horizontal_scale_division: scopePlotArea.horizontal_axis.scale_division
        show_minor_horizontal: scopePlotArea.horizontal_axis.show_minor_grid_lines
    }

    Item {
        id: plotItemPlaceholder
        anchors.fill: parent
    }

    // The band being listened to, mirrored back onto the plot -- and the
    // handle for moving and resizing it.
    Rectangle {
        id: bandOverlay
        objectName: "listen_band_overlay"
        visible: scopePlotArea.has_freq_axis && scopePlotArea.listen_band.enabled

        // The comma is doing real work: reading range_min/range_max is what
        // ties these bindings to the axis. toScreen() is a slot, so the
        // engine cannot see the axis state it reads inside, and without this
        // the overlay would stay put when the plot's frequency range or
        // scale changes.
        readonly property real axisEpoch: scopePlotArea.freq_axis_object.range_min + scopePlotArea.freq_axis_object.range_max
        readonly property real relLo: scopePlotArea.has_freq_axis
            ? (axisEpoch, scopePlotArea.freq_axis_object.coordinate_transform.toScreen(scopePlotArea.listen_band.f_lo))
            : 0.
        readonly property real relHi: scopePlotArea.has_freq_axis
            ? (axisEpoch, scopePlotArea.freq_axis_object.coordinate_transform.toScreen(scopePlotArea.listen_band.f_hi))
            : 0.

        x: scopePlotArea.freq_axis === "horizontal" ? relLo * scopePlotArea.width : 0
        width: scopePlotArea.freq_axis === "horizontal" ? (relHi - relLo) * scopePlotArea.width : scopePlotArea.width
        // Screen y grows downwards, so the top edge is the high frequency.
        y: scopePlotArea.freq_axis === "vertical" ? (1. - relHi) * scopePlotArea.height : 0
        height: scopePlotArea.freq_axis === "vertical" ? (relHi - relLo) * scopePlotArea.height : scopePlotArea.height

        color: bandDrag.pressed ? "#5548c0ff" : "#3348c0ff"
        border.color: "#cc48c0ff"
        border.width: 1
    }

    // Grabbing the band. On top of the plot's own mouse area, so inside the
    // band a press drags it and outside the crosshair and click-to-centre
    // carry on exactly as before.
    MouseArea {
        id: bandDrag
        z: 2
        enabled: bandOverlay.visible
        visible: enabled
        hoverEnabled: true

        // The band, grown by the grab margin along the frequency axis.
        x: scopePlotArea.freq_is_vertical ? 0 : bandOverlay.x - scopePlotArea.band_grab_margin
        y: scopePlotArea.freq_is_vertical ? bandOverlay.y - scopePlotArea.band_grab_margin : 0
        width: scopePlotArea.freq_is_vertical
            ? scopePlotArea.width
            : bandOverlay.width + 2 * scopePlotArea.band_grab_margin
        height: scopePlotArea.freq_is_vertical
            ? bandOverlay.height + 2 * scopePlotArea.band_grab_margin
            : scopePlotArea.height

        // 0 none, 1 move the band, 2 pull an edge
        property int grabbed: 0
        // Where in the band it was grabbed, so it stays under the cursor
        // instead of jumping its centre there on the first press.
        property real grabOffset: 0

        readonly property real span: scopePlotArea.freq_is_vertical ? height : width
        readonly property real bandSpan: scopePlotArea.freq_is_vertical ? bandOverlay.height : bandOverlay.width

        // Reaching in from each edge. Capped at a third of the band so that
        // however thin it gets there is always some of it left to grab and
        // move, rather than the two edge zones meeting in the middle.
        readonly property real edgeZone: scopePlotArea.band_grab_margin
            + Math.min(scopePlotArea.band_grab_margin, bandSpan / 3.)

        function zoneAt(position) {
            if (position <= edgeZone || position >= span - edgeZone)
                return 2
            return 1
        }

        cursorShape: {
            if (grabbed === 1 || (grabbed === 0 && zoneAt(scopePlotArea.freq_is_vertical ? mouseY : mouseX) === 1))
                return pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
            return scopePlotArea.freq_is_vertical ? Qt.SizeVerCursor : Qt.SizeHorCursor
        }

        onPressed: {
            var along = scopePlotArea.freq_is_vertical ? mouse.y : mouse.x
            grabbed = zoneAt(along)
            if (grabbed === 1) {
                var centre = scopePlotArea.freq_is_vertical
                    ? bandOverlay.y + bandOverlay.height / 2. - y
                    : bandOverlay.x + bandOverlay.width / 2. - x
                // in pixels, not Hz: on a mel or log axis a fixed number of
                // hertz is not a fixed distance, and it is the distance that
                // has to stay put under the cursor
                grabOffset = along - centre
            }
        }

        onReleased: grabbed = 0

        onPositionChanged: {
            if (grabbed === 0)
                return

            var point = mapToItem(scopePlotArea, mouse.x, mouse.y)
            if (grabbed === 2) {
                scopePlotArea.listen_band.drag_edge(
                    scopePlotArea.frequency_at(point.x, point.y))
            } else if (scopePlotArea.freq_is_vertical) {
                scopePlotArea.listen_band.click_center(
                    scopePlotArea.frequency_at(point.x, point.y - grabOffset))
            } else {
                scopePlotArea.listen_band.click_center(
                    scopePlotArea.frequency_at(point.x - grabOffset, point.y))
            }
        }
    }

    Rectangle {
        id: plotBorder
        anchors.fill: parent
        border.color: systemPalette.mid
        border.width: 1
        color: "transparent"
    }

    Item
    {
        id: crosshair
        visible: plotMouseArea.pressed
        anchors.fill: parent

        property double posX: Math.min(Math.max(plotMouseArea.mouseX, 0), scopePlotArea.width)
        property double posY: Math.min(Math.max(plotMouseArea.mouseY, 0), scopePlotArea.height)

        property double relativePosX: posX / scopePlotArea.width
        property double relativePosY: posY / scopePlotArea.height

        Rectangle
        {
            x: Math.min(crosshair.posX + 4, scopePlotArea.width - width)
            y: Math.max(crosshair.posY - mouseText.contentHeight - 4, 0)
            implicitWidth: mouseText.contentWidth
            implicitHeight: mouseText.contentHeight
            color: systemPalette.base

            Text {
                id: mouseText

                property double dataX: scopePlotArea.horizontal_axis.coordinate_transform.toPlot(crosshair.relativePosX)
                property double dataY: scopePlotArea.vertical_axis.coordinate_transform.toPlot(1. - crosshair.relativePosY)

                text: scopePlotArea.horizontal_axis.formatTracker(dataX)  + ", " + scopePlotArea.vertical_axis.formatTracker(dataY)
                color: systemPalette.windowText
            }
        }

        Shape {
            ShapePath {
                strokeWidth: 1
                strokeColor: systemPalette.windowText
                fillColor: "transparent"
                PathMove { x: crosshair.posX; y: 0 }
                PathLine { x: crosshair.posX; y: scopePlotArea.height }
                PathMove { x: 0; y: crosshair.posY }
                PathLine { x: scopePlotArea.width; y: crosshair.posY }
            }
        }
    }

    MouseArea {
        id: plotMouseArea
        anchors.fill: parent
        cursorShape: Qt.CrossCursor

        // Click a peak to hear it. The band is CENTRED on the clicked
        // frequency and the width is left alone, so peak after peak can be
        // clicked without touching anything else. Presses that land on the
        // band itself never reach here -- those are drags.
        onClicked: {
            if (!scopePlotArea.has_freq_axis)
                return
            scopePlotArea.listen_band.click_center(
                scopePlotArea.frequency_at(mouse.x, mouse.y))
        }
    }
}
