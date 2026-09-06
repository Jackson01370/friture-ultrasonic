import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

// The Digital Decode dock: what the detector sees in the listen band, what
// the clock recovery makes of it, and the bits -- when, and only when, there
// is a clock worth believing. In an analog mode (FM / AM / PM) the bits and
// the symbol strip give way to the demodulated waveform. Everything shown
// here comes from the view model; the numbers are made in
// friture.demod.decoder.
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
        spacing: 8

        RowLayout {
            spacing: 12
            Layout.fillWidth: true

            Label {
                text: root.viewModel.mode_text
                font.pointSize: 14
                font.bold: true
                color: systemPalette.windowText
                ToolTip.visible: modeHover.hovered
                ToolTip.text: root.viewModel.mode_hint
                HoverHandler { id: modeHover }
            }
            Label {
                text: root.viewModel.band_text
                color: systemPalette.windowText
            }
            Label {
                text: root.viewModel.range_text
                color: systemPalette.mid
            }
            Item { Layout.fillWidth: true }
        }

        // what the detector sees, before any clock
        Label {
            text: root.viewModel.signal_text
            color: systemPalette.windowText
            Layout.fillWidth: true
            elide: Text.ElideRight
        }

        // the clock's verdict: always the baud AND its confidence, never the
        // baud alone. Orange while there is nothing to trust, so a candidate
        // rate is not mistaken for a result.
        Label {
            text: root.viewModel.decode_text
            font.bold: root.viewModel.locked
            color: (root.viewModel.locked || root.viewModel.analog) ? systemPalette.windowText : "#d08040"
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        // the bit stream, newest at the right, in groups of eight. A '.' is
        // a symbol that could not be decided, kept in place so the bits
        // after it do not shift.
        Label {
            text: root.viewModel.bits
            font.family: root.fixedFont
            font.pointSize: 14
            color: systemPalette.windowText
            Layout.fillWidth: true
            wrapMode: Text.WrapAnywhere
            visible: !root.viewModel.analog && root.viewModel.bits !== ""
        }

        // The recovered symbols drawn as a multi-level trace: the eye, on
        // screen. One cell per symbol, newest at the right; the cell sits
        // higher for a higher symbol value (two rows for OOK / FSK / DBPSK,
        // four for 4-FSK / DQPSK), a grey full-height bar for undecided.
        // The OPACITY is the per-symbol agreement, so a marginal decode
        // fades instead of looking as solid as a clean one. A repeating
        // preamble shows as a pattern; random data looks random; noise
        // never gets this far.
        Canvas {
            id: strip
            objectName: "symbol_strip"
            Layout.fillWidth: true
            Layout.preferredHeight: 48
            visible: !root.viewModel.analog && root.viewModel.bits !== ""

            property var symbols: root.viewModel.symbols
            property var agreements: root.viewModel.agreements
            property int levels: root.viewModel.levels

            onSymbolsChanged: requestPaint()
            onAgreementsChanged: requestPaint()
            onLevelsChanged: requestPaint()
            onWidthChanged: requestPaint()

            onPaint: {
                var ctx = getContext("2d");
                var w = width;
                var h = height;
                ctx.fillStyle = "#18181c";
                ctx.fillRect(0, 0, w, h);
                var lv = Math.max(2, levels);
                var cellH = h / lv;
                ctx.strokeStyle = "#46464e";
                ctx.lineWidth = 1;
                for (var r = 1; r < lv; r++) {
                    ctx.beginPath();
                    ctx.moveTo(0, r * cellH);
                    ctx.lineTo(w, r * cellH);
                    ctx.stroke();
                }

                var n = symbols ? symbols.length : 0;
                if (n === 0)
                    return;
                var cw = w / n;
                var tints = ["255, 190, 90", "120, 220, 255", "140, 230, 140", "220, 140, 255"];
                for (var i = 0; i < n; i++) {
                    var s = symbols[i];
                    var a = (agreements && i < agreements.length) ? agreements[i] : 0;
                    var x = i * cw;
                    var cell = Math.max(cw - 0.5, 0.5);
                    if (s < 0) {
                        ctx.fillStyle = "rgba(110, 110, 118, 0.47)";
                        ctx.fillRect(x, 0, cell, h);
                        continue;
                    }
                    // Agreement 1/levels is a coin flip, so that is the
                    // floor of the visible range rather than 0.
                    var floor = 1.0 / lv;
                    var alpha = (60 + 195 * Math.min(Math.max((a - floor) / (1 - floor), 0), 1)) / 255;
                    ctx.fillStyle = "rgba(" + tints[s % tints.length] + ", " + alpha + ")";
                    ctx.fillRect(x, h - (s + 1) * cellH, cell, cellH);
                }
            }
        }

        Label {
            text: qsTr("Symbols, newest at the right: higher = larger symbol value, grey = undecided. Faint cells were marginal.")
            color: systemPalette.mid
            font.pointSize: 8
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            visible: !root.viewModel.analog && root.viewModel.bits !== ""
        }

        // The demodulated waveform of an analog mode: the last second,
        // newest at the right, with the range it spans at the left edge.
        Item {
            visible: root.viewModel.analog
            Layout.fillWidth: true
            Layout.preferredHeight: 150

            Label {
                text: root.viewModel.trace_hi_text
                anchors.top: parent.top
                anchors.left: parent.left
                color: systemPalette.mid
                font.pointSize: 8
            }
            Label {
                text: root.viewModel.trace_lo_text
                anchors.bottom: parent.bottom
                anchors.left: parent.left
                color: systemPalette.mid
                font.pointSize: 8
            }

            Canvas {
                id: waveform
                objectName: "analog_trace"
                anchors.fill: parent
                anchors.leftMargin: 84

                property var trace: root.viewModel.trace

                onTraceChanged: requestPaint()
                onWidthChanged: requestPaint()

                onPaint: {
                    var ctx = getContext("2d");
                    var w = width;
                    var h = height;
                    ctx.fillStyle = "#18181c";
                    ctx.fillRect(0, 0, w, h);
                    ctx.strokeStyle = "#46464e";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(0, h / 2);
                    ctx.lineTo(w, h / 2);
                    ctx.stroke();

                    var n = trace ? trace.length : 0;
                    if (n < 2)
                        return;
                    var lo = trace[0], hi = trace[0];
                    for (var i = 1; i < n; i++) {
                        if (trace[i] < lo) lo = trace[i];
                        if (trace[i] > hi) hi = trace[i];
                    }
                    var span = hi - lo;
                    if (span <= 0) span = 1;
                    ctx.strokeStyle = "#78dcff";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    for (var k = 0; k < n; k++) {
                        var x = k * (w - 1) / (n - 1);
                        var y = 4 + (h - 8) * (1 - (trace[k] - lo) / span);
                        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                    }
                    ctx.stroke();
                }
            }
        }

        Label {
            text: qsTr("The demodulated waveform, last second, newest at the right.")
            color: systemPalette.mid
            font.pointSize: 8
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
            visible: root.viewModel.analog
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
