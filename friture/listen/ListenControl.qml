import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Window 2.2
import QtQuick.Layouts 1.15
// Binding.restoreMode lives in QtQml, and QtQuick 2.15 does not re-export it
import QtQml 2.15
import Friture 1.0

// Flow, not RowLayout: laid out in a row this many controls come to well over
// a thousand pixels, and a RowLayout hands that up as a minimum width -- which
// would push the plot area wider than the window and hold the window open at
// that size. A Flow wraps onto a second line instead, so the bar costs one
// control's worth of width and no more.
Flow {
    id: listenRow

    required property ListenBandViewModel viewModel

    spacing: 6

    SystemPalette { id: systemPalette; colorGroup: SystemPalette.Active }

    // a label and its field, kept together so a wrap never separates them
    component Field: Row {
        property alias label: caption.text
        default property alias fields: fieldRow.children
        spacing: 4

        Text {
            id: caption
            color: systemPalette.text
            anchors.verticalCenter: parent.verticalCenter
        }

        Row { id: fieldRow; spacing: 4 }
    }

    CheckBox {
        id: enableBox
        text: qsTr("Listen")
        checked: listenRow.viewModel.enabled
        ToolTip.visible: hovered
        ToolTip.text: qsTr("Hear only the selected band. Click a peak on a spectrum or spectrogram to centre it there, or drag the box to move it and its edges to resize it.")
        onToggled: listenRow.viewModel.enabled = checked
    }

    // Both switches are also flipped from the model -- turning Live on turns
    // Listen on with it, and a monitor that finds no output device turns
    // itself back off. A plain `checked:` binding dies the first time the box
    // is clicked, so it would not show either.
    Binding {
        target: enableBox
        property: "checked"
        value: listenRow.viewModel.enabled
        restoreMode: Binding.RestoreNone
    }

    ComboBox {
        id: modeBox
        model: [qsTr("Band-pass"), qsTr("Heterodyne")]
        visible: listenRow.viewModel.enabled
        currentIndex: listenRow.viewModel.mode
        ToolTip.visible: hovered
        ToolTip.text: qsTr("Band-pass keeps the band where it is, so the pitch is unchanged - but playback runs at 48 kHz, so anything selected above ~24 kHz is silent. Heterodyne moves the band down to 0 Hz instead, which is what makes ultrasound audible, at the cost of changing the pitch.")
        onActivated: listenRow.viewModel.mode = currentIndex
    }

    Field {
        label: qsTr("Centre")
        visible: listenRow.viewModel.enabled

        SpinBox {
            id: centreBox
            width: 120
            from: 0
            to: listenRow.viewModel.max_freq_hz
            stepSize: 100
            editable: true
            textFromValue: function (value, locale) { return value + " Hz" }
            valueFromText: function (text, locale) { return Number.fromLocaleString(locale, text.replace(" Hz", "").trim()) }
            onValueModified: listenRow.viewModel.centre_hz = value
        }
    }

    // A SpinBox breaks a plain `value:` binding as soon as its own arrows are
    // used, and the whole point of this feature is that a click on the plot
    // moves the centre afterwards. An explicit Binding survives that.
    Binding {
        target: centreBox
        property: "value"
        value: listenRow.viewModel.centre_hz
        restoreMode: Binding.RestoreNone
    }

    Field {
        label: qsTr("Width")
        visible: listenRow.viewModel.enabled

        SpinBox {
            id: widthBox
            width: 120
            // The floor comes from the filter, not from taste: below it the
            // band would be nothing but roll-off.
            from: listenRow.viewModel.min_width_hz
            to: listenRow.viewModel.max_freq_hz
            // Coarse steps up where the bands are wide, fine ones down at
            // the narrow end where the whole usable range is a few hundred Hz
            stepSize: value <= 1000 ? 10 : (value <= 20000 ? 100 : 1000)
            editable: true
            textFromValue: function (value, locale) { return value + " Hz" }
            valueFromText: function (text, locale) { return Number.fromLocaleString(locale, text.replace(" Hz", "").trim()) }
            onValueModified: listenRow.viewModel.width_hz = value
        }

        // A narrow band needs a long filter, and a long filter delays what you
        // hear. Shown only once it is enough to notice, since that is exactly
        // when someone would otherwise wonder what is wrong.
        Text {
            id: latencyText
            text: "+" + listenRow.viewModel.latency_ms + qsTr(" ms")
            visible: listenRow.viewModel.latency_ms >= 10
            color: systemPalette.mid
            anchors.verticalCenter: parent.verticalCenter

            HoverHandler { id: latencyHover }
            ToolTip.visible: latencyHover.hovered
            ToolTip.text: qsTr("Delay added by the filter. Narrow bands need long filters: separating frequencies this close means listening to that much signal first.")
        }
    }

    Binding {
        target: widthBox
        property: "value"
        value: listenRow.viewModel.width_hz
        restoreMode: Binding.RestoreNone
    }

    Field {
        label: qsTr("Gain")
        visible: listenRow.viewModel.enabled

        CheckBox {
            id: agcBox
            text: qsTr("Auto")
            anchors.verticalCenter: parent.verticalCenter
            checked: listenRow.viewModel.agc_enabled
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Bring quiet bands up to a usable level automatically. Gain below stays a trim on top. Note that this lifts the noise floor with the signal - it helps a quiet sound, not one buried under noise.")
            onToggled: listenRow.viewModel.agc_enabled = checked
        }

        // What the AGC is doing, so it is not a black box: a reading pinned at
        // the ceiling means the band is quieter than the AGC can fix.
        Text {
            text: (listenRow.viewModel.agc_gain_db >= 0 ? "+" : "") + listenRow.viewModel.agc_gain_db + qsTr(" dB")
            visible: listenRow.viewModel.agc_enabled
            color: systemPalette.mid
            anchors.verticalCenter: parent.verticalCenter
        }

        Slider {
            id: gainSlider
            width: 120
            anchors.verticalCenter: parent.verticalCenter
            from: -20
            // Far past the point of usefulness on purpose. A 40-50 kHz band
            // in a quiet room measures about -73 dBFS, so 60 dB is where it
            // sits comfortably and 75 dB is already clipping half its
            // samples -- but where to stop is the listener's call, and the
            // clip warning beside this says when it has been passed.
            to: 150
            stepSize: 1
            value: listenRow.viewModel.gain_db
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Drag for coarse, arrow keys for 1 dB. Past about +60 dB on a quiet band the peaks flatten and more gain buys distortion rather than volume.")
            onMoved: listenRow.viewModel.gain_db = value
        }

        Text {
            text: (listenRow.viewModel.gain_db > 0 ? "+" : "") + listenRow.viewModel.gain_db + qsTr(" dB")
            color: systemPalette.text
            anchors.verticalCenter: parent.verticalCenter
        }

        CheckBox {
            id: limiterBox
            text: qsTr("Limit")
            anchors.verticalCenter: parent.verticalCenter
            checked: listenRow.viewModel.limiter_enabled
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Hold the output inside full scale by turning the gain down for the moment a peak lasts, instead of flattening the peak. Looks 2 ms ahead so a call's attack is already covered when it arrives. Below the ceiling it does nothing.")
            onToggled: listenRow.viewModel.limiter_enabled = checked
        }

        // How hard the limiter is working. With it on there is no clipping to
        // warn about, but knowing the gain is 40 dB past what fits is still
        // worth having.
        Text {
            text: listenRow.viewModel.limiter_reduction_db + qsTr(" dB")
            visible: listenRow.viewModel.limiter_enabled && listenRow.viewModel.limiter_reduction_db <= -1
            color: "#d08040"
            anchors.verticalCenter: parent.verticalCenter

            HoverHandler { id: limitHover }
            ToolTip.visible: limitHover.hovered
            ToolTip.text: qsTr("Gain reduction the limiter is applying. Deep and constant means the gain is set far higher than the signal needs.")
        }

        // Only reachable with the limiter switched off, and then it matters.
        Text {
            text: qsTr("CLIP")
            visible: listenRow.viewModel.clipping
            color: "#e05030"
            font.bold: true
            anchors.verticalCenter: parent.verticalCenter

            HoverHandler { id: clipHover }
            ToolTip.visible: clipHover.hovered
            ToolTip.text: qsTr("The output is past full scale and the peaks are being flattened. More gain now adds distortion, not volume. Switch Limit on to hold it back instead.")
        }
    }

    Binding {
        target: agcBox
        property: "checked"
        value: listenRow.viewModel.agc_enabled
        restoreMode: Binding.RestoreNone
    }

    Binding {
        target: gainSlider
        property: "value"
        value: listenRow.viewModel.gain_db
        restoreMode: Binding.RestoreNone
    }

    Binding {
        target: limiterBox
        property: "checked"
        value: listenRow.viewModel.limiter_enabled
        restoreMode: Binding.RestoreNone
    }

    Field {
        label: qsTr("Noise")
        visible: listenRow.viewModel.enabled

        CheckBox {
            id: denoiseBox
            text: qsTr("Reduce")
            anchors.verticalCenter: parent.verticalCenter
            checked: listenRow.viewModel.denoise_enabled
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Subtract whatever is always there. Steady interference - this room has a 25 kHz pest repeller - is learned and removed, and the hiss comes down with it. Costs 6 ms of delay, and a very faint call can be taken for background.")
            onToggled: listenRow.viewModel.denoise_enabled = checked
        }

        CheckBox {
            id: gateBox
            text: qsTr("Gate")
            anchors.verticalCenter: parent.verticalCenter
            checked: listenRow.viewModel.gate_enabled
            ToolTip.visible: hovered
            ToolTip.text: qsTr("Silence between calls. Opens for anything well above the recent quietest level and holds briefly so a call's tail is not clipped. Does not improve what you hear during a call - it removes the hiss in between.")
            onToggled: listenRow.viewModel.gate_enabled = checked
        }
    }

    Binding {
        target: denoiseBox
        property: "checked"
        value: listenRow.viewModel.denoise_enabled
        restoreMode: Binding.RestoreNone
    }

    Binding {
        target: gateBox
        property: "checked"
        value: listenRow.viewModel.gate_enabled
        restoreMode: Binding.RestoreNone
    }

    CheckBox {
        id: liveBox
        text: qsTr("Live")
        visible: listenRow.viewModel.enabled
        checked: listenRow.viewModel.monitoring
        ToolTip.visible: hovered
        ToolTip.text: qsTr("Play the band through the speakers as it comes in. Use headphones: this feeds the microphone back out.")
        onToggled: listenRow.viewModel.monitoring = checked
    }

    Binding {
        target: liveBox
        property: "checked"
        value: listenRow.viewModel.monitoring
        restoreMode: Binding.RestoreNone
    }

    // Band-pass above the playback rate selects something the sound card
    // then discards. Silence with no explanation reads as "nothing is there".
    Text {
        text: listenRow.viewModel.playback_note
        visible: listenRow.viewModel.enabled && text !== ""
        color: "#d08040"
        height: enableBox.height
        verticalAlignment: Text.AlignVCenter
    }

    Text {
        text: listenRow.viewModel.status_text
        // In band-pass mode this only restates the two spin boxes. It earns
        // its width in heterodyne, where it is the only thing that says where
        // the band comes back out.
        visible: listenRow.viewModel.enabled && listenRow.viewModel.shifts_frequency
        color: systemPalette.text
        height: enableBox.height
        verticalAlignment: Text.AlignVCenter
        // never widen the bar: take what is left on the line and elide
        width: Math.max(0, Math.min(implicitWidth, listenRow.width - x))
        elide: Text.ElideRight
    }
}
