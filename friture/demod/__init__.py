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

"""Digital demodulation of the listen band: from a keyed carrier to bits.

Ported from ultraScan's M22-M26 work (dsp/demod.py, demod_digital.py,
burst.py, symbols.py and the live wiring in demod_audio.py), reduced to what
the bit decoder needs and rewritten on numpy alone, like friture.listen.

The pieces, in signal order:
  ddc         real 250 kHz capture -> complex baseband of the selected band
  detectors   what frequency / what amplitude / which tone / which phase side,
              per sample
  burst       the envelope's ON/OFF state machine (OOK)
  symbols     symbol clock recovery and slicing: from a per-sample trace to
              symbols and bits, without being told the baud rate
  decoder     the live chain, one block at a time, with the readouts the
              Digital Decode dock shows

The whole point of the design, carried over from ultraScan: a decoder that
shows bits without showing how much to believe them is a random bit
generator with a nice font. Every number here comes with its confidence, and
bits are published only behind that gate.
"""
