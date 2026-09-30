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

"""Continuous recording of everything the microphone hears.

Not a band, not a spectrum: the raw capture, every channel, at the capture
rate, as 16-bit samples. That is the only form from which ANY band can be
listened to, any spectrum drawn and any demodulation run again later -- a
spectrum alone has thrown the phase away and cannot be heard.

    wav_segment   one WAV file being written, and repairing one a crash left
    store         the folder of segments: listing, sizes, rotation
    recorder      the thread that takes blocks from the capture and writes them
"""
