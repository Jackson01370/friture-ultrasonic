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

"""Band listening: click a peak on a plot, hear only that band.

The pieces:
  band_dsp                the two filters (band-pass and heterodyne)
  audio_fifo              GUI thread -> output callback hand-off
  listen_band_view_model  the one band everything shares, and the QML surface
  processor               band -> filter, with a thread-safe retune
  monitor                 the live output stream
"""
