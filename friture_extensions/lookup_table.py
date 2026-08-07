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

"""Pure-numpy stand-in for the compiled lookup_table.pyx.

Python imports a compiled extension in preference to a .py file of the same
name, so this is used only until friture_extensions is built.

A fancy-index into the palette does the whole lookup in one go, so nothing
is lost against the compiled version here.
"""

import numpy as np

dtype = np.float64


def _palette_index(values, size):
    """Value in [0, 1] -> palette row, truncating toward zero as C does.

    The compiled version runs with bounds checking off, so an out-of-range
    value there reads whatever is next in memory. Callers clip first, but
    clamping rather than trusting them costs nothing and turns a would-be
    IndexError into the same colour the compiled build would most likely
    have shown.
    """
    return np.clip((values * 255).astype(np.intp), 0, size - 1)


def pyx_color_from_float(lut, values):
    """1-D of values in [0, 1] -> 1-D of packed uint32 colours."""
    return lut[_palette_index(values, lut.shape[0])]


def pyx_color_from_float_2D(lut, values):
    """2-D of values in [0, 1] -> 2-D of packed uint32 colours."""
    return lut[_palette_index(values, lut.shape[0])]


def pyx_rgb_from_float_2D(lut, values):
    """2-D of values in [0, 1] -> 3-D (rows, columns, 3) of uint32 channels."""
    return lut[_palette_index(values, lut.shape[0])]
