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

"""Pure-numpy stand-in for the compiled linear_interp.pyx.

Python imports a compiled extension in preference to a .py file of the same
name, so this is used only until friture_extensions is built.
"""

import numpy as np

dtype = np.float64


def pyx_linear_interp_2D(resampled_buffer, data, old_data, orig_index,
                         resampled_index, resampling_ratio, n):
    """Fill n columns by interpolating between old_data and data.

    The loop over columns is kept in Python and only the column itself is
    vectorised. Two reasons: n is the handful of new screen columns per
    frame while the column is hundreds of frequency bins, so that is where
    the work is; and resampled_index is accumulated one step at a time here
    exactly as the compiled version does, rather than jumped in one
    multiply, so the position it hands back drifts identically.
    """
    for i in range(n):
        resampled_index += resampling_ratio
        a = orig_index - resampled_index
        resampled_buffer[:, i] = (1 - a) * data + a * old_data

    return resampled_index
