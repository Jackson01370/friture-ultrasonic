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

"""Pure-numpy stand-in for the compiled exp_smoothing_conv.pyx.

Python imports a compiled extension in preference to a .py file of the same
name, so this is used only until friture_extensions is built.

Both functions are the unrolled form of y_i = alpha*x_i + (1-alpha)*y_{i-1}:
the recurrence is folded into a precomputed kernel so a whole block can be
reduced in one dot product. That makes them a straight numpy translation
with no loop and no speed penalty -- unlike lfilter, nothing is lost here.
"""

import numpy as np

dtype = np.float64


def _kernel_span(kernel_length, data_length, alpha):
    """How much of the block the kernel can actually cover, and the decay.

    The kernel is finite, so a block longer than the kernel is truncated to
    it. When that happens the previous value's weight is dropped to zero:
    the kernel is long enough that (1-alpha)^length has already decayed to
    nothing, so there is no history left to carry.
    """
    if data_length > kernel_length:
        return kernel_length, 0.0
    return data_length, (1.0 - alpha) ** data_length


def pyx_exp_smoothed_value(kernel, alpha, data, previous):
    """Exponentially smoothed value of a 1-D block. Returns a scalar."""
    kernel_length = kernel.shape[0]
    span, decay = _kernel_span(kernel_length, data.shape[0], alpha)

    # The kernel is right-aligned on the block: its last tap weighs the
    # newest sample.
    conv = float(np.dot(kernel[kernel_length - span:kernel_length], data[:span]))
    return alpha * conv + previous * decay


def pyx_exp_smoothed_value_numpy(kernel, alpha, data, previous):
    """Same, for a 2-D block filtered along axis 1, one row at a time.

    Returns a 1-D array with one value per row.
    """
    kernel_length = kernel.shape[0]
    span, decay = _kernel_span(kernel_length, data.shape[1], alpha)

    if span == 0:
        return previous

    conv = data[:, :span] @ kernel[kernel_length - span:kernel_length]
    return alpha * conv + previous * decay
