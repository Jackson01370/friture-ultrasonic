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

"""Pure-Python stand-in for the compiled lfilter.pyx.

Python imports a compiled extension in preference to a .py file of the same
name, so building friture_extensions makes this file dead automatically --
there is nothing to switch off, and no import site to change. It exists so
Friture starts on a machine with no C compiler.

Speed matters here: this filter is on the live audio path (octave bands,
long levels, decimation), several instances per block. scipy's lfilter is
the same direct-form-II-transposed algorithm in C, so it is used when it is
importable. The Python loop below is the correctness backstop, and it is
too slow to keep up with live audio -- expect the octave spectrum and level
widgets to lag if it is what ends up running.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.signal import lfilter as _scipy_lfilter
except ImportError:  # pragma: no cover - depends on the machine, not the code
    _scipy_lfilter = None
    logger.warning(
        "friture_extensions is not compiled and scipy is not available: "
        "filtering falls back to pure Python and will not keep up with live audio")
else:
    logger.info("friture_extensions.lfilter: using the scipy implementation "
                "(the compiled extension is not built)")


def pyx_lfilter_float64_1D(b, a, x, zi):
    """Filter x with the IIR/FIR filter (b, a), carrying the state zi.

    Direct form II transposed, matching the compiled extension exactly:

        y[k]      = z[0] + b[0]*x[k]
        z[n]      = z[n+1] + b[n+1]*x[k] - a[n+1]*y[k]
        z[len-2]  = b[len-1]*x[k] - a[len-1]*y[k]

    Returns (y, zf) -- the output and the final state.
    """
    assert b.shape[0] == a.shape[0], "a and b must be of the same shape"
    assert zi.shape[0] == b.shape[0] - 1

    if _scipy_lfilter is not None:
        y, zf = _scipy_lfilter(b, a, x, zi=zi)
        return y, zf

    len_b = b.shape[0]
    if len_b <= 1:
        return x * b[0], np.array(zi, copy=True)

    # Plain Python lists and floats: for the filter orders Friture uses (a
    # handful of taps) the interpreter overhead of numpy scalars costs more
    # than it saves.
    bl = b.tolist()
    al = a.tolist()
    z = zi.tolist()
    y = np.empty(x.shape[0], dtype=np.float64)
    yl = [0.0] * x.shape[0]

    for k, xk in enumerate(x.tolist()):
        yk = z[0] + bl[0] * xk
        yl[k] = yk
        for n in range(len_b - 2):
            z[n] = z[1 + n] + xk * bl[1 + n] - yk * al[1 + n]
        z[len_b - 2] = xk * bl[len_b - 1] - yk * al[len_b - 1]

    y[:] = yl
    return y, np.array(z, dtype=np.float64)
