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

"""The Python stand-ins must agree with the .pyx they replace.

Whichever build is installed is what gets tested: on a machine with the
extensions compiled these check the compiled code, on one without them they
check the fallbacks. The oracle in each test is a literal transcription of
the loop in the corresponding .pyx file, so the two can only agree by
actually computing the same thing.
"""

import unittest

import numpy as np

from friture_extensions.exp_smoothing_conv import (
    pyx_exp_smoothed_value,
    pyx_exp_smoothed_value_numpy,
)
from friture_extensions.lfilter import pyx_lfilter_float64_1D
from friture_extensions.linear_interp import pyx_linear_interp_2D
from friture_extensions.lookup_table import (
    pyx_color_from_float,
    pyx_color_from_float_2D,
    pyx_rgb_from_float_2D,
)


# -- oracles: the .pyx loops, written out in Python ----------------------

def reference_lfilter(b, a, x, zi):
    len_b = b.shape[0]
    y = np.empty(x.shape[0])
    z = np.array(zi, copy=True)
    if len_b > 1:
        for k in range(x.shape[0]):
            y[k] = z[0] + b[0] * x[k]
            for n in range(len_b - 2):
                z[n] = z[1 + n] + x[k] * b[1 + n] - y[k] * a[1 + n]
            z[len_b - 2] = x[k] * b[len_b - 1] - y[k] * a[len_b - 1]
    else:
        for k in range(x.shape[0]):
            y[k] = x[k] * b[0]
    return y, z


def reference_exp_smoothed_value(kernel, alpha, data, previous):
    N = data.shape[0]
    Nk = kernel.shape[0]
    a = (1. - alpha) ** N
    if N > Nk:
        N = Nk
        a = 0.
    conv = 0.
    for i in range(0, N):
        conv = conv + kernel[Nk - N + i] * data[i]
    return alpha * conv + previous * a


def reference_exp_smoothed_value_numpy(kernel, alpha, data, previous):
    Nf = data.shape[0]
    Nt = data.shape[1]
    Nk = kernel.shape[0]
    a = (1. - alpha) ** Nt
    if Nt > Nk:
        Nt = Nk
        a = 0.
    if Nt == 0:
        return previous
    conv = np.zeros(Nf)
    value = np.zeros(Nf)
    for i in range(0, Nt):
        for j in range(Nf):
            conv[j] = conv[j] + kernel[Nk - Nt + i] * data[j, i]
    for j in range(Nf):
        value[j] = alpha * conv[j] + previous[j] * a
    return value


def reference_linear_interp_2D(resampled_buffer, data, old_data, orig_index,
                               resampled_index, resampling_ratio, n):
    N = data.shape[0]
    for i in range(n):
        resampled_index += resampling_ratio
        a = orig_index - resampled_index
        for j in range(N):
            resampled_buffer[j, i] = (1 - a) * data[j] + a * old_data[j]
    return resampled_index


def reference_color_from_float_2D(lut, values):
    M, N = values.shape
    out = np.zeros([M, N], dtype=np.uint32)
    for i in range(M):
        for j in range(N):
            out[i, j] = lut[int(values[i, j] * 255)]
    return out


# -- tests ----------------------------------------------------------------

class TestLfilter(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(20260804)

    def test_matches_the_reference_for_an_iir_filter(self):
        # a 4th-order filter shaped like the octave-band ones Friture uses
        b = np.array([0.2, -0.1, 0.05, 0.01, 0.0])
        a = np.array([1.0, -0.5, 0.3, -0.1, 0.02])
        x = self.rng.standard_normal(500)
        zi = np.zeros(b.shape[0] - 1)

        y, zf = pyx_lfilter_float64_1D(b, a, x, zi)
        ry, rzf = reference_lfilter(b, a, x, zi)

        np.testing.assert_allclose(y, ry, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(zf, rzf, rtol=1e-12, atol=1e-12)

    def test_state_carries_across_blocks(self):
        b = np.array([0.2, -0.1, 0.05, 0.01, 0.0])
        a = np.array([1.0, -0.5, 0.3, -0.1, 0.02])
        x = self.rng.standard_normal(600)

        whole, _ = pyx_lfilter_float64_1D(b, a, x, np.zeros(4))

        blocked = []
        zi = np.zeros(4)
        for i in range(0, 600, 128):
            y, zi = pyx_lfilter_float64_1D(b, a, x[i:i + 128], zi)
            blocked.append(y)

        np.testing.assert_allclose(np.concatenate(blocked), whole, rtol=1e-12, atol=1e-12)

    def test_a_non_zero_initial_state_is_used(self):
        b = np.array([1.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])
        x = np.zeros(4)
        zi = np.array([7.0, 3.0])

        y, zf = pyx_lfilter_float64_1D(b, a, x, zi)
        ry, rzf = reference_lfilter(b, a, x, zi)

        np.testing.assert_allclose(y, ry)
        np.testing.assert_allclose(zf, rzf)

    def test_single_tap_filter_is_a_plain_scaling(self):
        b = np.array([2.0])
        a = np.array([1.0])
        x = self.rng.standard_normal(10)
        y, _ = pyx_lfilter_float64_1D(b, a, x, np.zeros(0))
        np.testing.assert_allclose(y, 2.0 * x)


class TestLfilterWithoutScipy(unittest.TestCase):
    """The last-resort path, for a machine with neither the extension nor scipy.

    Nothing here can reach it by accident, so it is forced: without this the
    Python loop would ship untested and only ever run somewhere we cannot
    look.
    """

    def test_the_python_loop_matches_the_reference(self):
        import friture_extensions.lfilter as module

        if not hasattr(module, '_scipy_lfilter'):
            self.skipTest("the compiled extension is installed, so there is no Python path")

        b = np.array([0.2, -0.1, 0.05, 0.01, 0.0])
        a = np.array([1.0, -0.5, 0.3, -0.1, 0.02])
        x = np.random.default_rng(20260804).standard_normal(200)
        zi = np.array([0.3, -0.2, 0.1, 0.05])

        original = module._scipy_lfilter
        module._scipy_lfilter = None
        try:
            y, zf = module.pyx_lfilter_float64_1D(b, a, x, zi)
        finally:
            module._scipy_lfilter = original

        ry, rzf = reference_lfilter(b, a, x, zi)
        np.testing.assert_allclose(y, ry, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(zf, rzf, rtol=1e-12, atol=1e-12)


class TestExpSmoothing(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(20260804)
        self.alpha = 0.02
        self.kernel = (1. - self.alpha) ** np.arange(512)[::-1]

    def test_1d_matches_the_reference(self):
        data = self.rng.standard_normal(300)
        self.assertAlmostEqual(
            pyx_exp_smoothed_value(self.kernel, self.alpha, data, 0.75),
            reference_exp_smoothed_value(self.kernel, self.alpha, data, 0.75),
            places=12)

    def test_1d_block_longer_than_the_kernel_drops_the_history(self):
        data = self.rng.standard_normal(700)  # kernel is 512
        self.assertAlmostEqual(
            pyx_exp_smoothed_value(self.kernel, self.alpha, data, 0.75),
            reference_exp_smoothed_value(self.kernel, self.alpha, data, 0.75),
            places=12)

    def test_1d_empty_block_returns_the_previous_value(self):
        self.assertAlmostEqual(
            pyx_exp_smoothed_value(self.kernel, self.alpha, np.zeros(0), 0.75),
            0.75, places=12)

    def test_2d_matches_the_reference(self):
        data = self.rng.standard_normal((17, 300))
        previous = self.rng.standard_normal(17)
        np.testing.assert_allclose(
            pyx_exp_smoothed_value_numpy(self.kernel, self.alpha, data, previous),
            reference_exp_smoothed_value_numpy(self.kernel, self.alpha, data, previous),
            rtol=1e-12, atol=1e-12)

    def test_2d_block_longer_than_the_kernel_drops_the_history(self):
        data = self.rng.standard_normal((17, 700))
        previous = self.rng.standard_normal(17)
        np.testing.assert_allclose(
            pyx_exp_smoothed_value_numpy(self.kernel, self.alpha, data, previous),
            reference_exp_smoothed_value_numpy(self.kernel, self.alpha, data, previous),
            rtol=1e-12, atol=1e-12)

    def test_2d_empty_block_returns_the_previous_value(self):
        previous = self.rng.standard_normal(17)
        np.testing.assert_allclose(
            pyx_exp_smoothed_value_numpy(self.kernel, self.alpha, np.zeros((17, 0)), previous),
            previous)


class TestLinearInterp(unittest.TestCase):

    def test_matches_the_reference_including_the_returned_index(self):
        rng = np.random.default_rng(20260804)
        data = rng.standard_normal(64)
        old_data = rng.standard_normal(64)

        mine = np.zeros((64, 10))
        theirs = np.zeros((64, 10))

        index_mine = pyx_linear_interp_2D(mine, data, old_data, 3.25, 0.5, 0.3, 10)
        index_theirs = reference_linear_interp_2D(theirs, data, old_data, 3.25, 0.5, 0.3, 10)

        np.testing.assert_allclose(mine, theirs, rtol=1e-12, atol=1e-12)
        self.assertEqual(index_mine, index_theirs)

    def test_zero_columns_leaves_the_buffer_and_index_alone(self):
        buffer = np.ones((4, 3))
        index = pyx_linear_interp_2D(buffer, np.zeros(4), np.zeros(4), 1.0, 0.5, 0.3, 0)
        self.assertEqual(index, 0.5)
        np.testing.assert_array_equal(buffer, np.ones((4, 3)))


class TestLookupTable(unittest.TestCase):

    def setUp(self):
        self.lut = np.arange(256, dtype=np.uint32) * 7
        self.rgb_lut = np.arange(256 * 3, dtype=np.uint32).reshape(256, 3)

    def test_2d_matches_the_reference(self):
        values = np.random.default_rng(20260804).random((5, 9))
        np.testing.assert_array_equal(
            pyx_color_from_float_2D(self.lut, values),
            reference_color_from_float_2D(self.lut, values))

    def test_the_ends_of_the_range_land_on_the_end_colours(self):
        values = np.array([[0.0, 1.0]])
        out = pyx_color_from_float_2D(self.lut, values)
        self.assertEqual(out[0, 0], self.lut[0])
        self.assertEqual(out[0, 1], self.lut[255])

    def test_the_result_is_uint32(self):
        out = pyx_color_from_float_2D(self.lut, np.zeros((2, 2)))
        self.assertEqual(out.dtype, np.uint32)

    def test_1d_lookup(self):
        values = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_equal(
            pyx_color_from_float(self.lut, values),
            np.array([self.lut[0], self.lut[127], self.lut[255]]))

    def test_rgb_lookup_gives_three_channels(self):
        out = pyx_rgb_from_float_2D(self.rgb_lut, np.array([[0.0, 1.0]]))
        self.assertEqual(out.shape, (1, 2, 3))
        np.testing.assert_array_equal(out[0, 0], self.rgb_lut[0])
        np.testing.assert_array_equal(out[0, 1], self.rgb_lut[255])


class TestColorTransformIntegration(unittest.TestCase):
    """The spectrogram's colour stage must work end to end on the palette."""

    def test_push_returns_a_colour_for_every_cell(self):
        from friture.signal.color_tranform import Color_Transform

        transform = Color_Transform()
        data = np.random.default_rng(20260804).random((32, 4)) * 1.5  # deliberately over 1
        out = transform.push(data)

        self.assertEqual(out.shape, (32, 4))
        self.assertEqual(out.dtype, np.uint32)


if __name__ == '__main__':
    unittest.main()
