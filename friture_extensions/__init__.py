#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Timothée Lecomte

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

"""Friture's compiled hot paths, with a Python stand-in for each.

Every .pyx here has a .py of the same name beside it. Python's import
machinery tries extension modules before source files, so the compiled
build wins automatically wherever it exists and the .py only runs on a
machine that has not built (or cannot build) the extensions.

That means `python setup.py build_ext --inplace` needs no other change to
take effect, and deleting the .pyd files is enough to go back. The one
thing to know is that the lfilter stand-in is materially slower unless
scipy is installed -- see friture_extensions/lfilter.py.
"""
