"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
from pytest import approx


def assert_vector_similar(a, b):
    assert tuple(a) == approx(tuple(b))


def assert_bounding_box_similar(a, b):
    assert_vector_similar(a.lowerCorner, b.lowerCorner)
    assert_vector_similar(a.upperCorner, b.upperCorner)
