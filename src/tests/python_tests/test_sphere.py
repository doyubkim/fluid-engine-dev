"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet


def test_init():
    a = pyjet.Sphere3()
    assert a.isNormalFlipped == False
    assert a.center == (0, 0, 0)
    assert a.radius == 1.0
