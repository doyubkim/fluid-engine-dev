"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import math
import pyjet
from pytest import approx


def test_init():
    a = pyjet.QuaternionD()
    assert a.w == 1.0
    assert a.x == 0.0
    assert a.y == 0.0
    assert a.z == 0.0
    b = pyjet.QuaternionD(-1, 2, 3, 4)
    assert b.w == -1.0
    assert b.x == 2.0
    assert b.y == 3.0
    assert b.z == 4.0
    c = pyjet.QuaternionD(x=2, z=4, w=1, y=3)
    assert c.w == 1.0
    assert c.x == 2.0
    assert c.y == 3.0
    assert c.z == 4.0


def test_getters():
    a = pyjet.QuaternionD(1, 2, 3, 4)
    assert a[0] == a.w
    assert a[1] == a.x
    assert a[2] == a.y
    assert a[3] == a.z
    a.normalize()
    axis = a.axis()
    angle = a.angle()
    denom = math.sqrt(1 - a.w * a.w)
    assert axis.x == approx(math.sqrt(2.0 / 15.0) / denom)
    assert axis.y == approx(math.sqrt(3.0 / 10.0) / denom)
    assert axis.z == approx(2.0 * math.sqrt(2.0 / 15.0) / denom)
    assert angle == approx(2.0 * math.acos(1.0 / math.sqrt(30.0)))


def test_setters():
    a = pyjet.QuaternionD()
    a.setAxisAngle((0, -1, 0), math.pi / 2)
    axis = a.axis()
    angle = a.angle()
    assert axis.x == 0
    assert axis.y == -1
    assert axis.z == 0
    assert angle == math.pi / 2
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    a[3] = 4.0
    assert 1 == a.w
    assert 2 == a.x
    assert 3 == a.y
    assert 4 == a.z
