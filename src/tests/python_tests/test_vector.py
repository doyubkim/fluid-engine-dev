"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
from pytest import approx


def test_vector2f_Init():
    a = pyjet.Vector2F()
    assert a.x == 0.0
    assert a.y == 0.0
    b = pyjet.Vector2F(1, 2)
    assert b.x == 1.0
    assert b.y == 2.0
    c = pyjet.Vector2F(y=2, x=1)
    assert c.x == 1.0
    assert c.y == 2.0


def test_vector2f_getters():
    a = pyjet.Vector2F(1, 2)
    assert a[0] == 1
    assert a[1] == 2


def test_vector2f_setters():
    a = pyjet.Vector2F(1, 2)
    a[0] = 4
    a[1] = 5
    assert a[0] == 4
    assert a[1] == 5


def test_vector2f_calc():
    a = pyjet.Vector2F(1, 2)
    b = pyjet.Vector2F(4, 6)
    c = a + b
    assert c.x == 5.0
    assert c.y == 8.0
    c = a - b
    assert c.x == -3.0
    assert c.y == -4.0
    c = a * b
    assert c.x == 4.0
    assert c.y == 12.0
    c = a / b
    assert c.x == approx(1.0 / 4.0)
    assert c.y == approx(1.0 / 3.0)


def test_vector2d_init():
    a = pyjet.Vector2D()
    assert a.x == 0.0
    assert a.y == 0.0
    b = pyjet.Vector2D(1, 2)
    assert b.x == 1.0
    assert b.y == 2.0
    c = pyjet.Vector2D(y=2, x=1)
    assert c.x == 1.0
    assert c.y == 2.0


def test_vector2d_getters():
    a = pyjet.Vector2D(1, 2)
    assert a[0] == 1
    assert a[1] == 2


def test_vector2d_setters():
    a = pyjet.Vector2D(1, 2)
    a[0] = 4
    a[1] = 5
    assert a[0] == 4
    assert a[1] == 5


def test_vector2d_calc():
    a = pyjet.Vector2D(1, 2)
    b = pyjet.Vector2D(4, 6)
    c = a + b
    assert c.x == 5.0
    assert c.y == 8.0
    c = a - b
    assert c.x == -3.0
    assert c.y == -4.0
    c = a * b
    assert c.x == 4.0
    assert c.y == 12.0
    c = a / b
    assert c.x == approx(1.0 / 4.0)
    assert c.y == approx(1.0 / 3.0)


def test_vector3f_init():
    a = pyjet.Vector3F()
    assert a.x == 0.0
    assert a.y == 0.0
    assert a.z == 0.0
    b = pyjet.Vector3F(1, 2, 3)
    assert b.x == 1.0
    assert b.y == 2.0
    assert b.z == 3.0
    c = pyjet.Vector3F(y=2, x=1, z=3)
    assert c.x == 1.0
    assert c.y == 2.0
    assert c.z == 3.0


def test_vector3f_getters():
    a = pyjet.Vector3F(1, 2, 3)
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3


def test_vector3f_setters():
    a = pyjet.Vector3F(1, 2, 3)
    a[0] = 4
    a[1] = 5
    a[2] = 6
    assert a[0] == 4
    assert a[1] == 5
    assert a[2] == 6


def test_vector3f_calc():
    a = pyjet.Vector3F(1, 2, 3)
    b = pyjet.Vector3F(4, 6, 8)
    c = a + b
    assert c.x == 5.0
    assert c.y == 8.0
    assert c.z == 11.0
    c = a - b
    assert c.x == -3.0
    assert c.y == -4.0
    assert c.z == -5.0
    c = a * b
    assert c.x == 4.0
    assert c.y == 12.0
    assert c.z == 24.0
    c = a / b
    assert c.x == approx(1.0 / 4.0)
    assert c.y == approx(1.0 / 3.0)
    assert c.z == approx(3.0 / 8.0)


def test_vector3d_Init():
    a = pyjet.Vector3D()
    assert a.x == 0.0
    assert a.y == 0.0
    assert a.z == 0.0
    b = pyjet.Vector3D(1, 2, 3)
    assert b.x == 1.0
    assert b.y == 2.0
    assert b.z == 3.0
    c = pyjet.Vector3D(y=2, x=1, z=3)
    assert c.x == 1.0
    assert c.y == 2.0
    assert c.z == 3.0


def test_vector3d_getters():
    a = pyjet.Vector3D(1, 2, 3)
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3


def test_vector3d_setters():
    a = pyjet.Vector3D(1, 2, 3)
    a[0] = 4
    a[1] = 5
    a[2] = 6
    assert a[0] == 4
    assert a[1] == 5
    assert a[2] == 6


def test_vector3d_calc():
    a = pyjet.Vector3D(1, 2, 3)
    b = pyjet.Vector3D(4, 6, 8)
    c = a + b
    assert c.x == 5.0
    assert c.y == 8.0
    assert c.z == 11.0
    c = a - b
    assert c.x == -3.0
    assert c.y == -4.0
    assert c.z == -5.0
    c = a * b
    assert c.x == 4.0
    assert c.y == 12.0
    assert c.z == 24.0
    c = a / b
    assert c.x == approx(1.0 / 4.0)
    assert c.y == approx(1.0 / 3.0)
    assert c.z == approx(3.0 / 8.0)
