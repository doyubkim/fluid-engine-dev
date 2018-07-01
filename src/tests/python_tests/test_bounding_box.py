"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet


def test_bounding_box2f_init():
    a = pyjet.BoundingBox2D()
    assert a.lowerCorner.x > a.upperCorner.x
    assert a.lowerCorner.y > a.upperCorner.y

    b = pyjet.BoundingBox2D((-1, -2), (4, 2))
    b.lowerCorner.x == -1.0
    b.lowerCorner.y == -2.0
    b.upperCorner.x == 4.0
    b.upperCorner.y == 2.0

    l, c = pyjet.Vector2D(-1, -2), pyjet.Vector2D(4, 2)
    c = pyjet.BoundingBox2D(l, c)
    c.lowerCorner.x == -1.0
    c.lowerCorner.y == -2.0
    c.upperCorner.x == 4.0
    c.upperCorner.y == 2.0


def test_bounding_box2f_is_empty():
    a = pyjet.BoundingBox2D((-2.0, -2.0), (4.0, 3.0))
    assert not a.isEmpty()


# ------------------------------------------------------------------------------


def test_bounding_box2d_init():
    a = pyjet.BoundingBox2D()
    assert a.lowerCorner.x > a.upperCorner.x
    assert a.lowerCorner.y > a.upperCorner.y

    b = pyjet.BoundingBox2D((-1, -2), (4, 2))
    b.lowerCorner.x == -1.0
    b.lowerCorner.y == -2.0
    b.upperCorner.x == 4.0
    b.upperCorner.y == 2.0

    l, c = pyjet.Vector2D(-1, -2), pyjet.Vector2D(4, 2)
    c = pyjet.BoundingBox2D(l, c)
    c.lowerCorner.x == -1.0
    c.lowerCorner.y == -2.0
    c.upperCorner.x == 4.0
    c.upperCorner.y == 2.0


def test_bounding_box2d_is_empty():
    a = pyjet.BoundingBox2D((-2.0, -2.0), (4.0, 3.0))
    assert not a.isEmpty()


# ------------------------------------------------------------------------------


def test_bounding_box3f_init():
    a = pyjet.BoundingBox3F()
    assert a.lowerCorner.x > a.upperCorner.x
    assert a.lowerCorner.y > a.upperCorner.y
    assert a.lowerCorner.z > a.upperCorner.z

    b = pyjet.BoundingBox3F((-1, -2, -3), (4, 2, 5))
    assert b.lowerCorner.x == -1.0
    assert b.lowerCorner.y == -2.0
    assert b.lowerCorner.z == -3.0
    assert b.upperCorner.x == 4.0
    assert b.upperCorner.y == 2.0
    assert b.upperCorner.z == 5.0

    l, c = pyjet.Vector3F(-1, -2, -3), pyjet.Vector3F(4, 2, 5)
    c = pyjet.BoundingBox3F(l, c)
    assert c.lowerCorner.x == -1.0
    assert c.lowerCorner.y == -2.0
    assert c.lowerCorner.z == -3.0
    assert c.upperCorner.x == 4.0
    assert c.upperCorner.y == 2.0
    assert c.upperCorner.z == 5.0


def test_bounding_box3f_is_empty():
    a = pyjet.BoundingBox3F((-2.0, -2.0, 1.0), (4.0, 3.0, 5.0))
    assert not a.isEmpty()


# ------------------------------------------------------------------------------


def test_bounding_box3d_init():
    a = pyjet.BoundingBox3D()
    assert a.lowerCorner.x > a.upperCorner.x
    assert a.lowerCorner.y > a.upperCorner.y
    assert a.lowerCorner.z > a.upperCorner.z

    b = pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 5))
    assert b.lowerCorner.x == -1.0
    assert b.lowerCorner.y == -2.0
    assert b.lowerCorner.z == -3.0
    assert b.upperCorner.x == 4.0
    assert b.upperCorner.y == 2.0
    assert b.upperCorner.z == 5.0

    l, c = pyjet.Vector3D(-1, -2, -3), pyjet.Vector3D(4, 2, 5)
    c = pyjet.BoundingBox3D(l, c)
    assert c.lowerCorner.x == -1.0
    assert c.lowerCorner.y == -2.0
    assert c.lowerCorner.z == -3.0
    assert c.upperCorner.x == 4.0
    assert c.upperCorner.y == 2.0
    assert c.upperCorner.z == 5.0


def test_bounding_box3d_is_empty():
    a = pyjet.BoundingBox3D((-2.0, -2.0, 1.0), (4.0, 3.0, 5.0))
    assert not a.isEmpty()
