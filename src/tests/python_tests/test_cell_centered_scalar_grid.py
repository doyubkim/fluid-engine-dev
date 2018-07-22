"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import numpy as np
import pyjet
from pytest import approx
from pytest_utils import *


cnt = 0


def test_grid2():
    global cnt

    a = pyjet.CellCenteredScalarGrid2(resolution=(3, 4),
                                      gridSpacing=(1, 2),
                                      gridOrigin=(7, 5))

    assert a.resolution == (3, 4)
    assert_vector_similar(a.origin, (7, 5))
    assert_vector_similar(a.gridSpacing, (1, 2))
    assert_bounding_box_similar(
        a.boundingBox, pyjet.BoundingBox2D((7, 5), (10, 13)))
    f = a.cellCenterPosition
    assert_vector_similar(f(0, 0), (7.5, 6))

    b = pyjet.CellCenteredScalarGrid2(resolution=(3, 4),
                                      gridSpacing=(1, 2),
                                      gridOrigin=(7, 5))
    assert a.hasSameShape(b)

    def func(i, j):
        global cnt
        assert i >= 0 and i < 3
        assert j >= 0 and j < 4
        cnt += 1
    cnt = 0
    a.forEachCellIndex(func)
    assert cnt == 12


def test_scalar_grid2():
    global cnt
    a = pyjet.CellCenteredScalarGrid2(resolution=(3, 4),
                                      gridSpacing=(1, 2),
                                      gridOrigin=(7, 5))

    a.resize(resolution=(12, 7),
             gridSpacing=(3, 4),
             gridOrigin=(9, 2))
    assert a.resolution == (12, 7)
    assert_vector_similar(a.origin, (9, 2))
    assert_vector_similar(a.gridSpacing, (3, 4))

    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            assert a[i, j] == 0.0

    a[5, 6] = 17.0
    assert a[5, 6] == 17.0

    a.fill(42.0)
    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            assert a[i, j] == 42.0

    def func(pt):
        return pt.x ** 2 + pt.y ** 2

    a.fill(func)
    pos = a.dataPosition()
    acc = np.array(a.dataAccessor(), copy=False)
    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            pt = pos(i, j)
            assert func(pt) == a[i, j]
            assert func(pt) == approx(a.sample(pt))
            assert acc[j, i] == a[i, j]
            # Can't compare to analytic solution because FDM with such a coarse
            # grid will return inaccurate results by design.
            assert_vector_similar(a.gradientAtDataPoint(i, j), a.gradient(pt))
            assert a.laplacianAtDataPoint(i, j) == a.laplacian(pt)

    def func(i, j):
        global cnt
        assert i >= 0 and i < a.resolution.x
        assert j >= 0 and j < a.resolution.y
        cnt += 1
    cnt = 0
    a.forEachDataPointIndex(func)
    assert cnt == a.resolution.x * a.resolution.y

    blob = a.serialize()
    b = pyjet.CellCenteredScalarGrid2()
    b.deserialize(blob)
    assert b.resolution == (12, 7)
    assert_vector_similar(b.origin, (9, 2))
    assert_vector_similar(b.gridSpacing, (3, 4))
    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            assert a[i, j] == b[i, j]


def test_cell_centered_scalar_grid2():
    # CTOR
    a = pyjet.CellCenteredScalarGrid2()
    assert a.resolution == (1, 1)
    assert_vector_similar(a.origin, (0.0, 0.0))
    assert_vector_similar(a.gridSpacing, (1.0, 1.0))

    a = pyjet.CellCenteredScalarGrid2((3, 4), (1, 2), (7, 5))
    assert a.resolution == (3, 4)
    assert_vector_similar(a.origin, (7, 5))
    assert_vector_similar(a.gridSpacing, (1, 2))

    a = pyjet.CellCenteredScalarGrid2(resolution=(3, 4),
                                      gridSpacing=(1, 2),
                                      gridOrigin=(7, 5))
    assert a.resolution == (3, 4)
    assert_vector_similar(a.origin, (7, 5))
    assert_vector_similar(a.gridSpacing, (1, 2))

    a = pyjet.CellCenteredScalarGrid2(resolution=(3, 4),
                                      domainSizeX=12.0,
                                      gridOrigin=(7, 5))
    assert a.resolution == (3, 4)
    assert_vector_similar(a.origin, (7, 5))
    assert_vector_similar(a.gridSpacing, (4, 4))

    # Properties
    a = pyjet.CellCenteredScalarGrid2(resolution=(3, 4),
                                      gridSpacing=(1, 2),
                                      gridOrigin=(7, 5))
    assert_vector_similar(a.dataSize, (3, 4))
    assert_vector_similar(a.dataOrigin, (7.5, 6))

    # Modifiers
    b = pyjet.CellCenteredScalarGrid2(resolution=(6, 3),
                                      gridSpacing=(5, 9),
                                      gridOrigin=(1, 2))
    a.fill(42.0)
    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            assert a[i, j] == 42.0

    a.swap(b)
    assert a.resolution == (6, 3)
    assert_vector_similar(a.origin, (1, 2))
    assert_vector_similar(a.gridSpacing, (5, 9))
    assert b.resolution == (3, 4)
    assert_vector_similar(b.origin, (7, 5))
    assert_vector_similar(b.gridSpacing, (1, 2))
    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            assert a[i, j] == 0.0
    for j in range(b.resolution.y):
        for i in range(b.resolution.x):
            assert b[i, j] == 42.0

    a.set(b)
    assert a.resolution == (3, 4)
    assert_vector_similar(a.origin, (7, 5))
    assert_vector_similar(a.gridSpacing, (1, 2))
    for j in range(a.resolution.y):
        for i in range(a.resolution.x):
            assert a[i, j] == 42.0

    c = a.clone()
    assert c.resolution == (3, 4)
    assert_vector_similar(c.origin, (7, 5))
    assert_vector_similar(c.gridSpacing, (1, 2))
    for j in range(c.resolution.y):
        for i in range(c.resolution.x):
            assert c[i, j] == 42.0

# ------------------------------------------------------------------------------


def test_grid3():
    global cnt

    a = pyjet.CellCenteredScalarGrid3(resolution=(3, 4, 5),
                                      gridSpacing=(1, 2, 3),
                                      gridOrigin=(7, 5, 3))

    assert a.resolution == (3, 4, 5)
    assert_vector_similar(a.origin, (7, 5, 3))
    assert_vector_similar(a.gridSpacing, (1, 2, 3))
    assert_bounding_box_similar(
        a.boundingBox, pyjet.BoundingBox3D((7, 5, 3), (10, 13, 18)))
    f = a.cellCenterPosition
    assert_vector_similar(f(0, 0, 0), (7.5, 6, 4.5))

    b = pyjet.CellCenteredScalarGrid3(resolution=(3, 4, 5),
                                      gridSpacing=(1, 2, 3),
                                      gridOrigin=(7, 5, 3))
    assert a.hasSameShape(b)

    def func(i, j, k):
        global cnt
        assert i >= 0 and i < 3
        assert j >= 0 and j < 4
        assert k >= 0 and k < 5
        cnt += 1
    cnt = 0
    a.forEachCellIndex(func)
    assert cnt == 60


def test_scalar_grid3():
    global cnt
    a = pyjet.CellCenteredScalarGrid3(resolution=(3, 4, 5),
                                      gridSpacing=(1, 2, 3),
                                      gridOrigin=(7, 5, 3))

    a.resize(resolution=(12, 7, 2),
             gridSpacing=(3, 4, 5),
             gridOrigin=(9, 2, 5))
    assert a.resolution == (12, 7, 2)
    assert_vector_similar(a.origin, (9, 2, 5))
    assert_vector_similar(a.gridSpacing, (3, 4, 5))

    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                assert a[i, j, k] == 0.0

    a[5, 6, 1] = 17.0
    assert a[5, 6, 1] == 17.0

    a.fill(42.0)
    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                assert a[i, j, k] == 42.0

    def func(pt):
        return pt.x ** 2 + pt.y ** 2 + pt.z ** 2

    a.fill(func)
    pos = a.dataPosition()
    acc = np.array(a.dataAccessor(), copy=False)
    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                pt = pos(i, j, k)
                assert func(pt) == a[i, j, k]
                assert func(pt) == approx(a.sample(pt))
                assert acc[k, j, i] == a[i, j, k]
                # Can't compare to analytic solution because FDM with such a
                # coarse grid will return inaccurate results by design.
                assert_vector_similar(
                    a.gradientAtDataPoint(i, j, k), a.gradient(pt))
                assert a.laplacianAtDataPoint(i, j, k) == a.laplacian(pt)

    def func(i, j, k):
        global cnt
        assert i >= 0 and i < a.resolution.x
        assert j >= 0 and j < a.resolution.y
        assert k >= 0 and k < a.resolution.z
        cnt += 1
    cnt = 0
    a.forEachDataPointIndex(func)
    assert cnt == a.resolution.x * a.resolution.y * a.resolution.z

    blob = a.serialize()
    b = pyjet.CellCenteredScalarGrid3()
    b.deserialize(blob)
    assert b.resolution == (12, 7, 2)
    assert_vector_similar(b.origin, (9, 2, 5))
    assert_vector_similar(b.gridSpacing, (3, 4, 5))
    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                assert a[i, j, k] == b[i, j, k]


def test_cell_centered_scalar_grid3():
    # CTOR
    a = pyjet.CellCenteredScalarGrid3()
    assert a.resolution == (1, 1, 1)
    assert_vector_similar(a.origin, (0.0, 0.0, 0.0))
    assert_vector_similar(a.gridSpacing, (1.0, 1.0, 1.0))

    a = pyjet.CellCenteredScalarGrid3((3, 4, 5), (1, 2, 3), (7, 5, 2))
    assert a.resolution == (3, 4, 5)
    assert_vector_similar(a.origin, (7, 5, 2))
    assert_vector_similar(a.gridSpacing, (1, 2, 3))

    a = pyjet.CellCenteredScalarGrid3(resolution=(3, 4, 5),
                                      gridSpacing=(1, 2, 3),
                                      gridOrigin=(7, 5, 2))
    assert a.resolution == (3, 4, 5)
    assert_vector_similar(a.origin, (7, 5, 2))
    assert_vector_similar(a.gridSpacing, (1, 2, 3))

    a = pyjet.CellCenteredScalarGrid3(resolution=(3, 4, 5),
                                      domainSizeX=12.0,
                                      gridOrigin=(7, 5, 2))
    assert a.resolution == (3, 4, 5)
    assert_vector_similar(a.origin, (7, 5, 2))
    assert_vector_similar(a.gridSpacing, (4, 4, 4))

    # Properties
    a = pyjet.CellCenteredScalarGrid3(resolution=(3, 4, 5),
                                      gridSpacing=(1, 2, 3),
                                      gridOrigin=(7, 5, 2))
    assert_vector_similar(a.dataSize, (3, 4, 5))
    assert_vector_similar(a.dataOrigin, (7.5, 6, 3.5))

    # Modifiers
    b = pyjet.CellCenteredScalarGrid3(resolution=(6, 3, 7),
                                      gridSpacing=(5, 9, 3),
                                      gridOrigin=(1, 2, 8))
    a.fill(42.0)
    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                assert a[i, j, k] == 42.0

    a.swap(b)
    assert a.resolution == (6, 3, 7)
    assert_vector_similar(a.origin, (1, 2, 8))
    assert_vector_similar(a.gridSpacing, (5, 9, 3))
    assert b.resolution == (3, 4, 5)
    assert_vector_similar(b.origin, (7, 5, 2))
    assert_vector_similar(b.gridSpacing, (1, 2, 3))
    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                assert a[i, j, k] == 0.0
    for k in range(b.resolution.z):
        for j in range(b.resolution.y):
            for i in range(b.resolution.x):
                assert b[i, j, k] == 42.0

    a.set(b)
    assert a.resolution == (3, 4, 5)
    assert_vector_similar(a.origin, (7, 5, 2))
    assert_vector_similar(a.gridSpacing, (1, 2, 3))
    for k in range(a.resolution.z):
        for j in range(a.resolution.y):
            for i in range(a.resolution.x):
                assert a[i, j, k] == 42.0

    c = a.clone()
    assert c.resolution == (3, 4, 5)
    assert_vector_similar(c.origin, (7, 5, 2))
    assert_vector_similar(c.gridSpacing, (1, 2, 3))
    for k in range(c.resolution.z):
        for j in range(c.resolution.y):
            for i in range(c.resolution.x):
                assert c[i, j, k] == 42.0
