"""
Copyright (c) 2019 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
from pytest import approx
from pytest_utils import *


def test_marching_cubes_connectivity():
    grid = pyjet.VertexCenteredScalarGrid3((1, 1, 1))
    grid[(0, 0, 0)] = -0.5
    grid[(0, 0, 1)] = -0.5
    grid[(0, 1, 0)] = 0.5
    grid[(0, 1, 1)] = 0.5
    grid[(1, 0, 0)] = -0.5
    grid[(1, 0, 1)] = -0.5
    grid[(1, 1, 0)] = 0.5
    grid[(1, 1, 1)] = 0.5

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_NONE)
    assert mesh.numberOfPoints() == 24

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_BACK)
    assert mesh.numberOfPoints() == 22

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_FRONT)
    assert mesh.numberOfPoints() == 22

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_LEFT)
    assert mesh.numberOfPoints() == 22

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_RIGHT)
    assert mesh.numberOfPoints() == 22

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_DOWN)
    assert mesh.numberOfPoints() == 24

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_UP)
    assert mesh.numberOfPoints() == 24

    mesh = pyjet.marchingCubes(grid, (1, 1, 1), (0, 0, 0), 0.0, pyjet.DIRECTION_ALL, pyjet.DIRECTION_ALL)
    assert mesh.numberOfPoints() == 8
