"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet


def test_parameters():
    data = pyjet.SphSystemData2()

    data.targetDensity = 123.0
    data.targetSpacing = 0.549
    data.relativeKernelRadius = 2.5

    assert data.targetDensity == 123.0
    assert data.targetSpacing == 0.549
    assert data.radius == 0.549
    assert data.relativeKernelRadius == 2.5
    assert data.kernelRadius == 2.5 * 0.549

    data.kernelRadius = 1.9
    assert data.kernelRadius == 1.9
    assert data.targetSpacing == 1.9 / 2.5

    data.radius = 0.413
    assert data.targetSpacing == 0.413
    assert data.radius == 0.413
    assert data.relativeKernelRadius == 2.5
    assert data.kernelRadius == 2.5 * 0.413

    data.mass = 2.0 * data.mass
    assert data.targetDensity == 246.0
