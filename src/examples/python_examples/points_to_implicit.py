#!/usr/bin/env python

"""
Copyright (c) 2018 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

from pyjet import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(0)
    points = np.random.rand(100, 2) * 0.6 + 0.2

    grid = CellCenteredScalarGrid2((512, 512), (1.0 / 512.0, 1.0 / 512.0))

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    kernel_radius = 0.1
    cutoff = 0.5

    converter = SphericalPointsToImplicit2(kernel_radius * cutoff)
    converter.convert(points.tolist(), grid)
    plt.contour(grid.dataAccessor(), levels=[0.0], colors=("g"))

    converter = SphPointsToImplicit2(kernel_radius, cutoff)
    converter.convert(points.tolist(), grid)
    plt.contour(grid.dataAccessor(), levels=[0.0], colors=("b"))

    converter = ZhuBridsonPointsToImplicit2(2.0 * kernel_radius, 0.5 * cutoff)
    converter.convert(points.tolist(), grid)
    plt.contour(grid.dataAccessor(), levels=[0.0], colors=("purple"))

    converter = AnisotropicPointsToImplicit2(kernel_radius, cutoff, 0.0, 8)
    converter.convert(points.tolist(), grid)
    plt.contour(grid.dataAccessor(), levels=[0.0], colors=("r"))

    plt.scatter(points[:, 0] * 512, points[:, 1] * 512, c="black")
    plt.show()


if __name__ == "__main__":
    Logging.mute()
    main()
