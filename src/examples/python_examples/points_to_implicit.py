#!/usr/bin/env python

"""
Copyright (c) 2017 Doyub Kim

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
    ax.set_aspect('equal')

    converter = AnisotropicPointsToImplicit2(0.1, 0.5, 0.0, 8)
    converter.convert(points.tolist(), grid)
    den = np.array(grid.dataAccessor(), copy=False)
    plt.imshow(den, cmap=plt.cm.gray, origin="lower")
    plt.contour(grid.dataAccessor(), levels=[0.0], colors=('r'))

    converter = AnisotropicPointsToImplicit2(0.1, 0.5, 0.0, 1000)
    converter.convert(points.tolist(), grid)
    plt.contour(grid.dataAccessor(), levels=[0.0], colors=('b'))

    plt.scatter(points[:, 0] * 512, points[:, 1] * 512)
    plt.show()


if __name__ == "__main__":
    Logging.mute()
    main()
