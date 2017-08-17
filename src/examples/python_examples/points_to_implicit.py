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
    points = np.random.rand(200, 2) * 0.6 + 0.2
    grid = CellCenteredScalarGrid2((512, 512), (1.0 / 512.0, 1.0 / 512.0))

    converter = AnisotropicPointsToImplicit2(0.15)
    converter.convert(points.tolist(), grid)

    # Visualization
    fig = plt.figure()
    den = np.array(grid.dataAccessor(), copy=False)
    # im = plt.imshow(den, cmap=plt.cm.gray, origin="lower")
    cs = plt.contour(grid.dataAccessor(), 1)
    plt.show()


if __name__ == "__main__":
    Logging.mute()
    main()
