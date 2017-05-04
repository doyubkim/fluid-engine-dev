#!/usr/bin/env python

from pyjet import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    solver = GridSmokeSolver2(resolution=(32, 64), domainSizeX=1.0)

    sphere = Sphere2(center=(0.5, 0.5), radius=0.15)
    emitter = VolumeGridEmitter2(sourceRegion=sphere)
    solver.emitter = emitter
    emitter.addStepFunctionTarget(solver.smokeDensity, 0.0, 1.0)
    emitter.addStepFunctionTarget(solver.temperature, 0.0, 1.0)

    frame = Frame()
    while frame.index < 120:
        solver.update(frame)
        frame.advance()

    den = np.array(solver.smokeDensity.dataAccessor(), copy=False)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    den = np.flipud(den)
    im = ax.imshow(den, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    Logging.mute()
    main()
