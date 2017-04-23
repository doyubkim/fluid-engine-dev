#!/usr/bin/env python

from pyjet import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    solver = FlipSolver3(resolution=(32, 64, 32), domainSizeX=1.0)

    sphere = Sphere3(center=(0.5, 1.0, 0.5), radius=0.15)
    emitter = VolumeParticleEmitter3(implicitSurface=sphere, spacing=1.0/128.0)
    solver.particleEmitter = emitter

    anotherSphere = Sphere3(center=(0.5, 0.5, 0.5), radius=0.15)
    collider = RigidBodyCollider3(surface=anotherSphere)
    solver.collider = collider

    frame = Frame()
    while frame.index < 120:
        solver.update(frame)
        pos = np.array(solver.particleSystemData.positions, copy=False)
        frame.advance()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(pos[:,0], pos[:,1])
    plt.show()

if __name__ == '__main__':
    Logging.mute()
    main()
