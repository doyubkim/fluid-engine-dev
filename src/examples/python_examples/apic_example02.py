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
import matplotlib.animation as animation

ANIM_NUM_FRAMES = 360
ANIM_FPS = 60


def main():
    # Create APIC solver
    resX = 32
    solver = ApicSolver3(resolution=(resX, 2 * resX, resX), domainSizeX=1.0)

    pressureSolver = GridFractionalSinglePhasePressureSolver3()
    pressureSolver.linearSystemSolver = FdmMgpcgSolver3(20, 3)
    solver.pressureSolver = pressureSolver

    # Setup emitter
    sphere = Sphere3(center=(0.5, 1.0, 0.5), radius=0.15)
    emitter = VolumeParticleEmitter3(
        implicitSurface=sphere, spacing=1.0 / (2 * resX))
    solver.particleEmitter = emitter

    # Setup collider
    anotherSphere = Sphere3(center=(0.5, 0.5, 0.5), radius=0.15)
    collider = RigidBodyCollider3(surface=anotherSphere)
    solver.collider = collider

    # Visualization
    fig = plt.figure(figsize=(3, 6))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 2), ax.set_yticks([])

    # Make first frame
    frame = Frame(0, 1.0 / ANIM_FPS)
    solver.update(frame)
    frame.advance()

    # Visualization
    pos = np.array(solver.particleSystemData.positions, copy=False)
    scat = ax.scatter(pos[:, 0], pos[:, 1])

    # Animation
    def updatefig(*args):
        solver.update(frame)
        frame.advance()
        pos = np.array(solver.particleSystemData.positions, copy=False)
        scat.set_offsets(np.vstack((pos[:, 0], pos[:, 1])).transpose())
        return scat,

    anim = animation.FuncAnimation(fig, updatefig, frames=ANIM_NUM_FRAMES,
                                   interval=1, blit=True)
    plt.show()


if __name__ == '__main__':
    Logging.mute()
    main()
