#!/usr/bin/env python

from pyjet import *

def main():
    solver = FlipSolver3(resolution=(32, 64, 32), domainSizeX=1.0)
    sphere = Sphere3(center=(0.5, 1.0, 0.5), radius=0.15)
    emitter = VolumeParticleEmitter3(implicitSurface=sphere, spacing=1.0/128.0)
    solver.particleEmitter = emitter

    frame = Frame()
    while frame.index < 120:
        solver.update(frame)
        frame.advance()

if __name__ == '__main__':
    Logging.mute()
    main()
