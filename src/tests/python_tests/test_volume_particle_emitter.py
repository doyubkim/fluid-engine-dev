"""
Copyright (c) 2019 Doyub Kim

I am making my contributions/submissions to this project solely in my personal
capacity and am not conveying any rights to any intellectual property of any
third parties.
"""

import pyjet
from pytest_utils import *


def test_volume_particle_emitter2():
    # Basic ctor test
    sphere = pyjet.Sphere2()
    emitter = pyjet.VolumeParticleEmitter2(
        sphere,
        pyjet.BoundingBox2D((-1, -2), (4, 2)),
        0.1,
        (-1, 0.5),
        (3, 4),
        5.0,
        30,
        0.01,
        False,
        True,
        42)

    assert emitter.surface
    assert_bounding_box_similar(
        emitter.maxRegion, pyjet.BoundingBox2D((-1, -2), (4, 2)))
    assert emitter.spacing == 0.1
    assert_vector_similar(emitter.initialVelocity, (-1, 0.5))
    assert_vector_similar(emitter.linearVelocity, (3, 4))
    assert emitter.angularVelocity == 5.0
    assert emitter.maxNumberOfParticles == 30
    assert emitter.jitter == 0.01
    assert not emitter.isOneShot
    assert emitter.allowOverlapping
    assert emitter.isEnabled

    # Another basic ctor test
    emitter2 = pyjet.VolumeParticleEmitter2(
        implicitSurface=sphere,
        maxRegion=pyjet.BoundingBox2D((-1, -2), (4, 2)),
        spacing=0.1,
        initialVelocity=(-1, 0.5),
        linearVelocity=(3, 4),
        angularVelocity=5.0,
        maxNumberOfParticles=3000,
        jitter=0.01,
        isOneShot=False,
        allowOverlapping=True,
        seed=42)

    assert emitter2.surface
    assert_bounding_box_similar(
        emitter2.maxRegion, pyjet.BoundingBox2D((-1, -2), (4, 2)))
    assert emitter2.spacing == 0.1
    assert_vector_similar(emitter2.initialVelocity, (-1, 0.5))
    assert_vector_similar(emitter2.linearVelocity, (3, 4))
    assert emitter2.angularVelocity == 5.0
    assert emitter2.maxNumberOfParticles == 3000
    assert emitter2.jitter == 0.01
    assert not emitter2.isOneShot
    assert emitter2.allowOverlapping
    assert emitter2.isEnabled

    # Emit some particles
    frame = pyjet.Frame()
    solver = pyjet.ParticleSystemSolver2()
    solver.emitter = emitter2
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > 0

    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > old_num_particles

    # Disabling emitter should stop emitting particles
    emitter2.isEnabled = False
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles == old_num_particles

    # Re-enabling emitter should resume emission
    emitter2.isEnabled = True
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > old_num_particles

    # One-shot emitter
    emitter3 = pyjet.VolumeParticleEmitter2(
        implicitSurface=sphere,
        maxRegion=pyjet.BoundingBox2D((-1, -2), (4, 2)),
        spacing=0.1,
        initialVelocity=(-1, 0.5),
        linearVelocity=(3, 4),
        angularVelocity=5.0,
        maxNumberOfParticles=3000,
        jitter=0.01,
        isOneShot=True,
        allowOverlapping=True,
        seed=42)

    # Emit some particles
    frame = pyjet.Frame()
    solver = pyjet.ParticleSystemSolver2()
    solver.emitter = emitter3
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > 0
    assert not emitter3.isEnabled

    # Should not emit more particles
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles == old_num_particles

    # Re-enabling the emitter should make it emit one more time
    emitter3.isEnabled = True
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > old_num_particles

    # ...and gets disabled again
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles == old_num_particles


def test_volume_particle_emitter3():
    sphere = pyjet.Sphere3()
    emitter = pyjet.VolumeParticleEmitter3(
        sphere,
        pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 9)),
        0.1,
        (-1, 0.5, 2),
        (3, 4, 5),
        (6, 7, 8),
        30,
        0.01,
        False,
        True,
        42)

    assert emitter.surface
    assert_bounding_box_similar(
        emitter.maxRegion, pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 9)))
    assert emitter.spacing == 0.1
    assert_vector_similar(emitter.initialVelocity, (-1, 0.5, 2))
    assert_vector_similar(emitter.linearVelocity, (3, 4, 5))
    assert_vector_similar(emitter.angularVelocity, (6, 7, 8))
    assert emitter.maxNumberOfParticles == 30
    assert emitter.jitter == 0.01
    assert not emitter.isOneShot
    assert emitter.allowOverlapping
    assert emitter.isEnabled

    emitter2 = pyjet.VolumeParticleEmitter3(
        implicitSurface=sphere,
        maxRegion=pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 9)),
        spacing=0.1,
        initialVelocity=(-1, 0.5, 2),
        linearVelocity=(3, 4, 5),
        angularVelocity=(6, 7, 8),
        maxNumberOfParticles=300000,
        jitter=0.01,
        isOneShot=False,
        allowOverlapping=True,
        seed=42)

    assert emitter2.surface
    assert_bounding_box_similar(
        emitter2.maxRegion, pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 9)))
    assert emitter2.spacing == 0.1
    assert_vector_similar(emitter2.initialVelocity, (-1, 0.5, 2))
    assert_vector_similar(emitter2.linearVelocity, (3, 4, 5))
    assert_vector_similar(emitter2.angularVelocity, (6, 7, 8))
    assert emitter2.maxNumberOfParticles == 300000
    assert emitter2.jitter == 0.01
    assert not emitter2.isOneShot
    assert emitter2.allowOverlapping
    assert emitter2.isEnabled

    # Emit some particles
    frame = pyjet.Frame()
    solver = pyjet.ParticleSystemSolver3()
    solver.emitter = emitter2
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > 0

    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > old_num_particles

    # Disabling emitter should stop emitting particles
    emitter2.isEnabled = False
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles == old_num_particles

    # Re-enabling emitter should resume emission
    emitter2.isEnabled = True
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > old_num_particles

    # One-shot emitter
    emitter3 = pyjet.VolumeParticleEmitter3(
        implicitSurface=sphere,
        maxRegion=pyjet.BoundingBox3D((-1, -2, -3), (4, 2, 9)),
        spacing=0.1,
        initialVelocity=(-1, 0.5, 2),
        linearVelocity=(3, 4, 5),
        angularVelocity=(6, 7, 8),
        maxNumberOfParticles=300000,
        jitter=0.01,
        isOneShot=True,
        allowOverlapping=True,
        seed=42)

    # Emit some particles
    frame = pyjet.Frame()
    solver = pyjet.ParticleSystemSolver3()
    solver.emitter = emitter3
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > 0
    assert not emitter3.isEnabled

    # Should not emit more particles
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles == old_num_particles

    # Re-enabling the emitter should make it emit one more time
    emitter3.isEnabled = True
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles > old_num_particles

    # ...and gets disabled again
    old_num_particles = solver.particleSystemData.numberOfParticles
    solver.update(frame)
    frame.advance()
    assert solver.particleSystemData.numberOfParticles == old_num_particles