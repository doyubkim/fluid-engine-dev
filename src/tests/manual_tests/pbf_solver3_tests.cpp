// Copyright (c) 2017 Doyub Kim

#include <manual_tests.h>

#include <jet/box3.h>
#include <jet/rigid_body_collider3.h>
#include <jet/pbf_solver3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(PbfSolver3);

JET_BEGIN_TEST_F(PbfSolver3, SteadyState) {
    // Build solver
    auto solver = PbfSolver3::builder()
        .withTargetDensity(1000.0)
        .withTargetSpacing(0.05)
        .makeShared();

    const auto particles = solver->sphSystemData();
    const double targetSpacing = particles->targetSpacing();

    // Build emitter
    auto box = Box3::builder()
        .withLowerCorner({targetSpacing, targetSpacing, targetSpacing})
        .withUpperCorner({1.0 - targetSpacing, 0.5, 1.0 - targetSpacing})
        .makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
        .withSurface(box)
        .withSpacing(targetSpacing)
        .withIsOneShot(true)
        .makeShared();

    solver->setEmitter(emitter);

    // Build collider
    auto anotherBox = Box3::builder()
        .withLowerCorner({0, 0, 0})
        .withUpperCorner({1, 1, 1})
        .withIsNormalFlipped(true)
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(anotherBox)
        .makeShared();

    solver->setCollider(collider);

    // Simulate
    for (Frame frame; frame.index < 100; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PbfSolver3, DamBreaking) {
    // Build solver
    auto solver = PbfSolver3::builder()
        .withTargetDensity(1000.0)
        .withTargetSpacing(0.01)
        .makeShared();

    solver->setMaxNumberOfIterations(20);

    const auto particles = solver->sphSystemData();
    const double targetSpacing = particles->targetSpacing();

    // Build emitter
    auto box = Box3::builder()
        .withLowerCorner({targetSpacing, targetSpacing, targetSpacing})
        .withUpperCorner({0.2, 0.8, 0.2})
        .makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
        .withSurface(box)
        .withSpacing(targetSpacing)
        .withIsOneShot(true)
        .makeShared();

    solver->setEmitter(emitter);

    // Build collider
    auto anotherBox = Box3::builder()
        .withLowerCorner({0, 0, 0})
        .withUpperCorner({1, 1, 1})
        .withIsNormalFlipped(true)
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(anotherBox)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
