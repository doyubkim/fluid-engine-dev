// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/box2.h>
#include <jet/pic_solver2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;

JET_TESTS(PicSolver2);

JET_BEGIN_TEST_F(PicSolver2, SteadyState) {
    // Build solver
    auto solver = PicSolver2::builder()
        .withResolution({32, 32})
        .withDomainSizeX(1.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.0, 0.0})
        .withUpperCorner({1.0, 0.5})
        .makeShared();

    auto emitter = VolumeParticleEmitter2::builder()
        .withSurface(box)
        .withSpacing(1.0 / 64.0)
        .withIsOneShot(true)
        .makeShared();

    solver->setParticleEmitter(emitter);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PicSolver2, DamBreaking) {
    // Build solver
    auto solver = PicSolver2::builder()
        .withResolution({100, 100})
        .withDomainSizeX(1.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.0, 0.0})
        .withUpperCorner({0.2, 0.8})
        .makeShared();

    auto emitter = VolumeParticleEmitter2::builder()
        .withSurface(box)
        .withSpacing(0.005)
        .withIsOneShot(true)
        .makeShared();

    solver->setParticleEmitter(emitter);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PicSolver2, DamBreakingWithCollider) {
    // Build solver
    auto solver = PicSolver2::builder()
        .withResolution({100, 100})
        .withDomainSizeX(1.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.0, 0.0})
        .withUpperCorner({0.2, 0.8})
        .makeShared();

    auto emitter = VolumeParticleEmitter2::builder()
        .withSurface(box)
        .withSpacing(0.005)
        .withIsOneShot(true)
        .makeShared();

    solver->setParticleEmitter(emitter);

    // Build collider
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.0})
        .withRadius(0.15)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame(0, 1.0 / 60.0); frame.index < 240; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
