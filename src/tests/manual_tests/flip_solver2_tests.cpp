// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/box2.h>
#include <jet/flip_solver2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;

JET_TESTS(FlipSolver2);

JET_BEGIN_TEST_F(FlipSolver2, SteadyState) {
    // Build solver
    auto solver = FlipSolver2::builder()
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

JET_BEGIN_TEST_F(FlipSolver2, DamBreaking) {
    // Build solver
    auto solver = FlipSolver2::builder()
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

JET_BEGIN_TEST_F(FlipSolver2, DamBreakingWithCollider) {
    // Build solver
    auto solver = FlipSolver2::builder()
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

JET_BEGIN_TEST_F(FlipSolver2, RotatingTank) {
    // Build solver
    auto solver = FlipSolver2::builder()
        .withResolution({32, 32})
        .withDomainSizeX(1.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.25, 0.25})
        .withUpperCorner({0.75, 0.50})
        .makeShared();

    auto emitter = VolumeParticleEmitter2::builder()
        .withSurface(box)
        .withSpacing(1.0 / 64.0)
        .withIsOneShot(true)
        .makeShared();

    solver->setParticleEmitter(emitter);

    // Build collider
    auto tank = Box2::builder()
        .withLowerCorner({-0.25, -0.25})
        .withUpperCorner({ 0.25,  0.25})
        .withTranslation({0.5, 0.5})
        .withOrientation(0.0)
        .withIsNormalFlipped(true)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(tank)
        .withAngularVelocity(2.0)
        .makeShared();

    collider->setOnBeginUpdateCallback([&] (Collider2* col, double t, double) {
        if (t < 1.0) {
            col->surface()->transform.setOrientation(2.0 * t);
            static_cast<RigidBodyCollider2*>(col)->angularVelocity = 2.0;
        } else {
            static_cast<RigidBodyCollider2*>(col)->angularVelocity = 0.0;
        }
    });

    solver->setCollider(collider);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
