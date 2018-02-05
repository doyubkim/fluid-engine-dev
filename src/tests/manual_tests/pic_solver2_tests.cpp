// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

JET_BEGIN_TEST_F(PicSolver2, Rotation) {
    // Build solver
    auto solver = PicSolver2::builder()
        .withResolution({10, 10})
        .withDomainSizeX(1.0)
        .makeShared();

    solver->setGravity({0, 0});

    // Build emitter
    auto box = Sphere2::builder()
        .withCenter({0.5, 0.5})
        .withRadius(0.4)
        .makeShared();

    auto emitter = VolumeParticleEmitter2::builder()
        .withSurface(box)
        .withSpacing(1.0 / 20.0)
        .withIsOneShot(true)
        .makeShared();

    solver->setParticleEmitter(emitter);

    Array1<double> r;

    for (Frame frame; frame.index < 360; ++frame) {
        auto x = solver->particleSystemData()->positions();
        auto v = solver->particleSystemData()->velocities();
        r.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            r[i] = (x[i] - Vector2D(0.5, 0.5)).length();
        }

        solver->update(frame);

        if (frame.index == 0) {
            x = solver->particleSystemData()->positions();
            v = solver->particleSystemData()->velocities();
            for (size_t i = 0; i < x.size(); ++i) {
                Vector2D rp = x[i] - Vector2D(0.5, 0.5);
                v[i].x = rp.y;
                v[i].y = -rp.x;
            }
        } else {
            for (size_t i = 0; i < x.size(); ++i) {
                Vector2D rp = x[i] - Vector2D(0.5, 0.5);
                if (rp.lengthSquared() > 0.0) {
                    double scale = r[i] / rp.length();
                    x[i] = scale * rp + Vector2D(0.5, 0.5);
                }
            }
        }

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
