// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>
#include <jet/jet.h>

using namespace jet;

JET_TESTS(Tutorial);

JET_BEGIN_TEST_F(Tutorial, FlipMinimal) {
    auto solver = FlipSolver3::builder()
        .withResolution({32, 64, 32})
        .withDomainSizeX(1.0)
        .makeShared();

    auto sphere = Sphere3::builder()
        .withCenter({0.5, 1.0, 0.5})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
        .withSurface(sphere)
        .withSpacing(0.5 / 64.0)
        .makeShared();

    solver->setParticleEmitter(emitter);

    auto anotherSphere = Sphere3::builder()
        .withCenter({0.5, 0.5, 0.5})
        .withRadius(0.15)
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(anotherSphere)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
