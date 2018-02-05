// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>
#include <manual_tests.h>

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

    auto collider =
        RigidBodyCollider3::builder().withSurface(anotherSphere).makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(Tutorial, TriangleMeshDemo) {
    const size_t resX = 50;

    auto solver = ApicSolver3::builder()
                      .withResolution({resX, 3 * resX, resX})
                      .withDomainSizeX(4)
                      .withOrigin({-2, -2, -2})
                      .makeShared();

    double gridSpacing = solver->gridSpacing().x;

    auto sphere1 = Sphere3::builder()
                       .withCenter({-0.01, 1.0, -0.01})
                       .withRadius(0.4)
                       .makeShared();
    auto sphere2 = Sphere3::builder()
                       .withCenter({0.01, 2.0, 0.01})
                       .withRadius(0.4)
                       .makeShared();
    auto spheres =
        SurfaceSet3::builder().withSurfaces({sphere1, sphere2}).makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
                       .withSurface(spheres)
                       .withSpacing(0.5 * gridSpacing)
                       .withJitter(0.1)
                       .makeShared();

    solver->setParticleEmitter(emitter);

    auto cupMesh = TriangleMesh3::builder().makeShared();
    std::ifstream file(getResourceFileName("cup.obj").c_str());
    if (file) {
        cupMesh->readObj(&file);
        file.close();
    }

    auto cup = ImplicitTriangleMesh3::builder()
                   .withTriangleMesh(cupMesh)
                   .withResolutionX(100)
                   .makeShared();

    auto collider = RigidBodyCollider3::builder().withSurface(cup).makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F