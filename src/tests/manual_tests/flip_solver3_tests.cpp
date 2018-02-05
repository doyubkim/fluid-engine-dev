// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/box3.h>
#include <jet/cylinder3.h>
#include <jet/flip_solver3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/grid_point_generator3.h>
#include <jet/implicit_surface_set3.h>
#include <jet/level_set_utils.h>
#include <jet/particle_emitter_set3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(FlipSolver3);

JET_BEGIN_TEST_F(FlipSolver3, WaterDrop) {
    //
    // This is a replica of hybrid_liquid_sim example 1.
    //

    size_t resolutionX = 32;

    // Build solver
    auto solver =
        FlipSolver3::builder()
            .withResolution({resolutionX, 2 * resolutionX, resolutionX})
            .withDomainSizeX(1.0)
            .makeShared();

    auto grids = solver->gridSystemData();
    auto particles = solver->particleSystemData();

    Vector3D gridSpacing = grids->gridSpacing();
    double dx = gridSpacing.x;
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto plane = Plane3::builder()
                     .withNormal({0, 1, 0})
                     .withPoint({0, 0.25 * domain.height(), 0})
                     .makeShared();

    auto sphere = Sphere3::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto emitter1 = VolumeParticleEmitter3::builder()
                        .withSurface(plane)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter1->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitter2 = VolumeParticleEmitter3::builder()
                        .withSurface(sphere)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter2->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitterSet = ParticleEmitterSet3::builder()
                          .withEmitters({emitter1, emitter2})
                          .makeShared();

    solver->setParticleEmitter(emitterSet);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver3, WaterDropWithBlending) {
    size_t resolutionX = 32;

    // Build solver
    auto solver =
        FlipSolver3::builder()
            .withResolution({resolutionX, 2 * resolutionX, resolutionX})
            .withDomainSizeX(1.0)
            .makeShared();

    solver->setPicBlendingFactor(0.05);

    auto grids = solver->gridSystemData();
    auto particles = solver->particleSystemData();

    Vector3D gridSpacing = grids->gridSpacing();
    double dx = gridSpacing.x;
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto plane = Plane3::builder()
                     .withNormal({0, 1, 0})
                     .withPoint({0, 0.25 * domain.height(), 0})
                     .makeShared();

    auto sphere = Sphere3::builder()
                      .withCenter(domain.midPoint())
                      .withRadius(0.15 * domain.width())
                      .makeShared();

    auto emitter1 = VolumeParticleEmitter3::builder()
                        .withSurface(plane)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter1->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitter2 = VolumeParticleEmitter3::builder()
                        .withSurface(sphere)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter2->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitterSet = ParticleEmitterSet3::builder()
                          .withEmitters({emitter1, emitter2})
                          .makeShared();

    solver->setParticleEmitter(emitterSet);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver3, DamBreakingWithCollider) {
    size_t resolutionX = 50;

    //
    // This is a replica of hybrid_liquid_sim example 3.
    //

    // Build solver
    Size3 resolution{3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2};
    auto solver = FlipSolver3::builder()
                      .withResolution(resolution)
                      .withDomainSizeX(3.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    auto grids = solver->gridSystemData();
    double dx = grids->gridSpacing().x;
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Build emitter
    auto box1 =
        Box3::builder()
            .withLowerCorner({0, 0, 0})
            .withUpperCorner({0.5 + 0.001, 0.75 + 0.001, 0.75 * lz + 0.001})
            .makeShared();

    auto box2 =
        Box3::builder()
            .withLowerCorner({2.5 - 0.001, 0, 0.25 * lz - 0.001})
            .withUpperCorner({3.5 + 0.001, 0.75 + 0.001, 1.5 * lz + 0.001})
            .makeShared();

    auto boxSet = ImplicitSurfaceSet3::builder()
                      .withExplicitSurfaces({box1, box2})
                      .makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
                       .withSurface(boxSet)
                       .withMaxRegion(domain)
                       .withSpacing(0.5 * dx)
                       .makeShared();

    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    solver->setParticleEmitter(emitter);

    // Build collider
    auto cyl1 = Cylinder3::builder()
                    .withCenter({1, 0.375, 0.375})
                    .withRadius(0.1)
                    .withHeight(0.75)
                    .makeShared();

    auto cyl2 = Cylinder3::builder()
                    .withCenter({1.5, 0.375, 0.75})
                    .withRadius(0.1)
                    .withHeight(0.75)
                    .makeShared();

    auto cyl3 = Cylinder3::builder()
                    .withCenter({2, 0.375, 1.125})
                    .withRadius(0.1)
                    .withHeight(0.75)
                    .makeShared();

    auto cylSet = ImplicitSurfaceSet3::builder()
                      .withExplicitSurfaces({cyl1, cyl2, cyl3})
                      .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(cylSet).makeShared();

    solver->setCollider(collider);

    // Run simulation
    for (Frame frame; frame.index < 200; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver3, RotatingTank) {
    // Build solver
    auto solver = FlipSolver3::builder()
                      .withResolution({32, 32, 32})
                      .withDomainSizeX(1.0)
                      .makeShared();

    // Build emitter
    auto box = Box3::builder()
                   .withLowerCorner({0.25, 0.25, 0.25})
                   .withUpperCorner({0.75, 0.50, 0.75})
                   .makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
                       .withSurface(box)
                       .withSpacing(1.0 / 64.0)
                       .withIsOneShot(true)
                       .makeShared();

    solver->setParticleEmitter(emitter);

    // Build collider
    auto tank = Box3::builder()
                    .withLowerCorner({-0.25, -0.25, -0.25})
                    .withUpperCorner({0.25, 0.25, 0.25})
                    .withTranslation({0.5, 0.5, 0.5})
                    .withOrientation({{0, 0, 1}, 0.0})
                    .withIsNormalFlipped(true)
                    .makeShared();

    auto collider = RigidBodyCollider3::builder()
                        .withSurface(tank)
                        .withAngularVelocity({0, 0, 2})
                        .makeShared();

    collider->setOnBeginUpdateCallback([&](Collider3* col, double t, double) {
        if (t < 1.0) {
            col->surface()->transform.setOrientation({{0, 0, 1}, 2.0 * t});
            static_cast<RigidBodyCollider3*>(col)->angularVelocity = {0, 0, 2};
        } else {
            static_cast<RigidBodyCollider3*>(col)->angularVelocity = {0, 0, 0};
        }
    });

    solver->setCollider(collider);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
