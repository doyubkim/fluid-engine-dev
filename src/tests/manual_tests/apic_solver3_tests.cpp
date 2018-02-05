// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/apic_solver3.h>
#include <jet/box3.h>
#include <jet/cylinder3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/grid_point_generator3.h>
#include <jet/grid_single_phase_pressure_solver3.h>
#include <jet/implicit_surface_set3.h>
#include <jet/level_set_utils.h>
#include <jet/particle_emitter_set3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(ApicSolver3);

JET_BEGIN_TEST_F(ApicSolver3, WaterDrop) {
    //
    // This is a replica of hybrid_liquid_sim example 2.
    //

    size_t resolutionX = 32;

    // Build solver
    auto solver =
        ApicSolver3::builder()
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

JET_BEGIN_TEST_F(ApicSolver3, DamBreakingWithCollider) {
    //
    // This is a replica of hybrid_liquid_sim example 4.
    //

    size_t resolutionX = 50;

    // Build solver
    Size3 resolution{3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2};
    auto solver = ApicSolver3::builder()
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

JET_BEGIN_TEST_F(ApicSolver3, Spherical) {
    // Build solver
    auto solver = ApicSolver3::builder()
                      .withResolution({30, 30, 30})
                      .withDomainSizeX(1.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    // Build collider
    auto sphere = Sphere3::builder()
                      .withCenter({0.5, 0.5, 0.5})
                      .withRadius(0.4)
                      .withIsNormalFlipped(true)
                      .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(sphere).makeShared();

    solver->setCollider(collider);

    // Manually emit particles
    size_t resX = solver->resolution().x;
    std::mt19937 rng;
    std::uniform_real_distribution<> dist(0, 1);
    for (int i = 0; i < 8 * resX * resX * resX; ++i) {
        Vector3D pt{dist(rng), dist(rng), dist(rng)};
        if ((pt - sphere->center).length() < sphere->radius && pt.x > 0.5) {
            solver->particleSystemData()->addParticle(pt);
        }
    }

    for (Frame frame(0, 0.01); frame.index < 240; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ApicSolver3, SphericalNonVariational) {
    // Build solver
    auto solver = ApicSolver3::builder()
                      .withResolution({30, 30, 30})
                      .withDomainSizeX(1.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    solver->setPressureSolver(
        std::make_shared<GridSinglePhasePressureSolver3>());

    // Build collider
    auto sphere = Sphere3::builder()
                      .withCenter({0.5, 0.5, 0.5})
                      .withRadius(0.4)
                      .withIsNormalFlipped(true)
                      .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(sphere).makeShared();

    solver->setCollider(collider);

    // Manually emit particles
    size_t resX = solver->resolution().x;
    std::mt19937 rng;
    std::uniform_real_distribution<> dist(0, 1);
    for (int i = 0; i < 8 * resX * resX * resX; ++i) {
        Vector3D pt{dist(rng), dist(rng), dist(rng)};
        if ((pt - sphere->center).length() < sphere->radius && pt.x > 0.5) {
            solver->particleSystemData()->addParticle(pt);
        }
    }

    for (Frame frame(0, 0.01); frame.index < 240; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
