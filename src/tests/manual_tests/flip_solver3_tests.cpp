// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/box3.h>
#include <jet/cylinder3.h>
#include <jet/flip_solver3.h>
#include <jet/grid_point_generator3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/implicit_surface_set3.h>
#include <jet/particle_emitter_set3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(FlipSolver3);

JET_BEGIN_TEST_F(FlipSolver3, WaterDrop) {
    // Build solver
    auto solver = FlipSolver3::builder()
        .withResolution({32, 64, 32})
        .withDomainSizeX(1.0)
        .makeShared();

    double dx = solver->gridSystemData()->gridSpacing().x;
    BoundingBox3D domain = solver->gridSystemData()->boundingBox();

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
    // Build solver
    auto solver = FlipSolver3::builder()
        .withResolution({150, 100, 75})
        .withDomainSizeX(3.0)
        .makeShared();

    double dx = solver->gridSystemData()->gridSpacing().x;
    BoundingBox3D domain = solver->gridSystemData()->boundingBox();
    double lz = domain.depth();

    // Build emitter
    auto box1 = Box3::builder()
        .withLowerCorner({0, 0, 0})
        .withUpperCorner({0.5 + 0.001, 0.75 + 0.001, 0.75 * lz + 0.001})
        .makeShared();

    auto box2 = Box3::builder()
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
        .withCenter({1, -0.75 / 2.0, 0.25 * lz})
        .withRadius(0.1)
        .withHeight(0.75)
        .makeShared();

    auto cyl2 = Cylinder3::builder()
        .withCenter({1.5, -0.75 / 2.0, 0.5 * lz})
        .withRadius(0.1)
        .withHeight(0.75)
        .makeShared();

    auto cyl3 = Cylinder3::builder()
        .withCenter({2, -0.75 / 2.0, 0.75 * lz})
        .withRadius(0.1)
        .withHeight(0.75)
        .makeShared();

    auto cylSet = ImplicitSurfaceSet3::builder()
        .withExplicitSurfaces({cyl1, cyl2, cyl3})
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(cylSet)
        .makeShared();

    solver->setCollider(collider);

    // Run simulation
    for (Frame frame; frame.index < 200; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
