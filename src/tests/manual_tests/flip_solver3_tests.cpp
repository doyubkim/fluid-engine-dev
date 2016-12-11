// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/box3.h>
#include <jet/cylinder3.h>
#include <jet/flip_solver3.h>
#include <jet/grid_point_generator3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/implicit_surface_set3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(FlipSolver3);

JET_BEGIN_TEST_F(FlipSolver3, WaterDrop) {
    size_t resolutionX = 32;
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    FlipSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    // Initialize particles
    GridPointGenerator3 pointsGen;
    Array1<Vector3D> points;
    pointsGen.forEachPoint(
        domain,
        0.5 * dx,
        [&](const Vector3D& pt) {
            if (isInsideSdf(surfaceSet.signedDistance(pt))) {
                points.append(pt);
            }
            return true;
        });
    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);
    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver3, DamBreakingWithCollider) {
    size_t resolutionX = 50;
    Size3 resolution(3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    FlipSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(0, 0, 0),
            Vector3D(0.5 + 0.001, 0.75 + 0.001, 0.75 * lz + 0.001)));
    surfaceSet->addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(2.5 - 0.001, 0, 0.25 * lz - 0.001),
            Vector3D(3.5 + 0.001, 0.75 + 0.001, 1.5 * lz + 0.001)));

    // Initialize particles
    auto particles = solver.particleSystemData();
    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        domain,
        0.5 * dx,
        Vector3D());
    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    solver.setParticleEmitter(emitter);

    // Collider setting
    double height = 0.75;
    auto columns = std::make_shared<ImplicitSurfaceSet3>();
    columns->addExplicitSurface(
        std::make_shared<Cylinder3>(
            Vector3D(1, -height / 2.0, 0.25 * lz), 0.1, height));
    columns->addExplicitSurface(
        std::make_shared<Cylinder3>(
            Vector3D(1.5, -height / 2.0, 0.5 * lz), 0.1, height));
    columns->addExplicitSurface(
        std::make_shared<Cylinder3>(
            Vector3D(2, -height / 2.0, 0.75 * lz), 0.1, height));
    auto collider = std::make_shared<RigidBodyCollider3>(columns);
    solver.setCollider(collider);

    saveParticleDataXy(particles, 0);
    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 200; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F
