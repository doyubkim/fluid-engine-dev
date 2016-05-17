// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/flip_solver2.h>
#include <jet/flip_solver3.h>
#include <jet/grid_points_generator2.h>
#include <jet/grid_points_generator3.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>
#include <jet/implicit_surface_set2.h>
#include <jet/implicit_surface_set3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider2.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere2.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>

using namespace jet;

JET_TESTS(FlipSolver2);

JET_BEGIN_TEST_F(FlipSolver2, Empty) {
    FlipSolver2 solver;

    for (Frame frame; frame.index < 1; frame.advance()) {
        solver.update(frame);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver2, SteadyState) {
    FlipSolver2 solver;

    GridSystemData2Ptr grid = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    grid->resize(Size2(32, 32), Vector2D(dx, dx), Vector2D());

    GridPointsGenerator2 pointsGen;
    Array1<Vector2D> points;
    pointsGen.generate(
        BoundingBox2D(Vector2D(), Vector2D(1.0, 0.5)), 0.5 * dx, &points);

    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);

    auto sdf = solver.signedDistanceField();
    saveData(sdf->constDataAccessor(), "sdf_#grid2,0000.npy");

    for (Frame frame; frame.index < 120; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index + 1);

        char filename[256];
        snprintf(
            filename,
            sizeof(filename),
            "sdf_#grid2,%04d.npy",
            frame.index + 1);
        saveData(sdf->constDataAccessor(), filename);
    }

    Array2<double> dataU(32, 32);
    Array2<double> dataV(32, 32);
    auto velocity = grid->velocity();

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = velocity->valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver2, DamBreaking) {
    FlipSolver2 solver;

    GridSystemData2Ptr grid = solver.gridSystemData();
    double dx = 1.0 / 64.0;
    grid->resize(Size2(64, 64), Vector2D(dx, dx), Vector2D());

    GridPointsGenerator2 pointsGen;
    Array1<Vector2D> points;
    pointsGen.generate(
        BoundingBox2D(Vector2D(), Vector2D(0.2, 0.6)), 0.5 * dx, &points);

    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);

    for (Frame frame; frame.index < 240; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index + 1);
    }

    Array2<double> dataU(64, 64);
    Array2<double> dataV(64, 64);
    auto velocity = grid->velocity();

    dataU.forEachIndex([&](size_t i, size_t j) {
        Vector2D vel = velocity->valueAtCellCenter(i, j);
        dataU(i, j) = vel.x;
        dataV(i, j) = vel.y;
    });

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
    saveData(
        solver.signedDistanceField()->constDataAccessor(),
        "sdf_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FlipSolver2, DamBreakingWithCollider) {
    FlipSolver2 solver;

    // Collider setting
    auto sphere = std::make_shared<Sphere2>(
        Vector2D(0.5, 0.0), 0.15);
    auto surface = std::make_shared<SurfaceToImplicit2>(sphere);
    auto collider = std::make_shared<RigidBodyCollider2>(surface);
    solver.setCollider(collider);

    GridSystemData2Ptr grid = solver.gridSystemData();
    double dx = 1.0 / 100.0;
    grid->resize(Size2(100, 100), Vector2D(dx, dx), Vector2D());

    GridPointsGenerator2 pointsGen;
    Array1<Vector2D> points;
    pointsGen.generate(
        BoundingBox2D(Vector2D(), Vector2D(0.2, 0.8)), 0.5 * dx, &points);

    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);

    for (Frame frame; frame.index < 240; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index + 1);
    }
}
JET_END_TEST_F


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
    surfaceSet.addSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet.addSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    // Initialize particles
    GridPointsGenerator3 pointsGen;
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
    for (Frame frame; frame.index < 120; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index + 1);
    }
}
JET_END_TEST_F
