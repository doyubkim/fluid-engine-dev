// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/grid_point_generator2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/pic_solver2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

JET_TESTS(PicSolver2);

JET_BEGIN_TEST_F(PicSolver2, Empty) {
    PicSolver2 solver;

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 1; frame.advance()) {
        solver.update(frame);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PicSolver2, SteadyState) {
    PicSolver2 solver;

    GridSystemData2Ptr grid = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    grid->resize(Size2(32, 32), Vector2D(dx, dx), Vector2D());

    GridPointGenerator2 pointsGen;
    Array1<Vector2D> points;
    pointsGen.generate(
        BoundingBox2D(Vector2D(), Vector2D(1.0, 0.5)), 0.5 * dx, &points);

    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 60; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
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

JET_BEGIN_TEST_F(PicSolver2, DamBreaking) {
    PicSolver2 solver;

    GridSystemData2Ptr grid = solver.gridSystemData();
    double dx = 1.0 / 64.0;
    grid->resize(Size2(64, 64), Vector2D(dx, dx), Vector2D());

    GridPointGenerator2 pointsGen;
    Array1<Vector2D> points;
    pointsGen.generate(
        BoundingBox2D(Vector2D(), Vector2D(0.2, 0.6)), 0.5 * dx, &points);

    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 240; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
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
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PicSolver2, DamBreakingWithCollider) {
    PicSolver2 solver;

    // Collider setting
    auto sphere = std::make_shared<Sphere2>(
        Vector2D(0.5, 0.0), 0.15);
    auto surface = std::make_shared<SurfaceToImplicit2>(sphere);
    auto collider = std::make_shared<RigidBodyCollider2>(surface);
    solver.setCollider(collider);

    GridSystemData2Ptr grid = solver.gridSystemData();
    double dx = 1.0 / 100.0;
    grid->resize(Size2(100, 100), Vector2D(dx, dx), Vector2D());

    GridPointGenerator2 pointsGen;
    Array1<Vector2D> points;
    pointsGen.generate(
        BoundingBox2D(Vector2D(), Vector2D(0.2, 0.8)), 0.5 * dx, &points);

    auto particles = solver.particleSystemData();
    particles->addParticles(points);

    saveParticleDataXy(particles, 0);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 240; frame.advance()) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F
