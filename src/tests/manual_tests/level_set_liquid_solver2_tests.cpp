// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/level_set_liquid_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/level_set_utils.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>

using namespace jet;

JET_TESTS(LevelSetLiquidSolver2);

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, Drop) {
    LevelSetLiquidSolver2 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Plane2>(Vector2D(0, 1), Vector2D(0, 0.5)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropHighRes) {
    LevelSetLiquidSolver2 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 128.0;
    data->resize(Size2(128, 256), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Plane2>(Vector2D(0, 1), Vector2D(0, 0.5)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(128, 256);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropWithCollider) {
    LevelSetLiquidSolver2 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 150), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(std::make_shared<Plane2>(
        Vector2D(0, 1), Vector2D(0, 0.5)));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(32, 150);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    // Collider setting
    auto sphere = std::make_shared<Sphere2>(
        Vector2D(domain.midPoint().x, 0.75 - std::cos(0.0)), 0.2);
    auto surface = std::make_shared<SurfaceToImplicit2>(sphere);
    auto collider = std::make_shared<RigidBodyCollider2>(surface);
    solver.setCollider(collider);

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 240; frame.advance()) {
        double t = frame.timeInSeconds();
        sphere->center = Vector2D(domain.midPoint().x, 0.75 - std::cos(t));
        collider->linearVelocity = Vector2D(0, std::sin(t));

        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropVariational) {
    LevelSetLiquidSolver2 solver;
    solver.setPressureSolver(
        std::make_shared<GridFractionalSinglePhasePressureSolver2>());

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 150), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Plane2>(Vector2D(0, 1), Vector2D(0, 0.5)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(32, 150);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropWithColliderVariational) {
    LevelSetLiquidSolver2 solver;
    solver.setPressureSolver(
        std::make_shared<GridFractionalSinglePhasePressureSolver2>());

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 150), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(std::make_shared<Plane2>(
        Vector2D(0, 1), Vector2D(0, 0.5)));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(32, 150);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    // Collider setting
    auto sphere = std::make_shared<Sphere2>(
        Vector2D(domain.midPoint().x, 0.75 - std::cos(0.0)), 0.2);
    auto surface = std::make_shared<SurfaceToImplicit2>(sphere);
    auto collider = std::make_shared<RigidBodyCollider2>(surface);
    solver.setCollider(collider);

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 240; frame.advance()) {
        double t = frame.timeInSeconds();
        sphere->center = Vector2D(domain.midPoint().x, 0.75 - std::cos(t));
        collider->linearVelocity = Vector2D(0, std::sin(t));

        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, ViscousDropVariational) {
    LevelSetLiquidSolver2 solver;
    solver.setViscosityCoefficient(1.0);
    solver.setPressureSolver(
        std::make_shared<GridFractionalSinglePhasePressureSolver2>());

    auto data = solver.gridSystemData();
    double dx = 1.0 / 50.0;
    data->resize(Size2(50, 100), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(std::make_shared<Plane2>(
        Vector2D(0, 1), Vector2D(0, 0.5)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(50, 100);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropWithoutGlobalComp) {
    LevelSetLiquidSolver2 solver;
    solver.setIsGlobalCompensationEnabled(false);

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropWithGlobalComp) {
    LevelSetLiquidSolver2 solver;
    solver.setIsGlobalCompensationEnabled(true);

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Source setting
    BoundingBox2D domain = data->boundingBox();
    ImplicitSurfaceSet2 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&](const Vector2D& x) {
        return surfaceSet.signedDistance(x);
    });

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) / dx);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F
