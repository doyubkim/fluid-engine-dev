// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/box2.h>
#include <jet/box3.h>
#include <jet/grid_smoke_solver2.h>
#include <jet/grid_smoke_solver3.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver3.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/grid_single_phase_pressure_solver3.h>
#include <jet/implicit_surface_set3.h>
#include <jet/level_set_utils.h>
#include <jet/rigid_body_collider2.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere2.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>
#include <algorithm>

using namespace jet;

JET_TESTS(GridSmokeSolver2);

JET_BEGIN_TEST_F(GridSmokeSolver2, Rising) {
    GridSmokeSolver2 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Source setting
    SurfaceToImplicit2 sourceShape(
        std::make_shared<Box2>(Vector2D(0.4, 0.0), Vector2D(0.6, 0.2)));

    auto den = solver.smokeDensity();
    auto temp = solver.temperature();
    den->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });
    temp->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = (*den)(i, j);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    for (Frame frame; frame.index < 120; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = (*den)(i, j);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index + 1);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, RisingWithCollider) {
    GridSmokeSolver2 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Source setting
    SurfaceToImplicit2 sourceShape(
        std::make_shared<Box2>(Vector2D(0.3, 0.0), Vector2D(0.7, 0.4)));

    auto den = solver.smokeDensity();
    auto temp = solver.temperature();
    den->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });
    temp->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });

    // Collider setting
    BoundingBox2D domain = data->boundingBox();
    SurfaceToImplicit2Ptr sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(domain.midPoint(), 0.1));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(sphere);
    solver.setCollider(collider);

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = (*den)(i, j);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    for (Frame frame; frame.index < 240; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = (*den)(i, j);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index + 1);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, RisingWithColliderVariational) {
    GridSmokeSolver2 solver;
    solver.setPressureSolver(
        std::make_shared<GridFractionalSinglePhasePressureSolver2>());

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Source setting
    SurfaceToImplicit2 sourceShape(
        std::make_shared<Box2>(Vector2D(0.3, 0.0), Vector2D(0.7, 0.4)));

    auto den = solver.smokeDensity();
    auto temp = solver.temperature();
    den->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });
    temp->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });

    // Collider setting
    BoundingBox2D domain = data->boundingBox();
    SurfaceToImplicit2Ptr sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(domain.midPoint(), 0.1));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(sphere);
    solver.setCollider(collider);

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = (*den)(i, j);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    for (Frame frame; frame.index < 240; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = (*den)(i, j);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index + 1);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, RisingWithColliderAndDiffusion) {
    GridSmokeSolver2 solver;

    auto data = solver.gridSystemData();
    double dx = 1.0 / 32.0;
    data->resize(Size2(32, 64), Vector2D(dx, dx), Vector2D());

    // Parameter setting
    solver.setViscosityCoefficient(0.01);
    solver.setSmokeDiffusionCoefficient(0.01);
    solver.setTemperatureDiffusionCoefficient(0.01);

    // Source setting
    SurfaceToImplicit2 sourceShape(
        std::make_shared<Box2>(Vector2D(0.3, 0.0), Vector2D(0.7, 0.4)));

    auto den = solver.smokeDensity();
    auto temp = solver.temperature();
    den->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });
    temp->fill([&](const Vector2D& x) {
        if (sourceShape.signedDistance(x) < 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    });

    // Collider setting
    BoundingBox2D domain = data->boundingBox();
    SurfaceToImplicit2Ptr sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(domain.midPoint(), 0.1));
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(sphere);
    solver.setCollider(collider);

    Array2<double> output(32, 64);
    output.forEachIndex([&](size_t i, size_t j) {
        output(i, j) = (*den)(i, j);
    });

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    for (Frame frame; frame.index < 240; frame.advance()) {
        solver.update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = (*den)(i, j);
        });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index + 1);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F


JET_TESTS(GridSmokeSolver3);

JET_BEGIN_TEST_F(GridSmokeSolver3, RisingWithCollider) {
    size_t resolutionX = 50;
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX / 2);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    GridSmokeSolver3 solver;
    // solver.setPressureSolver(
    //     std::make_shared<GridSinglePhasePressureSolver3>());

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addSurface(
        std::make_shared<Box3>(
            Vector3D(0.4, -1, 0.2), Vector3D(0.6, 0.1, 0.3)));
    auto sourceFunc = [&] (const Vector3D& pt) {
        // Convert SDF to density-like field
        return 1.0 - smearedHeavisideSdf(surfaceSet.signedDistance(pt) / dx);
    };

    solver.smokeDensity()->fill(sourceFunc);
    solver.temperature()->fill(sourceFunc);

    // Collider setting
    auto sphere = std::make_shared<Sphere3>(
        Vector3D(0.5, 0.3, 0.25), 0.13 * domain.width());
    auto collider = std::make_shared<RigidBodyCollider3>(sphere);
    solver.setCollider(collider);

    auto density = solver.smokeDensity();
    auto densityPos = density->dataPosition();
    auto temperature = solver.temperature();
    auto temperaturePos = temperature->dataPosition();

    Array2<double> output(resolution.x, resolution.y);
    output.set(0.0);
    density->forEachDataPointIndex(
        [&] (size_t i, size_t j, size_t k) {
            output(i, j) += (*density)(i, j, k);
        });
    char filename[256];
    snprintf(filename, sizeof(filename), "data.#grid2,0000.npy");
    saveData(output.constAccessor(), filename);

    for (Frame frame; frame.index < 240; frame.advance()) {
        density->parallelForEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                double current = (*density)(i, j, k);
                (*density)(i, j, k)
                    = std::max(current, sourceFunc(densityPos(i, j, k)));
            });
        temperature->parallelForEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                double current = (*temperature)(i, j, k);
                (*temperature)(i, j, k)
                    = std::max(current, sourceFunc(temperaturePos(i, j, k)));
            });

        solver.update(frame);

        output.set(0.0);
        density->forEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                output(i, j) += (*density)(i, j, k);
            });
        snprintf(
            filename,
            sizeof(filename),
            "data.#grid2,%04d.npy",
            frame.index + 1);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F
