// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/box2.h>
#include <jet/level_set_liquid_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/level_set_utils.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>
#include <jet/volume_grid_emitter2.h>

#include <vector>

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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
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

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, DropStopAndGo) {
    Frame frame(0, 1.0 / 60.0);
    Array2<double> output(32, 64);
    std::vector<uint8_t> dump;
    char filename[256];

    {
        // Build solver
        auto solver = LevelSetLiquidSolver2::builder()
            .withResolution({32, 64})
            .withDomainSizeX(1.0)
            .makeShared();

        auto grids = solver->gridSystemData();
        auto domain = grids->boundingBox();
        auto sdf = solver->signedDistanceField();
        double dx = grids->gridSpacing().x;

        // Build emitter
        auto plane = Plane2::builder()
            .withNormal({0, 1})
            .withPoint({0, 0.5})
            .makeShared();

        auto sphere = Sphere2::builder()
            .withCenter(domain.midPoint())
            .withRadius(0.15)
            .makeShared();

        auto surfaceSet = ImplicitSurfaceSet2::builder()
            .withExplicitSurfaces({plane, sphere})
            .makeShared();

        auto emitter = VolumeGridEmitter2::builder()
            .withSourceRegion(surfaceSet)
            .makeShared();

        solver->setEmitter(emitter);
        emitter->addSignedDistanceTarget(solver->signedDistanceField());

        for (; frame.index < 60; ++frame) {
            solver->update(frame);

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

        grids->serialize(&dump);
    }

    {
        // Build solver
        auto solver = LevelSetLiquidSolver2::builder()
            .makeShared();
        solver->setCurrentFrame(frame);

        auto grids = solver->gridSystemData();
        grids->deserialize(dump);

        double dx = grids->gridSpacing().x;
        auto sdf = solver->signedDistanceField();

        for (; frame.index < 120; ++frame) {
            solver->update(frame);

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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 240; frame.advance()) {
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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 240; frame.advance()) {
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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
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

    for (Frame frame(0, 1.0 / 60.0); frame.index < 120; frame.advance()) {
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

JET_BEGIN_TEST_F(LevelSetLiquidSolver2, RisingFloor) {
    // Build solver
    auto solver = LevelSetLiquidSolver2::builder()
        .withResolution({5, 10})
        .withDomainSizeX(1.0)
        .makeShared();
    solver->setGravity({0, 0, 0});

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.0, 0.0})
        .withUpperCorner({1.0, 0.8})
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(box)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addSignedDistanceTarget(solver->signedDistanceField());

    // Build collider
    auto tank = Box2::builder()
        .withLowerCorner({-1, 0})
        .withUpperCorner({2, 2})
        .withIsNormalFlipped(true)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(tank)
        .makeShared();

    collider->setOnBeginUpdateCallback([] (Collider2* col, double t, double) {
        col->surface()->transform.setTranslation({0, t});
        static_cast<RigidBodyCollider2*>(col)->linearVelocity.x = 0.0;
        static_cast<RigidBodyCollider2*>(col)->linearVelocity.y = 1.0;
    });

    solver->setCollider(collider);

    char filename[256];
    Array2<double> output(5, 10);
    Array2<double> div(5, 10);
    auto data = solver->gridSystemData();
    auto sdf = solver->signedDistanceField();

    for (Frame frame(0, 1/100.0); frame.index < 120; ++frame) {
        solver->update(frame);

        output.forEachIndex([&](size_t i, size_t j) {
            output(i, j) = 1.0 - smearedHeavisideSdf((*sdf)(i, j) * 5.0);
        });
        snprintf(
            filename,
            sizeof(filename),
            "output.#grid2,%04d.npy",
            frame.index);
        saveData(output.constAccessor(), filename);
    }
}
JET_END_TEST_F
