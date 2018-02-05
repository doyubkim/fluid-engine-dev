// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/box2.h>
#include <jet/box3.h>
#include <jet/cubic_semi_lagrangian3.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver3.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/grid_single_phase_pressure_solver2.h>
#include <jet/grid_smoke_solver2.h>
#include <jet/grid_smoke_solver3.h>
#include <jet/implicit_surface_set3.h>
#include <jet/level_set_utils.h>
#include <jet/rigid_body_collider2.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere2.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit2.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_grid_emitter2.h>
#include <jet/volume_grid_emitter3.h>
#include <algorithm>

using namespace jet;

JET_TESTS(GridSmokeSolver2);

JET_BEGIN_TEST_F(GridSmokeSolver2, Rising) {
    // Build solver
    auto solver = GridSmokeSolver2::builder()
        .withResolution({32, 64})
        .withGridSpacing(1.0 / 32.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.3, 0.0})
        .withUpperCorner({0.7, 0.4})
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(box)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(solver->temperature(), 0.0, 1.0);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveData(solver->smokeDensity()->constDataAccessor(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, RisingWithCollider) {
    // Build solver
    auto solver = GridSmokeSolver2::builder()
        .withResolution({32, 64})
        .withGridSpacing(1.0 / 32.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.3, 0.0})
        .withUpperCorner({0.7, 0.4})
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(box)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(solver->temperature(), 0.0, 1.0);

    // Build collider
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 1.0})
        .withRadius(0.1)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveData(solver->smokeDensity()->constDataAccessor(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, MovingEmitterWithCollider) {
    // Build solver
    auto solver = GridSmokeSolver2::builder()
        .withResolution({32, 64})
        .withGridSpacing(1.0 / 32.0)
        .makeShared();

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.3, 0.0})
        .withUpperCorner({0.7, 0.1})
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(box)
        .withIsOneShot(false)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(solver->temperature(), 0.0, 1.0);
    emitter->setOnBeginUpdateCallback(
        [&box] (double t, double dt) {
            box->bound.lowerCorner.x = 0.1 * std::sin(kPiD * t) + 0.3;
            box->bound.upperCorner.x = 0.1 * std::sin(kPiD * t) + 0.7;
        });

    // Build collider
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 1.0})
        .withRadius(0.1)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveData(solver->smokeDensity()->constDataAccessor(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, RisingWithColliderNonVariational) {
    // Build solver
    auto solver = GridSmokeSolver2::builder()
        .withResolution({32, 64})
        .withGridSpacing(1.0 / 32.0)
        .makeShared();

    solver->setPressureSolver(
        std::make_shared<GridSinglePhasePressureSolver2>());

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.3, 0.0})
        .withUpperCorner({0.7, 0.4})
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(box)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(solver->temperature(), 0.0, 1.0);

    // Build collider
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 1.0})
        .withRadius(0.1)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveData(solver->smokeDensity()->constDataAccessor(), frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridSmokeSolver2, RisingWithColliderAndDiffusion) {
    // Build solver
    auto solver = GridSmokeSolver2::builder()
        .withResolution({32, 64})
        .withGridSpacing(1.0 / 32.0)
        .makeShared();

    // Parameter setting
    solver->setViscosityCoefficient(0.01);
    solver->setSmokeDiffusionCoefficient(0.01);
    solver->setTemperatureDiffusionCoefficient(0.01);

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.3, 0.0})
        .withUpperCorner({0.7, 0.4})
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(box)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0.0, 1.0);
    emitter->addStepFunctionTarget(solver->temperature(), 0.0, 1.0);

    // Build collider
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 1.0})
        .withRadius(0.1)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    for (Frame frame; frame.index < 240; ++frame) {
        solver->update(frame);

        saveData(solver->smokeDensity()->constDataAccessor(), frame.index);
    }
}
JET_END_TEST_F


JET_TESTS(GridSmokeSolver3);

JET_BEGIN_TEST_F(GridSmokeSolver3, Rising) {
    //
    // This is a replica of smoke_sim example 4
    //

    size_t resolutionX = 50;

    // Build solver
    auto solver = GridSmokeSolver3::builder()
        .withResolution({resolutionX, 6 * resolutionX / 5, resolutionX / 2})
        .withDomainSizeX(1.0)
        .makeShared();

    solver->setBuoyancyTemperatureFactor(2.0);

    // Build emitter
    auto box = Box3::builder()
        .withLowerCorner({0.05, 0.1, 0.225})
        .withUpperCorner({0.1, 0.15, 0.275})
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(box)
        .withIsOneShot(false)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);
    emitter->addTarget(
        solver->velocity(),
        [](double sdf, const Vector3D& pt, const Vector3D& oldVal) {
            if (sdf < 0.05) {
                return Vector3D(0.5, oldVal.y, oldVal.z);
            } else {
                return Vector3D(oldVal);
            }
        });

    auto grids = solver->gridSystemData();
    Size3 resolution = grids->resolution();
    Array2<double> output(resolution.x, resolution.y);
    auto density = solver->smokeDensity();
    char filename[256];

    for (Frame frame(0, 1.0 / 60.0); frame.index < 240; ++frame) {
        solver->update(frame);

        output.set(0.0);
        density->forEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                output(i, j) += (*density)(i, j, k);
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

JET_BEGIN_TEST_F(GridSmokeSolver3, RisingWithCollider) {
    //
    // This is a replica of smoke_sim example 1.
    //

    size_t resolutionX = 50;

    // Build solver
    auto solver = GridSmokeSolver3::builder()
        .withResolution({resolutionX, 2 * resolutionX, resolutionX})
        .withDomainSizeX(1.0)
        .makeShared();

    solver->setAdvectionSolver(std::make_shared<CubicSemiLagrangian3>());

    auto grids = solver->gridSystemData();
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto box = Box3::builder()
        .withLowerCorner({0.45, -1, 0.45})
        .withUpperCorner({0.55, 0.05, 0.55})
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(box)
        .withIsOneShot(false)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);

    // Build collider
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.3, 0.5})
        .withRadius(0.075 * domain.width())
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    Size3 resolution = grids->resolution();
    Array2<double> output(resolution.x, resolution.y);
    auto density = solver->smokeDensity();
    char filename[256];

    for (Frame frame(0, 1.0 / 60.0); frame.index < 240; ++frame) {
        solver->update(frame);

        output.set(0.0);
        density->forEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                output(i, j) += (*density)(i, j, k);
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

JET_BEGIN_TEST_F(GridSmokeSolver3, RisingWithColliderLinear) {
    //
    // This is a replica of smoke_sim example 2.
    //

    size_t resolutionX = 50;

    // Build solver
    auto solver = GridSmokeSolver3::builder()
        .withResolution({resolutionX, 2 * resolutionX, resolutionX})
        .withDomainSizeX(1.0)
        .makeShared();

    solver->setAdvectionSolver(std::make_shared<SemiLagrangian3>());

    auto grids = solver->gridSystemData();
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto box = Box3::builder()
        .withLowerCorner({0.45, -1, 0.45})
        .withUpperCorner({0.55, 0.05, 0.55})
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(box)
        .withIsOneShot(false)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);

    // Build collider
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.3, 0.5})
        .withRadius(0.075 * domain.width())
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(sphere)
        .makeShared();

    solver->setCollider(collider);

    Size3 resolution = grids->resolution();
    Array2<double> output(resolution.x, resolution.y);
    auto density = solver->smokeDensity();
    char filename[256];

    for (Frame frame(0, 1.0 / 60.0); frame.index < 240; ++frame) {
        solver->update(frame);

        output.set(0.0);
        density->forEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                output(i, j) += (*density)(i, j, k);
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
