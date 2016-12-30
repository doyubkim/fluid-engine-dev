// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/fdm_iccg_solver2.h>
#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/grid_single_phase_pressure_solver2.h>
#include <jet/level_set_utils.h>

using namespace jet;

const char kFluid = 0;
const char kAir = 1;
const char kBoundary = 2;

const double kDefaultTolerance = 1e-6;

GridSinglePhasePressureSolver2::GridSinglePhasePressureSolver2() {
    _systemSolver = std::make_shared<FdmIccgSolver2>(100, kDefaultTolerance);
}

GridSinglePhasePressureSolver2::~GridSinglePhasePressureSolver2() {
}

void GridSinglePhasePressureSolver2::solve(
    const FaceCenteredGrid2& input,
    double timeIntervalInSeconds,
    FaceCenteredGrid2* output,
    const ScalarField2& boundarySdf,
    const VectorField2& boundaryVelocity,
    const ScalarField2& fluidSdf) {
    UNUSED_VARIABLE(timeIntervalInSeconds);
    UNUSED_VARIABLE(boundaryVelocity);

    auto pos = input.cellCenterPosition();
    buildMarkers(
        input.resolution(),
        pos,
        boundarySdf,
        fluidSdf);
    buildSystem(input);

    if (_systemSolver != nullptr) {
        // Solve the system
        _systemSolver->solve(&_system);

        // Apply pressure gradient
        applyPressureGradient(input, output);
    }
}

GridBoundaryConditionSolver2Ptr
GridSinglePhasePressureSolver2::suggestedBoundaryConditionSolver() const {
    return std::make_shared<GridBlockedBoundaryConditionSolver2>();
}

void GridSinglePhasePressureSolver2::setLinearSystemSolver(
    const FdmLinearSystemSolver2Ptr& solver) {
    _systemSolver = solver;
}

const FdmVector2& GridSinglePhasePressureSolver2::pressure() const {
    return _system.x;
}

void GridSinglePhasePressureSolver2::buildMarkers(
    const Size2& size,
    const std::function<Vector2D(size_t, size_t)>& pos,
    const ScalarField2& boundarySdf,
    const ScalarField2& fluidSdf) {
    _markers.resize(size);
    _markers.parallelForEachIndex([&](size_t i, size_t j) {
        Vector2D pt = pos(i, j);
        if (isInsideSdf(boundarySdf.sample(pt))) {
            _markers(i, j) = kBoundary;
        } else if (isInsideSdf(fluidSdf.sample(pt))) {
            _markers(i, j) = kFluid;
        } else {
            _markers(i, j) = kAir;
        }
    });
}

void GridSinglePhasePressureSolver2::buildSystem(
    const FaceCenteredGrid2& input) {
    Size2 size = input.resolution();
    _system.A.resize(size);
    _system.x.resize(size);
    _system.b.resize(size);

    Vector2D invH = 1.0 / input.gridSpacing();
    Vector2D invHSqr = invH * invH;

    // Build linear system
    _system.A.parallelForEachIndex([&](size_t i, size_t j) {
        auto& row = _system.A(i, j);

        // initialize
        row.center = row.right = row.up = 0.0;
        _system.b(i, j) = 0.0;

        if (_markers(i, j) == kFluid) {
            _system.b(i, j) = input.divergenceAtCellCenter(i, j);

            if (i + 1 < size.x && _markers(i + 1, j) != kBoundary) {
                row.center += invHSqr.x;
                if (_markers(i + 1, j) == kFluid) {
                    row.right -= invHSqr.x;
                }
            }

            if (i > 0 && _markers(i - 1, j) != kBoundary) {
                row.center += invHSqr.x;
            }

            if (j + 1 < size.y && _markers(i, j + 1) != kBoundary) {
                row.center += invHSqr.y;
                if (_markers(i, j + 1) == kFluid) {
                    row.up -= invHSqr.y;
                }
            }

            if (j > 0 && _markers(i, j - 1) != kBoundary) {
                row.center += invHSqr.y;
            }
        } else {
            row.center = 1.0;
        }
    });
}

void GridSinglePhasePressureSolver2::applyPressureGradient(
    const FaceCenteredGrid2& input,
    FaceCenteredGrid2* output) {
    Size2 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();

    Vector2D invH = 1.0 / input.gridSpacing();

    _system.x.parallelForEachIndex([&](size_t i, size_t j) {
        if (_markers(i, j) == kFluid) {
            if (i + 1 < size.x && _markers(i + 1, j) != kBoundary) {
                u0(i + 1, j)
                    = u(i + 1, j)
                    + invH.x * (_system.x(i + 1, j) - _system.x(i, j));
            }
            if (j + 1 < size.y && _markers(i, j + 1) != kBoundary) {
                v0(i, j + 1)
                    = v(i, j + 1)
                    + invH.y * (_system.x(i, j + 1) - _system.x(i, j));
            }
        }
    });
}
