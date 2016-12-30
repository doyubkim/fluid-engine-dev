// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/fdm_iccg_solver3.h>
#include <jet/grid_blocked_boundary_condition_solver3.h>
#include <jet/grid_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>

using namespace jet;

const char kFluid = 0;
const char kAir = 1;
const char kBoundary = 2;

const double kDefaultTolerance = 1e-6;

GridSinglePhasePressureSolver3::GridSinglePhasePressureSolver3() {
    _systemSolver = std::make_shared<FdmIccgSolver3>(100, kDefaultTolerance);
}

GridSinglePhasePressureSolver3::~GridSinglePhasePressureSolver3() {
}

void GridSinglePhasePressureSolver3::solve(
    const FaceCenteredGrid3& input,
    double timeIntervalInSeconds,
    FaceCenteredGrid3* output,
    const ScalarField3& boundarySdf,
    const VectorField3& boundaryVelocity,
    const ScalarField3& fluidSdf) {
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

GridBoundaryConditionSolver3Ptr
GridSinglePhasePressureSolver3::suggestedBoundaryConditionSolver() const {
    return std::make_shared<GridBlockedBoundaryConditionSolver3>();
}

void GridSinglePhasePressureSolver3::setLinearSystemSolver(
    const FdmLinearSystemSolver3Ptr& solver) {
    _systemSolver = solver;
}

const FdmVector3& GridSinglePhasePressureSolver3::pressure() const {
    return _system.x;
}

void GridSinglePhasePressureSolver3::buildMarkers(
    const Size3& size,
    const std::function<Vector3D(size_t, size_t, size_t)>& pos,
    const ScalarField3& boundarySdf,
    const ScalarField3& fluidSdf) {
    _markers.resize(size);
    _markers.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = pos(i, j, k);
        if (isInsideSdf(boundarySdf.sample(pt))) {
            _markers(i, j, k) = kBoundary;
        } else if (isInsideSdf(fluidSdf.sample(pt))) {
            _markers(i, j, k) = kFluid;
        } else {
            _markers(i, j, k) = kAir;
        }
    });
}

void GridSinglePhasePressureSolver3::buildSystem(
    const FaceCenteredGrid3& input) {
    Size3 size = input.resolution();
    _system.A.resize(size);
    _system.x.resize(size);
    _system.b.resize(size);

    Vector3D invH = 1.0 / input.gridSpacing();
    Vector3D invHSqr = invH * invH;

    // Build linear system
    _system.A.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        auto& row = _system.A(i, j, k);

        // initialize
        row.center = row.right = row.up = row.front = 0.0;
        _system.b(i, j, k) = 0.0;

        if (_markers(i, j, k) == kFluid) {
            _system.b(i, j, k) = input.divergenceAtCellCenter(i, j, k);

            if (i + 1 < size.x && _markers(i + 1, j, k) != kBoundary) {
                row.center += invHSqr.x;
                if (_markers(i + 1, j, k) == kFluid) {
                    row.right -= invHSqr.x;
                }
            }

            if (i > 0 && _markers(i - 1, j, k) != kBoundary) {
                row.center += invHSqr.x;
            }

            if (j + 1 < size.y && _markers(i, j + 1, k) != kBoundary) {
                row.center += invHSqr.y;
                if (_markers(i, j + 1, k) == kFluid) {
                    row.up -= invHSqr.y;
                }
            }

            if (j > 0 && _markers(i, j - 1, k) != kBoundary) {
                row.center += invHSqr.y;
            }

            if (k + 1 < size.z && _markers(i, j, k + 1) != kBoundary) {
                row.center += invHSqr.z;
                if (_markers(i, j, k + 1) == kFluid) {
                    row.front -= invHSqr.z;
                }
            }

            if (k > 0 && _markers(i, j, k - 1) != kBoundary) {
                row.center += invHSqr.z;
            }
        } else {
            row.center = 1.0;
        }
    });
}

void GridSinglePhasePressureSolver3::applyPressureGradient(
    const FaceCenteredGrid3& input,
    FaceCenteredGrid3* output) {
    Size3 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto w = input.wConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();
    auto w0 = output->wAccessor();

    Vector3D invH = 1.0 / input.gridSpacing();

    _system.x.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (_markers(i, j, k) == kFluid) {
            if (i + 1 < size.x && _markers(i + 1, j, k) != kBoundary) {
                u0(i + 1, j, k)
                    = u(i + 1, j, k)
                    + invH.x
                    * (_system.x(i + 1, j, k) - _system.x(i, j, k));
            }
            if (j + 1 < size.y && _markers(i, j + 1, k) != kBoundary) {
                v0(i, j + 1, k)
                    = v(i, j + 1, k)
                    + invH.y
                    * (_system.x(i, j + 1, k) - _system.x(i, j, k));
            }
            if (k + 1 < size.z && _markers(i, j, k + 1) != kBoundary) {
                w0(i, j, k + 1)
                    = w(i, j, k + 1)
                    + invH.z
                    * (_system.x(i, j, k + 1) - _system.x(i, j, k));
            }
        }
    });
}
