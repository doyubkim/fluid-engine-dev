// Copyright (c) 2016 Doyub Kim

//
// Adopted the code from:
// http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/
// and
// https://github.com/christopherbatty/FluidRigidCoupling2D
//

#include <pch.h>
#include <jet/constants.h>
#include <jet/fdm_iccg_solver2.h>
#include <jet/grid_fractional_boundary_condition_solver2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <jet/level_set_utils.h>
#include <algorithm>

using namespace jet;

const double kDefaultTolerance = 1e-6;
const double kMinWeight = 0.01;

GridFractionalSinglePhasePressureSolver2
::GridFractionalSinglePhasePressureSolver2() {
    _systemSolver = std::make_shared<FdmIccgSolver2>(100, kDefaultTolerance);
}

GridFractionalSinglePhasePressureSolver2
::~GridFractionalSinglePhasePressureSolver2() {
}

void GridFractionalSinglePhasePressureSolver2::solve(
    const FaceCenteredGrid2& input,
    double timeIntervalInSeconds,
    FaceCenteredGrid2* output,
    const ScalarField2& boundarySdf,
    const VectorField2& boundaryVelocity,
    const ScalarField2& fluidSdf) {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    buildWeights(
        input,
        boundarySdf,
        boundaryVelocity,
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
GridFractionalSinglePhasePressureSolver2
::suggestedBoundaryConditionSolver() const {
    return std::make_shared<GridFractionalBoundaryConditionSolver2>();
}

void GridFractionalSinglePhasePressureSolver2::setLinearSystemSolver(
    const FdmLinearSystemSolver2Ptr& solver) {
    _systemSolver = solver;
}

const FdmVector2& GridFractionalSinglePhasePressureSolver2::pressure() const {
    return _system.x;
}

void GridFractionalSinglePhasePressureSolver2::buildWeights(
    const FaceCenteredGrid2& input,
    const ScalarField2& boundarySdf,
    const VectorField2& boundaryVelocity,
    const ScalarField2& fluidSdf) {
    Size2 uSize = input.uSize();
    Size2 vSize = input.vSize();
    auto uPos = input.uPosition();
    auto vPos = input.vPosition();
    _uWeights.resize(uSize);
    _vWeights.resize(vSize);
    _uBoundary.resize(uSize);
    _vBoundary.resize(vSize);
    _fluidSdf.resize(input.resolution(), input.gridSpacing(), input.origin());

    _fluidSdf.fill([&](const Vector2D& x) {
        return fluidSdf.sample(x);
    });

    Vector2D h = input.gridSpacing();

    _uWeights.parallelForEachIndex([&](size_t i, size_t j) {
        Vector2D pt = uPos(i, j);
        double phi0 = boundarySdf.sample(pt - Vector2D(0.5 * h.x, 0.0));
        double phi1 = boundarySdf.sample(pt + Vector2D(0.5 * h.x, 0.0));
        double frac = fractionInsideSdf(phi0, phi1);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _uWeights(i, j) = weight;
        _uBoundary(i, j) = boundaryVelocity.sample(pt).x;
    });

    _vWeights.parallelForEachIndex([&](size_t i, size_t j) {
        Vector2D pt = vPos(i, j);
        double phi0 = boundarySdf.sample(pt - Vector2D(0.0, 0.5 * h.y));
        double phi1 = boundarySdf.sample(pt + Vector2D(0.0, 0.5 * h.y));
        double frac = fractionInsideSdf(phi0, phi1);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _vWeights(i, j) = weight;
        _vBoundary(i, j) = boundaryVelocity.sample(pt).y;
    });
}

void GridFractionalSinglePhasePressureSolver2::buildSystem(
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

        double centerPhi = _fluidSdf(i, j);

        if (isInsideSdf(centerPhi)) {
            double term;

            if (i + 1 < size.x) {
                term = _uWeights(i + 1, j) * invHSqr.x;
                double rightPhi = _fluidSdf(i + 1, j);
                if (isInsideSdf(rightPhi)) {
                    row.center += term;
                    row.right -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, rightPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j)
                    += _uWeights(i + 1, j) * input.u(i + 1, j) * invH.x;
            } else {
                _system.b(i, j) += input.u(i + 1, j) * invH.x;
            }



            if (i > 0) {
                term = _uWeights(i, j) * invHSqr.x;
                double leftPhi = _fluidSdf(i - 1, j);
                if (isInsideSdf(leftPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, leftPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j)
                    -= _uWeights(i, j) * input.u(i, j) * invH.x;
            } else {
                _system.b(i, j) -= input.u(i, j) * invH.x;
            }

            if (j + 1 < size.y) {
                term = _vWeights(i, j + 1) * invHSqr.y;
                double upPhi = _fluidSdf(i, j + 1);
                if (isInsideSdf(upPhi)) {
                    row.center += term;
                    row.up -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, upPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j)
                    += _vWeights(i, j + 1) * input.v(i, j + 1) * invH.y;
            } else {
                _system.b(i, j) += input.v(i, j + 1) * invH.y;
            }

            if (j > 0) {
                term = _vWeights(i, j) * invHSqr.y;
                double downPhi = _fluidSdf(i, j - 1);
                if (isInsideSdf(downPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, downPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j)
                    -= _vWeights(i, j) * input.v(i, j) * invH.y;
            } else {
                _system.b(i, j) -= input.v(i, j) * invH.y;
            }

            // Accumulate contributions from the moving boundary
            double boundaryContribution
                = (1.0 - _uWeights(i + 1, j)) * _uBoundary(i + 1, j) * invH.x
                - (1.0 - _uWeights(i, j)) * _uBoundary(i, j) * invH.x
                + (1.0 - _vWeights(i, j + 1)) * _vBoundary(i, j + 1) * invH.y
                - (1.0 - _vWeights(i, j)) * _vBoundary(i, j) * invH.y;
            _system.b(i, j) += boundaryContribution;
        } else {
            row.center = 1.0;
        }
    });
}

void GridFractionalSinglePhasePressureSolver2::applyPressureGradient(
    const FaceCenteredGrid2& input,
    FaceCenteredGrid2* output) {
    Size2 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();

    Vector2D invH = 1.0 / input.gridSpacing();

    _system.x.parallelForEachIndex([&](size_t i, size_t j) {
        double centerPhi = _fluidSdf(i, j);

        if (i + 1 < size.x
            && _uWeights(i + 1, j) > 0.0
            && (isInsideSdf(centerPhi)
                || isInsideSdf(_fluidSdf(i + 1, j)))) {
            double rightPhi = _fluidSdf(i + 1, j);
            double theta = fractionInsideSdf(centerPhi, rightPhi);
            theta = std::max(theta, 0.01);

            u0(i + 1, j)
                = u(i + 1, j)
                + invH.x / theta * (_system.x(i + 1, j) - _system.x(i, j));
        }

        if (j + 1 < size.y
            && _vWeights(i, j + 1) > 0.0
            && (isInsideSdf(centerPhi)
                || isInsideSdf(_fluidSdf(i, j + 1)))) {
            double upPhi = _fluidSdf(i, j + 1);
            double theta = fractionInsideSdf(centerPhi, upPhi);
            theta = std::max(theta, 0.01);

            v0(i, j + 1)
                = v(i, j + 1)
                + invH.y / theta * (_system.x(i, j + 1) - _system.x(i, j));
        }
    });
}
