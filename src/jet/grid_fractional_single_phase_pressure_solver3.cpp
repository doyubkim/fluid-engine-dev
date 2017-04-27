// Copyright (c) 2016 Doyub Kim

//
// Adopted the code from:
// http://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/
// and
// https://github.com/christopherbatty/FluidRigidCoupling2D
//

#include <pch.h>

#include <jet/constants.h>
#include <jet/fdm_iccg_solver3.h>
#include <jet/grid_fractional_boundary_condition_solver3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <jet/level_set_utils.h>

#include <algorithm>

using namespace jet;

const double kDefaultTolerance = 1e-6;
const double kMinWeight = 0.01;

GridFractionalSinglePhasePressureSolver3::
    GridFractionalSinglePhasePressureSolver3() {
    _systemSolver = std::make_shared<FdmIccgSolver3>(100, kDefaultTolerance);
}

GridFractionalSinglePhasePressureSolver3::
    ~GridFractionalSinglePhasePressureSolver3() {}

void GridFractionalSinglePhasePressureSolver3::solve(
    const FaceCenteredGrid3& input, double timeIntervalInSeconds,
    FaceCenteredGrid3 *output, const ScalarField3& boundarySdf,
    const VectorField3& boundaryVelocity, const ScalarField3& fluidSdf) {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    buildWeights(input, boundarySdf, boundaryVelocity, fluidSdf);
    buildSystem(input);

    if (_systemSolver != nullptr) {
        // Solve the system
        _systemSolver->solve(&_system);

        // Apply pressure gradient
        applyPressureGradient(input, output);
    }
}

GridBoundaryConditionSolver3Ptr
GridFractionalSinglePhasePressureSolver3::suggestedBoundaryConditionSolver()
    const {
    return std::make_shared<GridFractionalBoundaryConditionSolver3>();
}

void GridFractionalSinglePhasePressureSolver3::setLinearSystemSolver(
    const FdmLinearSystemSolver3Ptr& solver) {
    _systemSolver = solver;
}

const FdmVector3& GridFractionalSinglePhasePressureSolver3::pressure() const {
    return _system.x;
}

void GridFractionalSinglePhasePressureSolver3::buildWeights(
    const FaceCenteredGrid3& input, const ScalarField3& boundarySdf,
    const VectorField3& boundaryVelocity, const ScalarField3& fluidSdf) {
    Size3 uSize = input.uSize();
    Size3 vSize = input.vSize();
    Size3 wSize = input.wSize();
    auto uPos = input.uPosition();
    auto vPos = input.vPosition();
    auto wPos = input.wPosition();
    _uWeights.resize(uSize);
    _vWeights.resize(vSize);
    _wWeights.resize(wSize);
    _uBoundary.resize(uSize);
    _vBoundary.resize(vSize);
    _wBoundary.resize(wSize);
    _fluidSdf.resize(input.resolution(), input.gridSpacing(), input.origin());

    _fluidSdf.fill([&](const Vector3D& x) {
        return fluidSdf.sample(x);
    });

    Vector3D h = input.gridSpacing();

    _uWeights.forEachIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = uPos(i, j, k);
        double phi0 =
            boundarySdf.sample(pt + Vector3D(0.0, -0.5 * h.y, -0.5 * h.z));
        double phi1 =
            boundarySdf.sample(pt + Vector3D(0.0, 0.5 * h.y, -0.5 * h.z));
        double phi2 =
            boundarySdf.sample(pt + Vector3D(0.0, -0.5 * h.y, 0.5 * h.z));
        double phi3 =
            boundarySdf.sample(pt + Vector3D(0.0, 0.5 * h.y, 0.5 * h.z));
        double frac = fractionInside(phi0, phi1, phi2, phi3);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _uWeights(i, j, k) = weight;
        _uBoundary(i, j, k) = boundaryVelocity.sample(pt).x;
    });

    _vWeights.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = vPos(i, j, k);
        double phi0 =
            boundarySdf.sample(pt + Vector3D(-0.5 * h.x, 0.0, -0.5 * h.z));
        double phi1 =
            boundarySdf.sample(pt + Vector3D(-0.5 * h.x, 0.0, 0.5 * h.z));
        double phi2 =
            boundarySdf.sample(pt + Vector3D(0.5 * h.x, 0.0, -0.5 * h.z));
        double phi3 =
            boundarySdf.sample(pt + Vector3D(0.5 * h.x, 0.0, 0.5 * h.z));
        double frac = fractionInside(phi0, phi1, phi2, phi3);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _vWeights(i, j, k) = weight;
        _vBoundary(i, j, k) = boundaryVelocity.sample(pt).y;
    });

    _wWeights.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        Vector3D pt = wPos(i, j, k);
        double phi0 =
            boundarySdf.sample(pt + Vector3D(-0.5 * h.x, -0.5 * h.y, 0.0));
        double phi1 =
            boundarySdf.sample(pt + Vector3D(-0.5 * h.x, 0.5 * h.y, 0.0));
        double phi2 =
            boundarySdf.sample(pt + Vector3D(0.5 * h.x, -0.5 * h.y, 0.0));
        double phi3 =
            boundarySdf.sample(pt + Vector3D(0.5 * h.x, 0.5 * h.y, 0.0));
        double frac = fractionInside(phi0, phi1, phi2, phi3);
        double weight = clamp(1.0 - frac, 0.0, 1.0);

        // Clamp non-zero weight to kMinWeight. Having nearly-zero element
        // in the matrix can be an issue.
        if (weight < kMinWeight && weight > 0.0) {
            weight = kMinWeight;
        }

        _wWeights(i, j, k) = weight;
        _wBoundary(i, j, k) = boundaryVelocity.sample(pt).z;
    });
}

void GridFractionalSinglePhasePressureSolver3::buildSystem(
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

        double centerPhi = _fluidSdf(i, j, k);

        if (isInsideSdf(centerPhi)) {
            double term;

            if (i + 1 < size.x) {
                term = _uWeights(i + 1, j, k) * invHSqr.x;
                double rightPhi = _fluidSdf(i + 1, j, k);
                if (isInsideSdf(rightPhi)) {
                    row.center += term;
                    row.right -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, rightPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j, k) +=
                    _uWeights(i + 1, j, k) * input.u(i + 1, j, k) * invH.x;
            } else {
                _system.b(i, j, k) += input.u(i + 1, j, k) * invH.x;
            }

            if (i > 0) {
                term = _uWeights(i, j, k) * invHSqr.x;
                double leftPhi = _fluidSdf(i - 1, j, k);
                if (isInsideSdf(leftPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, leftPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j, k) -=
                    _uWeights(i, j, k) * input.u(i, j, k) * invH.x;
            } else {
                _system.b(i, j, k) -= input.u(i, j, k) * invH.x;
            }

            if (j + 1 < size.y) {
                term = _vWeights(i, j + 1, k) * invHSqr.y;
                double upPhi = _fluidSdf(i, j + 1, k);
                if (isInsideSdf(upPhi)) {
                    row.center += term;
                    row.up -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, upPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j, k) +=
                    _vWeights(i, j + 1, k) * input.v(i, j + 1, k) * invH.y;
            } else {
                _system.b(i, j, k) += input.v(i, j + 1, k) * invH.y;
            }

            if (j > 0) {
                term = _vWeights(i, j, k) * invHSqr.y;
                double downPhi = _fluidSdf(i, j - 1, k);
                if (isInsideSdf(downPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, downPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j, k) -=
                    _vWeights(i, j, k) * input.v(i, j, k) * invH.y;
            } else {
                _system.b(i, j, k) -= input.v(i, j, k) * invH.y;
            }

            if (k + 1 < size.z) {
                term = _wWeights(i, j, k + 1) * invHSqr.z;
                double frontPhi = _fluidSdf(i, j, k + 1);
                if (isInsideSdf(frontPhi)) {
                    row.center += term;
                    row.front -= term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, frontPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j, k) +=
                    _wWeights(i, j, k + 1) * input.w(i, j, k + 1) * invH.z;
            } else {
                _system.b(i, j, k) += input.w(i, j, k + 1) * invH.z;
            }

            if (k > 0) {
                term = _wWeights(i, j, k) * invHSqr.z;
                double backPhi = _fluidSdf(i, j, k - 1);
                if (isInsideSdf(backPhi)) {
                    row.center += term;
                } else {
                    double theta = fractionInsideSdf(centerPhi, backPhi);
                    theta = std::max(theta, 0.01);
                    row.center += term / theta;
                }
                _system.b(i, j, k) -=
                    _wWeights(i, j, k) * input.w(i, j, k) * invH.z;
            } else {
                _system.b(i, j, k) -= input.w(i, j, k) * invH.z;
            }

            // Accumulate contributions from the moving boundary
            double boundaryContribution
             = (1.0 - _uWeights(i + 1, j, k)) * _uBoundary(i + 1, j, k) * invH.x
             - (1.0 - _uWeights(i, j, k)) * _uBoundary(i, j, k) * invH.x
             + (1.0 - _vWeights(i, j + 1, k)) * _vBoundary(i, j + 1, k) * invH.y
             - (1.0 - _vWeights(i, j, k)) * _vBoundary(i, j, k) * invH.y
             + (1.0 - _wWeights(i, j, k + 1)) * _wBoundary(i, j, k + 1) * invH.z
             - (1.0 - _wWeights(i, j, k)) * _wBoundary(i, j, k) * invH.z;
            _system.b(i, j, k) += boundaryContribution;
        } else {
            row.center = 1.0;
        }
    });
}

void GridFractionalSinglePhasePressureSolver3::applyPressureGradient(
    const FaceCenteredGrid3& input, FaceCenteredGrid3 *output) {
    Size3 size = input.resolution();
    auto u = input.uConstAccessor();
    auto v = input.vConstAccessor();
    auto w = input.wConstAccessor();
    auto u0 = output->uAccessor();
    auto v0 = output->vAccessor();
    auto w0 = output->wAccessor();

    Vector3D invH = 1.0 / input.gridSpacing();

    _system.x.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        double centerPhi = _fluidSdf(i, j, k);

        if (i + 1 < size.x && _uWeights(i + 1, j, k) > 0.0 &&
            (isInsideSdf(centerPhi) || isInsideSdf(_fluidSdf(i + 1, j, k)))) {
            double rightPhi = _fluidSdf(i + 1, j, k);
            double theta = fractionInsideSdf(centerPhi, rightPhi);
            theta = std::max(theta, 0.01);

            u0(i + 1, j, k) =
                u(i + 1, j, k) +
                invH.x / theta * (_system.x(i + 1, j, k) - _system.x(i, j, k));
        }

        if (j + 1 < size.y && _vWeights(i, j + 1, k) > 0.0 &&
            (isInsideSdf(centerPhi) || isInsideSdf(_fluidSdf(i, j + 1, k)))) {
            double upPhi = _fluidSdf(i, j + 1, k);
            double theta = fractionInsideSdf(centerPhi, upPhi);
            theta = std::max(theta, 0.01);

            v0(i, j + 1, k) =
                v(i, j + 1, k) +
                invH.y / theta * (_system.x(i, j + 1, k) - _system.x(i, j, k));
        }

        if (k + 1 < size.z && _wWeights(i, j, k + 1) > 0.0 &&
            (isInsideSdf(centerPhi) || isInsideSdf(_fluidSdf(i, j, k + 1)))) {
            double frontPhi = _fluidSdf(i, j, k + 1);
            double theta = fractionInsideSdf(centerPhi, frontPhi);
            theta = std::max(theta, 0.01);

            w0(i, j, k + 1) =
                w(i, j, k + 1) +
                invH.z / theta * (_system.x(i, j, k + 1) - _system.x(i, j, k));
        }
    });
}
