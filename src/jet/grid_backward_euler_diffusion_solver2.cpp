// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/constants.h>
#include <jet/fdm_iccg_solver2.h>
#include <jet/fdm_utils.h>
#include <jet/grid_backward_euler_diffusion_solver2.h>
#include <jet/level_set_utils.h>
#include <pch.h>

using namespace jet;

const char kFluid = 0;
const char kAir = 1;
const char kBoundary = 2;

GridBackwardEulerDiffusionSolver2::GridBackwardEulerDiffusionSolver2(
    BoundaryType boundaryType)
    : _boundaryType(boundaryType) {
    _systemSolver = std::make_shared<FdmIccgSolver2>(100, kEpsilonD);
}

void GridBackwardEulerDiffusionSolver2::solve(const ScalarGrid2& source,
                                              double diffusionCoefficient,
                                              double timeIntervalInSeconds,
                                              ScalarGrid2* dest,
                                              const ScalarField2& boundarySdf,
                                              const ScalarField2& fluidSdf) {
    if (_systemSolver != nullptr) {
        auto pos = source.dataPosition();
        Vector2D h = source.gridSpacing();
        Vector2D c = timeIntervalInSeconds * diffusionCoefficient / elemMul(h, h);

        buildMarkers(source.dataSize(), pos, boundarySdf, fluidSdf);
        buildMatrix(source.dataSize(), c);
        buildVectors(source.dataView(), c);

        // Solve the system
        _systemSolver->solve(&_system);

        // Assign the solution
        source.parallelForEachDataPointIndex(
            [&](size_t i, size_t j) { (*dest)(i, j) = _system.x(i, j); });
    }
}

void GridBackwardEulerDiffusionSolver2::solve(
    const CollocatedVectorGrid2& source, double diffusionCoefficient,
    double timeIntervalInSeconds, CollocatedVectorGrid2* dest,
    const ScalarField2& boundarySdf, const ScalarField2& fluidSdf) {
    if (_systemSolver != nullptr) {
        auto pos = source.dataPosition();
        Vector2D h = source.gridSpacing();
        Vector2D c = timeIntervalInSeconds * diffusionCoefficient / elemMul(h, h);

        buildMarkers(source.dataSize(), pos, boundarySdf, fluidSdf);
        buildMatrix(source.dataSize(), c);

        // u
        buildVectors(source.dataView(), c, 0);

        // Solve the system
        _systemSolver->solve(&_system);

        // Assign the solution
        source.parallelForEachDataPointIndex(
            [&](size_t i, size_t j) { (*dest)(i, j).x = _system.x(i, j); });

        // v
        buildVectors(source.dataView(), c, 1);

        // Solve the system
        _systemSolver->solve(&_system);

        // Assign the solution
        source.parallelForEachDataPointIndex(
            [&](size_t i, size_t j) { (*dest)(i, j).y = _system.x(i, j); });
    }
}

void GridBackwardEulerDiffusionSolver2::solve(const FaceCenteredGrid2& source,
                                              double diffusionCoefficient,
                                              double timeIntervalInSeconds,
                                              FaceCenteredGrid2* dest,
                                              const ScalarField2& boundarySdf,
                                              const ScalarField2& fluidSdf) {
    if (_systemSolver != nullptr) {
        Vector2D h = source.gridSpacing();
        Vector2D c = timeIntervalInSeconds * diffusionCoefficient / elemMul(h, h);

        // u
        auto uPos = source.uPosition();
        buildMarkers(source.uSize(), uPos, boundarySdf, fluidSdf);
        buildMatrix(source.uSize(), c);
        buildVectors(source.uView(), c);

        // Solve the system
        _systemSolver->solve(&_system);

        // Assign the solution
        source.parallelForEachUIndex(
            [&](size_t i, size_t j) { dest->u(i, j) = _system.x(i, j); });

        // v
        auto vPos = source.vPosition();
        buildMarkers(source.vSize(), vPos, boundarySdf, fluidSdf);
        buildMatrix(source.vSize(), c);
        buildVectors(source.vView(), c);

        // Solve the system
        _systemSolver->solve(&_system);

        // Assign the solution
        source.parallelForEachVIndex(
            [&](size_t i, size_t j) { dest->v(i, j) = _system.x(i, j); });
    }
}

void GridBackwardEulerDiffusionSolver2::setLinearSystemSolver(
    const FdmLinearSystemSolver2Ptr& solver) {
    _systemSolver = solver;
}

void GridBackwardEulerDiffusionSolver2::buildMarkers(
    const Vector2UZ& size, const std::function<Vector2D(size_t, size_t)>& pos,
    const ScalarField2& boundarySdf, const ScalarField2& fluidSdf) {
    _markers.resize(size);

    parallelForEachIndex(_markers.size(), [&](size_t i, size_t j) {
        if (isInsideSdf(boundarySdf.sample(pos(i, j)))) {
            _markers(i, j) = kBoundary;
        } else if (isInsideSdf(fluidSdf.sample(pos(i, j)))) {
            _markers(i, j) = kFluid;
        } else {
            _markers(i, j) = kAir;
        }
    });
}

void GridBackwardEulerDiffusionSolver2::buildMatrix(const Vector2UZ& size,
                                                    const Vector2D& c) {
    _system.A.resize(size);

    bool isDirichlet = (_boundaryType == Dirichlet);

    // Build linear system
    parallelForEachIndex(_system.A.size(), [&](size_t i, size_t j) {
        auto& row = _system.A(i, j);

        // Initialize
        row.center = 1.0;
        row.right = row.up = 0.0;

        if (_markers(i, j) == kFluid) {
            if (i + 1 < size.x) {
                if ((isDirichlet && _markers(i + 1, j) != kAir) ||
                    _markers(i + 1, j) == kFluid) {
                    row.center += c.x;
                }

                if (_markers(i + 1, j) == kFluid) {
                    row.right -= c.x;
                }
            }

            if (i > 0 && ((isDirichlet && _markers(i - 1, j) != kAir) ||
                          _markers(i - 1, j) == kFluid)) {
                row.center += c.x;
            }

            if (j + 1 < size.y) {
                if ((isDirichlet && _markers(i, j + 1) != kAir) ||
                    _markers(i, j + 1) == kFluid) {
                    row.center += c.y;
                }

                if (_markers(i, j + 1) == kFluid) {
                    row.up -= c.y;
                }
            }

            if (j > 0 && ((isDirichlet && _markers(i, j - 1) != kAir) ||
                          _markers(i, j - 1) == kFluid)) {
                row.center += c.y;
            }
        }
    });
}

void GridBackwardEulerDiffusionSolver2::buildVectors(
    const ConstArrayView2<double>& f, const Vector2D& c) {
    Vector2UZ size = f.size();

    _system.x.resize(size, 0.0);
    _system.b.resize(size, 0.0);

    // Build linear system
    parallelForEachIndex(_system.x.size(), [&](size_t i, size_t j) {
        _system.b(i, j) = _system.x(i, j) = f(i, j);

        if (_boundaryType == Dirichlet && _markers(i, j) == kFluid) {
            if (i + 1 < size.x && _markers(i + 1, j) == kBoundary) {
                _system.b(i, j) += c.x * f(i + 1, j);
            }

            if (i > 0 && _markers(i - 1, j) == kBoundary) {
                _system.b(i, j) += c.x * f(i - 1, j);
            }

            if (j + 1 < size.y && _markers(i, j + 1) == kBoundary) {
                _system.b(i, j) += c.y * f(i, j + 1);
            }

            if (j > 0 && _markers(i, j - 1) == kBoundary) {
                _system.b(i, j) += c.y * f(i, j - 1);
            }
        }
    });
}

void GridBackwardEulerDiffusionSolver2::buildVectors(
    const ConstArrayView2<Vector2D>& f, const Vector2D& c,
    size_t component) {
    Vector2UZ size = f.size();

    _system.x.resize(size, 0.0);
    _system.b.resize(size, 0.0);

    // Build linear system
    parallelForEachIndex(_system.x.size(), [&](size_t i, size_t j) {
        _system.b(i, j) = _system.x(i, j) = f(i, j)[component];

        if (_boundaryType == Dirichlet && _markers(i, j) == kFluid) {
            if (i + 1 < size.x && _markers(i + 1, j) == kBoundary) {
                _system.b(i, j) += c.x * f(i + 1, j)[component];
            }

            if (i > 0 && _markers(i - 1, j) == kBoundary) {
                _system.b(i, j) += c.x * f(i - 1, j)[component];
            }

            if (j + 1 < size.y && _markers(i, j + 1) == kBoundary) {
                _system.b(i, j) += c.y * f(i, j + 1)[component];
            }

            if (j > 0 && _markers(i, j - 1) == kBoundary) {
                _system.b(i, j) += c.y * f(i, j - 1)[component];
            }
        }
    });
}
