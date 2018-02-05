// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/array_utils.h>
#include <jet/fdm_utils.h>
#include <jet/iterative_level_set_solver2.h>
#include <jet/parallel.h>

#include <algorithm>
#include <limits>
#include <utility>  // just make cpplint happy..

using namespace jet;

IterativeLevelSetSolver2::IterativeLevelSetSolver2() {
}

IterativeLevelSetSolver2::~IterativeLevelSetSolver2() {
}

void IterativeLevelSetSolver2::reinitialize(
    const ScalarGrid2& inputSdf,
    double maxDistance,
    ScalarGrid2* outputSdf) {
    const Size2 size = inputSdf.dataSize();
    const Vector2D gridSpacing = inputSdf.gridSpacing();

    JET_THROW_INVALID_ARG_IF(!inputSdf.hasSameShape(*outputSdf));

    ArrayAccessor2<double> outputAcc = outputSdf->dataAccessor();

    const double dtau = pseudoTimeStep(
        inputSdf.constDataAccessor(), gridSpacing);
    const unsigned int numberOfIterations
        = distanceToNumberOfIterations(maxDistance, dtau);

    copyRange2(inputSdf.constDataAccessor(), size.x, size.y, &outputAcc);

    Array2<double> temp(size);
    ArrayAccessor2<double> tempAcc = temp.accessor();

    JET_INFO << "Reinitializing with pseudoTimeStep: " << dtau
             << " numberOfIterations: " << numberOfIterations;

    for (unsigned int n = 0; n < numberOfIterations; ++n) {
        inputSdf.parallelForEachDataPointIndex(
            [&](size_t i, size_t j) {
                double s = sign(outputAcc, gridSpacing, i, j);

                std::array<double, 2> dx, dy;

                getDerivatives(outputAcc, gridSpacing, i, j, &dx, &dy);

                // Explicit Euler step
                double val = outputAcc(i, j)
                    - dtau * std::max(s, 0.0)
                        * (std::sqrt(square(std::max(dx[0], 0.0))
                                   + square(std::min(dx[1], 0.0))
                                   + square(std::max(dy[0], 0.0))
                                   + square(std::min(dy[1], 0.0))) - 1.0)
                    - dtau * std::min(s, 0.0)
                        * (std::sqrt(square(std::min(dx[0], 0.0))
                                   + square(std::max(dx[1], 0.0))
                                   + square(std::min(dy[0], 0.0))
                                   + square(std::max(dy[1], 0.0))) - 1.0);
                tempAcc(i, j) = val;
            });

        std::swap(tempAcc, outputAcc);
    }

    auto outputSdfAcc = outputSdf->dataAccessor();
    copyRange2(outputAcc, size.x, size.y, &outputSdfAcc);
}

void IterativeLevelSetSolver2::extrapolate(
    const ScalarGrid2& input,
    const ScalarField2& sdf,
    double maxDistance,
    ScalarGrid2* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array2<double> sdfGrid(input.dataSize());
    auto pos = input.dataPosition();
    sdfGrid.parallelForEachIndex([&](size_t i, size_t j) {
        sdfGrid(i, j) = sdf.sample(pos(i, j));
    });

    extrapolate(
        input.constDataAccessor(),
        sdfGrid.constAccessor(),
        input.gridSpacing(),
        maxDistance,
        output->dataAccessor());
}

void IterativeLevelSetSolver2::extrapolate(
    const CollocatedVectorGrid2& input,
    const ScalarField2& sdf,
    double maxDistance,
    CollocatedVectorGrid2* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array2<double> sdfGrid(input.dataSize());
    auto pos = input.dataPosition();
    sdfGrid.parallelForEachIndex([&](size_t i, size_t j) {
        sdfGrid(i, j) = sdf.sample(pos(i, j));
    });

    const Vector2D gridSpacing = input.gridSpacing();

    Array2<double> u(input.dataSize());
    Array2<double> u0(input.dataSize());
    Array2<double> v(input.dataSize());
    Array2<double> v0(input.dataSize());

    input.parallelForEachDataPointIndex([&](size_t i, size_t j) {
        u(i, j) = input(i, j).x;
        v(i, j) = input(i, j).y;
    });

    extrapolate(
        u,
        sdfGrid.constAccessor(),
        gridSpacing,
        maxDistance,
        u0);

    extrapolate(
        v,
        sdfGrid.constAccessor(),
        gridSpacing,
        maxDistance,
        v0);

    output->parallelForEachDataPointIndex([&](size_t i, size_t j) {
        (*output)(i, j).x = u(i, j);
        (*output)(i, j).y = v(i, j);
    });
}

void IterativeLevelSetSolver2::extrapolate(
    const FaceCenteredGrid2& input,
    const ScalarField2& sdf,
    double maxDistance,
    FaceCenteredGrid2* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    const Vector2D gridSpacing = input.gridSpacing();

    auto u = input.uConstAccessor();
    auto uPos = input.uPosition();
    Array2<double> sdfAtU(u.size());
    input.parallelForEachUIndex([&](size_t i, size_t j) {
        sdfAtU(i, j) = sdf.sample(uPos(i, j));
    });

    extrapolate(
        u,
        sdfAtU,
        gridSpacing,
        maxDistance,
        output->uAccessor());

    auto v = input.vConstAccessor();
    auto vPos = input.vPosition();
    Array2<double> sdfAtV(v.size());
    input.parallelForEachVIndex([&](size_t i, size_t j) {
        sdfAtV(i, j) = sdf.sample(vPos(i, j));
    });

    extrapolate(
        v,
        sdfAtV,
        gridSpacing,
        maxDistance,
        output->vAccessor());
}

void IterativeLevelSetSolver2::extrapolate(
    const ConstArrayAccessor2<double>& input,
    const ConstArrayAccessor2<double>& sdf,
    const Vector2D& gridSpacing,
    double maxDistance,
    ArrayAccessor2<double> output) {
    const Size2 size = input.size();

    ArrayAccessor2<double> outputAcc = output;

    const double dtau = pseudoTimeStep(sdf, gridSpacing);
    const unsigned int numberOfIterations
        = distanceToNumberOfIterations(maxDistance, dtau);

    copyRange2(input, size.x, size.y, &outputAcc);

    Array2<double> temp(size);
    ArrayAccessor2<double> tempAcc = temp.accessor();

    for (unsigned int n = 0; n < numberOfIterations; ++n) {
        parallelFor(
            kZeroSize, size.x, kZeroSize, size.y,
            [&](size_t i, size_t j) {
                if (sdf(i, j) >= 0) {
                    std::array<double, 2> dx, dy;
                    Vector2D grad = gradient2(sdf, gridSpacing, i, j);

                    getDerivatives(outputAcc, gridSpacing, i, j, &dx, &dy);

                    tempAcc(i, j) = outputAcc(i, j)
                        - dtau * (std::max(grad.x, 0.0) * dx[0]
                                + std::min(grad.x, 0.0) * dx[1]
                                + std::max(grad.y, 0.0) * dy[0]
                                + std::min(grad.y, 0.0) * dy[1]);
                } else {
                    tempAcc(i, j) = outputAcc(i, j);
                }
          });

        std::swap(tempAcc, outputAcc);
    }

    copyRange2(outputAcc, size.x, size.y, &output);
}

double IterativeLevelSetSolver2::maxCfl() const {
    return _maxCfl;
}

void IterativeLevelSetSolver2::setMaxCfl(double newMaxCfl) {
    _maxCfl = std::max(newMaxCfl, 0.0);
}

unsigned int IterativeLevelSetSolver2::distanceToNumberOfIterations(
    double distance,
    double dtau) {
    return static_cast<unsigned int>(std::ceil(distance / dtau));
}

double IterativeLevelSetSolver2::sign(
    const ConstArrayAccessor2<double>& sdf,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j) {
    double d = sdf(i, j);
    double e = std::min(gridSpacing.x, gridSpacing.y);
    return d / std::sqrt(d * d + e * e);
}

double IterativeLevelSetSolver2::pseudoTimeStep(
    ConstArrayAccessor2<double> sdf,
    const Vector2D& gridSpacing) {
    const Size2 size = sdf.size();

    const double h = std::max(gridSpacing.x, gridSpacing.y);

    double maxS = -std::numeric_limits<double>::max();
    double dtau = _maxCfl * h;

    for (size_t j = 0; j < size.y; ++j) {
        for (size_t i = 0; i < size.x; ++i) {
            double s = sign(sdf, gridSpacing, i, j);
            maxS = std::max(s, maxS);
        }
    }

    while (dtau * maxS / h > _maxCfl) {
        dtau *= 0.5;
    }

    return dtau;
}
