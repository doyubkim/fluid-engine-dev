// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_utils.h>
#include <jet/fdm_utils.h>
#include <jet/iterative_level_set_solver3.h>
#include <jet/parallel.h>
#include <common.h>

#include <algorithm>
#include <limits>
#include <utility>  // just make cpplint happy..

using namespace jet;

IterativeLevelSetSolver3::IterativeLevelSetSolver3() {}

IterativeLevelSetSolver3::~IterativeLevelSetSolver3() {}

void IterativeLevelSetSolver3::reinitialize(const ScalarGrid3& inputSdf,
                                            double maxDistance,
                                            ScalarGrid3* outputSdf) {
    const Vector3UZ size = inputSdf.dataSize();
    const Vector3D gridSpacing = inputSdf.gridSpacing();

    JET_THROW_INVALID_ARG_IF(!inputSdf.hasSameShape(*outputSdf));

    auto outputAcc = outputSdf->dataView();

    const double dtau = pseudoTimeStep(inputSdf.dataView(), gridSpacing);
    const unsigned int numberOfIterations =
        distanceToNumberOfIterations(maxDistance, dtau);

    copy(inputSdf.dataView(), outputAcc);

    Array3<double> temp(size);
    ArrayView3<double> tempAcc(temp);

    JET_INFO << "Reinitializing with pseudoTimeStep: " << dtau
             << " numberOfIterations: " << numberOfIterations;

    for (unsigned int n = 0; n < numberOfIterations; ++n) {
        inputSdf.parallelForEachDataPointIndex(
            [&](size_t i, size_t j, size_t k) {
                double s = sign(outputAcc, gridSpacing, i, j, k);

                std::array<double, 2> dx, dy, dz;

                getDerivatives(outputAcc, gridSpacing, i, j, k, &dx, &dy, &dz);

                // Explicit Euler step
                double val = outputAcc(i, j, k) -
                             dtau * std::max(s, 0.0) *
                                 (std::sqrt(square(std::max(dx[0], 0.0)) +
                                            square(std::min(dx[1], 0.0)) +
                                            square(std::max(dy[0], 0.0)) +
                                            square(std::min(dy[1], 0.0)) +
                                            square(std::max(dz[0], 0.0)) +
                                            square(std::min(dz[1], 0.0))) -
                                  1.0) -
                             dtau * std::min(s, 0.0) *
                                 (std::sqrt(square(std::min(dx[0], 0.0)) +
                                            square(std::max(dx[1], 0.0)) +
                                            square(std::min(dy[0], 0.0)) +
                                            square(std::max(dy[1], 0.0)) +
                                            square(std::min(dz[0], 0.0)) +
                                            square(std::max(dz[1], 0.0))) -
                                  1.0);
                tempAcc(i, j, k) = val;
            });

        std::swap(tempAcc, outputAcc);
    }

    auto outputSdfAcc = outputSdf->dataView();
    copy(outputAcc, outputSdfAcc);
}

void IterativeLevelSetSolver3::extrapolate(const ScalarGrid3& input,
                                           const ScalarField3& sdf,
                                           double maxDistance,
                                           ScalarGrid3* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array3<double> sdfGrid(input.dataSize());
    auto pos = input.dataPosition();
    parallelForEachIndex(sdfGrid.size(), [&](size_t i, size_t j, size_t k) {
        sdfGrid(i, j, k) = sdf.sample(pos(i, j, k));
    });

    extrapolate(input.dataView(), sdfGrid, input.gridSpacing(), maxDistance,
                output->dataView());
}

void IterativeLevelSetSolver3::extrapolate(const CollocatedVectorGrid3& input,
                                           const ScalarField3& sdf,
                                           double maxDistance,
                                           CollocatedVectorGrid3* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array3<double> sdfGrid(input.dataSize());
    auto pos = input.dataPosition();
    parallelForEachIndex(sdfGrid.size(), [&](size_t i, size_t j, size_t k) {
        sdfGrid(i, j, k) = sdf.sample(pos(i, j, k));
    });

    const Vector3D gridSpacing = input.gridSpacing();

    Array3<double> u(input.dataSize());
    Array3<double> u0(input.dataSize());
    Array3<double> v(input.dataSize());
    Array3<double> v0(input.dataSize());
    Array3<double> w(input.dataSize());
    Array3<double> w0(input.dataSize());

    input.parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        u(i, j, k) = input(i, j, k).x;
        v(i, j, k) = input(i, j, k).y;
        w(i, j, k) = input(i, j, k).z;
    });

    extrapolate(u, sdfGrid, gridSpacing, maxDistance, u0);

    extrapolate(v, sdfGrid, gridSpacing, maxDistance, v0);

    extrapolate(w, sdfGrid, gridSpacing, maxDistance, w0);

    output->parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        (*output)(i, j, k).x = u(i, j, k);
        (*output)(i, j, k).y = v(i, j, k);
        (*output)(i, j, k).z = w(i, j, k);
    });
}

void IterativeLevelSetSolver3::extrapolate(const FaceCenteredGrid3& input,
                                           const ScalarField3& sdf,
                                           double maxDistance,
                                           FaceCenteredGrid3* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    const Vector3D gridSpacing = input.gridSpacing();

    auto u = input.uView();
    auto uPos = input.uPosition();
    Array3<double> sdfAtU(u.size());
    input.parallelForEachUIndex(
        [&](const Vector3UZ& idx) { sdfAtU(idx) = sdf.sample(uPos(idx)); });

    extrapolate(u, sdfAtU, gridSpacing, maxDistance, output->uView());

    auto v = input.vView();
    auto vPos = input.vPosition();
    Array3<double> sdfAtV(v.size());
    input.parallelForEachVIndex(
        [&](const Vector3UZ& idx) { sdfAtV(idx) = sdf.sample(vPos(idx)); });

    extrapolate(v, sdfAtV, gridSpacing, maxDistance, output->vView());

    auto w = input.wView();
    auto wPos = (input.wPosition());
    Array3<double> sdfAtW(w.size());
    input.parallelForEachWIndex(
        [&](const Vector3UZ& idx) { sdfAtW(idx) = sdf.sample(wPos(idx)); });

    extrapolate(w, sdfAtW, gridSpacing, maxDistance, output->wView());
}

void IterativeLevelSetSolver3::extrapolate(const ConstArrayView3<double>& input,
                                           const ConstArrayView3<double>& sdf,
                                           const Vector3D& gridSpacing,
                                           double maxDistance,
                                           ArrayView3<double> output) {
    const Vector3UZ size = input.size();

    ArrayView3<double> outputAcc = output;

    const double dtau = pseudoTimeStep(sdf, gridSpacing);
    const unsigned int numberOfIterations =
        distanceToNumberOfIterations(maxDistance, dtau);

    copy(input, outputAcc);

    Array3<double> temp(size);
    ArrayView3<double> tempAcc(temp);

    for (unsigned int n = 0; n < numberOfIterations; ++n) {
        parallelFor(
            kZeroSize, size.x, kZeroSize, size.y, kZeroSize, size.z,
            [&](size_t i, size_t j, size_t k) {
                if (sdf(i, j, k) >= 0) {
                    std::array<double, 2> dx, dy, dz;
                    Vector3D grad = gradient3(sdf, gridSpacing, i, j, k);

                    getDerivatives(outputAcc, gridSpacing, i, j, k, &dx, &dy,
                                   &dz);

                    tempAcc(i, j, k) = outputAcc(i, j, k) -
                                       dtau * (std::max(grad.x, 0.0) * dx[0] +
                                               std::min(grad.x, 0.0) * dx[1] +
                                               std::max(grad.y, 0.0) * dy[0] +
                                               std::min(grad.y, 0.0) * dy[1] +
                                               std::max(grad.z, 0.0) * dz[0] +
                                               std::min(grad.z, 0.0) * dz[1]);
                } else {
                    tempAcc(i, j, k) = outputAcc(i, j, k);
                }
            });

        std::swap(tempAcc, outputAcc);
    }

    copy(outputAcc, output);
}

double IterativeLevelSetSolver3::maxCfl() const { return _maxCfl; }

void IterativeLevelSetSolver3::setMaxCfl(double newMaxCfl) {
    _maxCfl = std::max(newMaxCfl, 0.0);
}

unsigned int IterativeLevelSetSolver3::distanceToNumberOfIterations(
    double distance, double dtau) {
    return static_cast<unsigned int>(std::ceil(distance / dtau));
}

double IterativeLevelSetSolver3::sign(const ConstArrayView3<double>& sdf,
                                      const Vector3D& gridSpacing, size_t i,
                                      size_t j, size_t k) {
    double d = sdf(i, j, k);
    double e = min3(gridSpacing.x, gridSpacing.y, gridSpacing.z);
    return d / std::sqrt(d * d + e * e);
}

double IterativeLevelSetSolver3::pseudoTimeStep(ConstArrayView3<double> sdf,
                                                const Vector3D& gridSpacing) {
    const Vector3UZ size = sdf.size();

    const double h = max3(gridSpacing.x, gridSpacing.y, gridSpacing.z);

    double maxS = -std::numeric_limits<double>::max();
    double dtau = _maxCfl * h;

    for (size_t k = 0; k < size.z; ++k) {
        for (size_t j = 0; j < size.y; ++j) {
            for (size_t i = 0; i < size.x; ++i) {
                double s = sign(sdf, gridSpacing, i, j, k);
                maxS = std::max(s, maxS);
            }
        }
    }

    while (dtau * maxS / h > _maxCfl) {
        dtau *= 0.5;
    }

    return dtau;
}
