// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/fdm_utils.h>
#include <jet/grid_forward_euler_diffusion_solver3.h>
#include <jet/level_set_utils.h>

using namespace jet;

static const char kFluid = 0;
static const char kAir = 1;
static const char kBoundary = 2;

template <typename T>
T laplacian(
    const ConstArrayAccessor3<T>& data,
    const Array3<char>& marker,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k) {
    const T center = data(i, j, k);
    const Size3 ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    T dleft = zero<T>();
    T dright = zero<T>();
    T ddown = zero<T>();
    T dup = zero<T>();
    T dback = zero<T>();
    T dfront = zero<T>();

    if (i > 0 && marker(i - 1, j, k) == kFluid) {
        dleft = center - data(i - 1, j, k);
    }
    if (i + 1 < ds.x && marker(i + 1, j, k) == kFluid) {
        dright = data(i + 1, j, k) - center;
    }

    if (j > 0 && marker(i, j - 1, k) == kFluid) {
        ddown = center - data(i, j - 1, k);
    }
    if (j + 1 < ds.y && marker(i, j + 1, k) == kFluid) {
        dup = data(i, j + 1, k) - center;
    }

    if (k > 0 && marker(i, j, k - 1) == kFluid) {
        dback = center - data(i, j, k - 1);
    }
    if (k + 1 < ds.z && marker(i, j, k + 1) == kFluid) {
        dfront = data(i, j, k + 1) - center;
    }

    return (dright - dleft) / square(gridSpacing.x)
        + (dup - ddown) / square(gridSpacing.y)
        + (dfront - dback) / square(gridSpacing.z);
}

GridForwardEulerDiffusionSolver3::GridForwardEulerDiffusionSolver3() {
}

void GridForwardEulerDiffusionSolver3::solve(
    const ScalarGrid3& source,
    double diffusionCoefficient,
    double timeIntervalInSeconds,
    ScalarGrid3* dest,
    const ScalarField3& boundarySdf,
    const ScalarField3& fluidSdf) {
    auto src = source.constDataAccessor();
    Vector3D h = source.gridSpacing();
    auto pos = source.dataPosition();

    buildMarkers(source.resolution(), pos, boundarySdf, fluidSdf);

    source.parallelForEachDataPointIndex(
        [&](size_t i, size_t j, size_t k) {
            if (_markers(i, j, k) == kFluid) {
                (*dest)(i, j, k)
                    = source(i, j, k)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian(src, _markers, h, i, j, k);
            } else {
                (*dest)(i, j, k) = source(i, j, k);
            }
        });
}

void GridForwardEulerDiffusionSolver3::solve(
    const CollocatedVectorGrid3& source,
    double diffusionCoefficient,
    double timeIntervalInSeconds,
    CollocatedVectorGrid3* dest,
    const ScalarField3& boundarySdf,
    const ScalarField3& fluidSdf) {
    auto src = source.constDataAccessor();
    Vector3D h = source.gridSpacing();
    auto pos = source.dataPosition();

    buildMarkers(source.resolution(), pos, boundarySdf, fluidSdf);

    source.parallelForEachDataPointIndex(
        [&](size_t i, size_t j, size_t k) {
            if (_markers(i, j, k) == kFluid) {
                (*dest)(i, j, k)
                    = src(i, j, k)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian(src, _markers, h, i, j, k);
            } else {
                (*dest)(i, j, k) = source(i, j, k);
            }
        });
}

void GridForwardEulerDiffusionSolver3::solve(
    const FaceCenteredGrid3& source,
    double diffusionCoefficient,
    double timeIntervalInSeconds,
    FaceCenteredGrid3* dest,
    const ScalarField3& boundarySdf,
    const ScalarField3& fluidSdf) {
    auto uSrc = source.uConstAccessor();
    auto vSrc = source.vConstAccessor();
    auto wSrc = source.wConstAccessor();
    auto u = dest->uAccessor();
    auto v = dest->vAccessor();
    auto w = dest->wAccessor();
    auto uPos = source.uPosition();
    auto vPos = source.vPosition();
    auto wPos = source.wPosition();
    Vector3D h = source.gridSpacing();

    buildMarkers(source.uSize(), uPos, boundarySdf, fluidSdf);

    source.parallelForEachUIndex(
        [&](size_t i, size_t j, size_t k) {
            if (!isInsideSdf(boundarySdf.sample(uPos(i, j, k)))) {
                u(i, j, k)
                    = uSrc(i, j, k)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian3(uSrc, h, i, j, k);
            }
        });

    buildMarkers(source.vSize(), vPos, boundarySdf, fluidSdf);

    source.parallelForEachVIndex(
        [&](size_t i, size_t j, size_t k) {
            if (!isInsideSdf(boundarySdf.sample(vPos(i, j, k)))) {
                v(i, j, k)
                    = vSrc(i, j, k)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian3(vSrc, h, i, j, k);
            }
        });

    buildMarkers(source.wSize(), wPos, boundarySdf, fluidSdf);

    source.parallelForEachWIndex(
        [&](size_t i, size_t j, size_t k) {
            if (!isInsideSdf(boundarySdf.sample(wPos(i, j, k)))) {
                w(i, j, k)
                    = wSrc(i, j, k)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian3(wSrc, h, i, j, k);
            }
        });
}

void GridForwardEulerDiffusionSolver3::buildMarkers(
    const Size3& size,
    const std::function<Vector3D(size_t, size_t, size_t)>& pos,
    const ScalarField3& boundarySdf,
    const ScalarField3& fluidSdf) {
    _markers.resize(size);

    _markers.forEachIndex(
        [&](size_t i, size_t j, size_t k) {
            if (isInsideSdf(boundarySdf.sample(pos(i, j, k)))) {
                _markers(i, j, k) = kBoundary;
            } else if (isInsideSdf(fluidSdf.sample(pos(i, j, k)))) {
                _markers(i, j, k) = kFluid;
            } else {
                _markers(i, j, k) = kAir;
            }
        });
}
