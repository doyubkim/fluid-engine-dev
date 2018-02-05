// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/fdm_utils.h>
#include <jet/grid_forward_euler_diffusion_solver2.h>
#include <jet/level_set_utils.h>

using namespace jet;

static const char kFluid = 0;
static const char kAir = 1;
static const char kBoundary = 2;

template <typename T>
inline T laplacian(
    const ConstArrayAccessor2<T>& data,
    const Array2<char>& marker,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j) {
    const T center = data(i, j);
    const Size2 ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    T dleft = zero<T>();
    T dright = zero<T>();
    T ddown = zero<T>();
    T dup = zero<T>();

    if (i > 0 && marker(i - 1, j) == kFluid) {
        dleft = center - data(i - 1, j);
    }
    if (i + 1 < ds.x && marker(i + 1, j) == kFluid) {
        dright = data(i + 1, j) - center;
    }

    if (j > 0 && marker(i, j - 1) == kFluid) {
        ddown = center - data(i, j - 1);
    }
    if (j + 1 < ds.y && marker(i, j + 1) == kFluid) {
        dup = data(i, j + 1) - center;
    }

    return (dright - dleft) / square(gridSpacing.x)
        + (dup - ddown) / square(gridSpacing.y);
}

GridForwardEulerDiffusionSolver2::GridForwardEulerDiffusionSolver2() {
}

void GridForwardEulerDiffusionSolver2::solve(
    const ScalarGrid2& source,
    double diffusionCoefficient,
    double timeIntervalInSeconds,
    ScalarGrid2* dest,
    const ScalarField2& boundarySdf,
    const ScalarField2& fluidSdf) {
    auto src = source.constDataAccessor();
    Vector2D h = source.gridSpacing();
    auto pos = source.dataPosition();

    buildMarkers(source.resolution(), pos, boundarySdf, fluidSdf);

    source.parallelForEachDataPointIndex(
        [&](size_t i, size_t j) {
            if (_markers(i, j) == kFluid) {
                (*dest)(i, j)
                    = source(i, j)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian(src, _markers, h, i, j);
            } else {
                (*dest)(i, j) = source(i, j);
            }
        });
}

void GridForwardEulerDiffusionSolver2::solve(
    const CollocatedVectorGrid2& source,
    double diffusionCoefficient,
    double timeIntervalInSeconds,
    CollocatedVectorGrid2* dest,
    const ScalarField2& boundarySdf,
    const ScalarField2& fluidSdf) {
    auto src = source.constDataAccessor();
    Vector2D h = source.gridSpacing();
    auto pos = source.dataPosition();

    buildMarkers(source.resolution(), pos, boundarySdf, fluidSdf);

    source.parallelForEachDataPointIndex(
        [&](size_t i, size_t j) {
            if (_markers(i, j) == kFluid) {
                (*dest)(i, j)
                    = src(i, j)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian(src, _markers, h, i, j);
            } else {
                (*dest)(i, j) = src(i, j);
            }
        });
}

void GridForwardEulerDiffusionSolver2::solve(
    const FaceCenteredGrid2& source,
    double diffusionCoefficient,
    double timeIntervalInSeconds,
    FaceCenteredGrid2* dest,
    const ScalarField2& boundarySdf,
    const ScalarField2& fluidSdf) {
    auto uSrc = source.uConstAccessor();
    auto vSrc = source.vConstAccessor();
    auto u = dest->uAccessor();
    auto v = dest->vAccessor();
    auto uPos = source.uPosition();
    auto vPos = source.vPosition();
    Vector2D h = source.gridSpacing();

    buildMarkers(source.uSize(), uPos, boundarySdf, fluidSdf);

    source.parallelForEachUIndex(
        [&](size_t i, size_t j) {
            if (_markers(i, j) == kFluid) {
                u(i, j)
                    = uSrc(i, j)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian(uSrc, _markers, h, i, j);
            } else {
                u(i, j) = uSrc(i, j);
            }
        });

    buildMarkers(source.vSize(), vPos, boundarySdf, fluidSdf);

    source.parallelForEachVIndex(
        [&](size_t i, size_t j) {
            if (_markers(i, j) == kFluid) {
                v(i, j)
                    = vSrc(i, j)
                    + diffusionCoefficient
                    * timeIntervalInSeconds
                    * laplacian(vSrc, _markers, h, i, j);
            } else {
                v(i, j) = vSrc(i, j);
            }
        });
}

void GridForwardEulerDiffusionSolver2::buildMarkers(
    const Size2& size,
    const std::function<Vector2D(size_t, size_t)>& pos,
    const ScalarField2& boundarySdf,
    const ScalarField2& fluidSdf) {
    _markers.resize(size);

    _markers.forEachIndex(
        [&](size_t i, size_t j) {
            if (isInsideSdf(boundarySdf.sample(pos(i, j)))) {
                _markers(i, j) = kBoundary;
            } else if (isInsideSdf(fluidSdf.sample(pos(i, j)))) {
                _markers(i, j) = kFluid;
            } else {
                _markers(i, j) = kAir;
            }
        });
}
