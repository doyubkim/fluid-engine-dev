// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_utils.h>
#include <jet/grid_forward_euler_diffusion_solver3.h>
#include <jet/level_set_utils.h>
#include <pch.h>

using namespace jet;

static const char kFluid = 0;
static const char kAir = 1;
static const char kBoundary = 2;

template <typename T>
T laplacian(const ConstArrayView3<T>& data, const Array3<char>& marker,
            const Vector3D& gridSpacing, size_t i, size_t j, size_t k) {
    const T center = data(i, j, k);
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    T dleft = T{};
    T dright = T{};
    T ddown = T{};
    T dup = T{};
    T dback = T{};
    T dfront = T{};

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

    return (dright - dleft) / square(gridSpacing.x) +
           (dup - ddown) / square(gridSpacing.y) +
           (dfront - dback) / square(gridSpacing.z);
}

GridForwardEulerDiffusionSolver3::GridForwardEulerDiffusionSolver3() {}

void GridForwardEulerDiffusionSolver3::solve(const ScalarGrid3& source,
                                             double diffusionCoefficient,
                                             double timeIntervalInSeconds,
                                             ScalarGrid3* dest,
                                             const ScalarField3& boundarySdf,
                                             const ScalarField3& fluidSdf) {
    auto src = source.dataView();
    Vector3D h = source.gridSpacing();
    auto pos = source.dataPosition();

    buildMarkers(source.resolution(), pos, boundarySdf, fluidSdf);

    source.parallelForEachDataPointIndex([&](const Vector3UZ& idx) {
        if (_markers(idx) == kFluid) {
            (*dest)(idx) = source(idx) +
                           diffusionCoefficient * timeIntervalInSeconds *
                               laplacian(src, _markers, h, idx.x, idx.y, idx.z);
        } else {
            (*dest)(idx) = source(idx);
        }
    });
}

void GridForwardEulerDiffusionSolver3::solve(
    const CollocatedVectorGrid3& source, double diffusionCoefficient,
    double timeIntervalInSeconds, CollocatedVectorGrid3* dest,
    const ScalarField3& boundarySdf, const ScalarField3& fluidSdf) {
    auto src = source.dataView();
    Vector3D h = source.gridSpacing();
    auto pos = source.dataPosition();

    buildMarkers(source.resolution(), pos, boundarySdf, fluidSdf);

    source.parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        if (_markers(i, j, k) == kFluid) {
            (*dest)(i, j, k) =
                src(i, j, k) + diffusionCoefficient * timeIntervalInSeconds *
                                   laplacian(src, _markers, h, i, j, k);
        } else {
            (*dest)(i, j, k) = source(i, j, k);
        }
    });
}

void GridForwardEulerDiffusionSolver3::solve(const FaceCenteredGrid3& source,
                                             double diffusionCoefficient,
                                             double timeIntervalInSeconds,
                                             FaceCenteredGrid3* dest,
                                             const ScalarField3& boundarySdf,
                                             const ScalarField3& fluidSdf) {
    auto uSrc = source.uView();
    auto vSrc = source.vView();
    auto wSrc = source.wView();
    auto u = dest->uView();
    auto v = dest->vView();
    auto w = dest->wView();
    auto uPos = source.uPosition();
    auto vPos = source.vPosition();
    auto wPos = source.wPosition();
    Vector3D h = source.gridSpacing();

    buildMarkers(source.uSize(), uPos, boundarySdf, fluidSdf);

    source.parallelForEachUIndex([&](const Vector3UZ& idx) {
        if (!isInsideSdf(boundarySdf.sample(uPos(idx)))) {
            u(idx) = uSrc(idx) + diffusionCoefficient * timeIntervalInSeconds *
                                     laplacian3(uSrc, h, idx.x, idx.y, idx.z);
        }
    });

    buildMarkers(source.vSize(), vPos, boundarySdf, fluidSdf);

    source.parallelForEachVIndex([&](const Vector3UZ& idx) {
        if (!isInsideSdf(boundarySdf.sample(vPos(idx)))) {
            v(idx) = vSrc(idx) + diffusionCoefficient * timeIntervalInSeconds *
                                     laplacian3(vSrc, h, idx.x, idx.y, idx.z);
        }
    });

    buildMarkers(source.wSize(), wPos, boundarySdf, fluidSdf);

    source.parallelForEachUIndex([&](const Vector3UZ& idx) {
        if (!isInsideSdf(boundarySdf.sample(wPos(idx)))) {
            w(idx) = wSrc(idx) + diffusionCoefficient *
                                             timeIntervalInSeconds *
                                             laplacian3(wSrc, h, idx.x, idx.y, idx.z);
        }
    });
}

void GridForwardEulerDiffusionSolver3::buildMarkers(
    const Vector3UZ& size,
    const std::function<Vector3D(const Vector3UZ&)>& pos,
    const ScalarField3& boundarySdf, const ScalarField3& fluidSdf) {
    _markers.resize(size);

    forEachIndex(_markers.size(), [&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(boundarySdf.sample(pos(Vector3UZ(i, j, k))))) {
            _markers(i, j, k) = kBoundary;
        } else if (isInsideSdf(fluidSdf.sample(pos(Vector3UZ(i, j, k))))) {
            _markers(i, j, k) = kFluid;
        } else {
            _markers(i, j, k) = kAir;
        }
    });
}
