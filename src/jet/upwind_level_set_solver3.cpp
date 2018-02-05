// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/pde.h>
#include <jet/upwind_level_set_solver3.h>

#include <algorithm>

using namespace jet;

UpwindLevelSetSolver3::UpwindLevelSetSolver3() {
    setMaxCfl(0.5);
}

void UpwindLevelSetSolver3::getDerivatives(
    ConstArrayAccessor3<double> grid,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k,
    std::array<double, 2>* dx,
    std::array<double, 2>* dy,
    std::array<double, 2>* dz) const {
    double D0[3];
    Size3 size = grid.size();

    const size_t im1 = (i < 1) ? 0 : i - 1;
    const size_t ip1 = std::min(i + 1, size.x - 1);
    const size_t jm1 = (j < 1) ? 0 : j - 1;
    const size_t jp1 = std::min(j + 1, size.y - 1);
    const size_t km1 = (k < 1) ? 0 : k - 1;
    const size_t kp1 = std::min(k + 1, size.z - 1);

    D0[0] = grid(im1, j, k);
    D0[1] = grid(i, j, k);
    D0[2] = grid(ip1, j, k);
    *dx = upwind1(D0, gridSpacing.x);

    D0[0] = grid(i, jm1, k);
    D0[1] = grid(i, j, k);
    D0[2] = grid(i, jp1, k);
    *dy = upwind1(D0, gridSpacing.y);

    D0[0] = grid(i, j, km1);
    D0[1] = grid(i, j, k);
    D0[2] = grid(i, j, kp1);
    *dz = upwind1(D0, gridSpacing.z);
}
