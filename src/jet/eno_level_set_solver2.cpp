// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/pde.h>
#include <jet/eno_level_set_solver2.h>
#include <algorithm>

using namespace jet;

EnoLevelSetSolver2::EnoLevelSetSolver2() {
    setMaxCfl(0.25);
}

void EnoLevelSetSolver2::getDerivatives(
    ConstArrayAccessor2<double> grid,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j,
    std::array<double, 2>* dx,
    std::array<double, 2>* dy) const {
    double D0[7];
    Size2 size = grid.size();

    const size_t im3 = (i < 3) ? 0 : i - 3;
    const size_t im2 = (i < 2) ? 0 : i - 2;
    const size_t im1 = (i < 1) ? 0 : i - 1;
    const size_t ip1 = std::min(i + 1, size.x - 1);
    const size_t ip2 = std::min(i + 2, size.x - 1);
    const size_t ip3 = std::min(i + 3, size.x - 1);
    const size_t jm3 = (j < 3) ? 0 : j - 3;
    const size_t jm2 = (j < 2) ? 0 : j - 2;
    const size_t jm1 = (j < 1) ? 0 : j - 1;
    const size_t jp1 = std::min(j + 1, size.y - 1);
    const size_t jp2 = std::min(j + 2, size.y - 1);
    const size_t jp3 = std::min(j + 3, size.y - 1);

    // 3rd-order ENO differencing
    D0[0] = grid(im3, j);
    D0[1] = grid(im2, j);
    D0[2] = grid(im1, j);
    D0[3] = grid(i, j);
    D0[4] = grid(ip1, j);
    D0[5] = grid(ip2, j);
    D0[6] = grid(ip3, j);
    *dx = eno3(D0, gridSpacing.x);

    D0[0] = grid(i, jm3);
    D0[1] = grid(i, jm2);
    D0[2] = grid(i, jm1);
    D0[3] = grid(i, j);
    D0[4] = grid(i, jp1);
    D0[5] = grid(i, jp2);
    D0[6] = grid(i, jp3);
    *dy = eno3(D0, gridSpacing.y);
}
