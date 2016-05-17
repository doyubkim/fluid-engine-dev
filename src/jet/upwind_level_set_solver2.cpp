// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/pde.h>
#include <jet/upwind_level_set_solver2.h>

#include <algorithm>

using namespace jet;

UpwindLevelSetSolver2::UpwindLevelSetSolver2() {
    setMaxCfl(0.5);
}

void UpwindLevelSetSolver2::getDerivatives(
    ConstArrayAccessor2<double> grid,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j,
    std::array<double, 2>* dx,
    std::array<double, 2>* dy) const {
    double D0[3];
    Size2 size = grid.size();

    const size_t im1 = (i < 1) ? 0 : i - 1;
    const size_t ip1 = std::min(i + 1, size.x - 1);
    const size_t jm1 = (j < 1) ? 0 : j - 1;
    const size_t jp1 = std::min(j + 1, size.y - 1);

    D0[0] = grid(im1, j);
    D0[1] = grid(i, j);
    D0[2] = grid(ip1, j);
    *dx = upwind1(D0, gridSpacing.x);

    D0[0] = grid(i, jm1);
    D0[1] = grid(i, j);
    D0[2] = grid(i, jp1);
    *dy = upwind1(D0, gridSpacing.y);
}
