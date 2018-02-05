// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/grid_forward_euler_diffusion_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridForwardEulerDiffusionSolver2, Solve) {
    CellCenteredScalarGrid2 src(3, 3, 1.0, 1.0, 0.0, 0.0);
    CellCenteredScalarGrid2 dst(3, 3, 1.0, 1.0, 0.0, 0.0);

    src(1, 1) = 1.0;

    GridForwardEulerDiffusionSolver2 diffusionSolver;
    diffusionSolver.solve(src, 1.0 / 8.0, 1.0, &dst);

    Array2<double> solution = {
        {0.0,     1.0/8.0, 0.0},
        {1.0/8.0, 1.0/2.0, 1.0/8.0},
        {0.0,     1.0/8.0, 0.0}
    };

    dst.forEachDataPointIndex([&](size_t i, size_t j) {
        EXPECT_NEAR(solution(i, j), dst(i, j), 1e-6);
    });
}
