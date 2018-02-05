// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid3.h>
#include <jet/grid_backward_euler_diffusion_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridBackwardEulerDiffusionSolver3, Solve) {
    CellCenteredScalarGrid3 src(3, 3, 3, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
    CellCenteredScalarGrid3 dst(3, 3, 3, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);

    src(1, 1, 1) = 1.0;

    GridBackwardEulerDiffusionSolver3 diffusionSolver;
    diffusionSolver.solve(src, 1.0 / 12.0, 1.0, &dst);

    Array3<double> solution = {
        {
            {0.001058, 0.005291, 0.001058},
            {0.005291, 0.041270, 0.005291},
            {0.001058, 0.005291, 0.001058}
        },
        {
            {0.005291, 0.041270, 0.005291},
            {0.041270, 0.680423, 0.041270},
            {0.005291, 0.041270, 0.005291}
        },
        {
            {0.001058, 0.005291, 0.001058},
            {0.005291, 0.041270, 0.005291},
            {0.001058, 0.005291, 0.001058}
        }
    };

    dst.forEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        EXPECT_NEAR(solution(i, j, k), dst(i, j, k), 1e-6);
    });
}
