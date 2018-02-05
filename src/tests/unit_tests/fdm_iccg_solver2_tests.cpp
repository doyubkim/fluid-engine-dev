// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_linear_system_solver_test_helper2.h"

#include <jet/fdm_iccg_solver2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmIccgSolver2, SolveLowRes) {
    FdmLinearSystem2 system;
    FdmLinearSystemSolverTestHelper2::buildTestLinearSystem(&system, {3, 3});

    FdmIccgSolver2 solver(10, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmIccgSolver2, Solve) {
    FdmLinearSystem2 system;
    FdmLinearSystemSolverTestHelper2::buildTestLinearSystem(&system,
                                                            {128, 128});

    FdmIccgSolver2 solver(200, 1e-4);
    EXPECT_TRUE(solver.solve(&system));
}

TEST(FdmIccgSolver2, SolveCompressed) {
    FdmLinearSystem2 system;
    FdmLinearSystemSolverTestHelper2::buildTestLinearSystem(&system, {3, 3});

    FdmIccgSolver2 solver(200, 1e-4);
    EXPECT_TRUE(solver.solve(&system));
}
