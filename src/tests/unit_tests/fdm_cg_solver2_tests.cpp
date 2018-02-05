// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_linear_system_solver_test_helper2.h"

#include <jet/fdm_cg_solver2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmCgSolver2, Solve) {
    FdmLinearSystem2 system;
    FdmLinearSystemSolverTestHelper2::buildTestLinearSystem(&system, {3, 3});

    FdmCgSolver2 solver(10, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmCgSolver2, SolveCompressed) {
    FdmCompressedLinearSystem2 system;
    FdmLinearSystemSolverTestHelper2::buildTestCompressedLinearSystem(&system,
                                                                      {3, 3});

    FdmCgSolver2 solver(10, 1e-9);
    solver.solveCompressed(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}
