// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "fdm_linear_system_solver_test_helper3.h"

#include <jet/fdm_gauss_seidel_solver3.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmGaussSeidelSolver3, SolveLowRes) {
    FdmLinearSystem3 system;
    FdmLinearSystemSolverTestHelper3::buildTestLinearSystem(&system, {3, 3, 3});

    FdmGaussSeidelSolver3 solver(100, 10, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmGaussSeidelSolver3, Solve) {
    FdmLinearSystem3 system;
    FdmLinearSystemSolverTestHelper3::buildTestLinearSystem(&system,
                                                            {32, 32, 32});

    auto buffer = system.x;
    FdmBlas3::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmBlas3::l2Norm(buffer);

    FdmGaussSeidelSolver3 solver(100, 10, 1e-9);
    solver.solve(&system);

    FdmBlas3::residual(system.A, system.x, system.b, &buffer);
    double norm1 = FdmBlas3::l2Norm(buffer);

    EXPECT_LT(norm1, norm0);
}

TEST(FdmGaussSeidelSolver3, Relax) {
    FdmLinearSystem3 system;
    FdmLinearSystemSolverTestHelper3::buildTestLinearSystem(&system,
                                                            {32, 32, 32});

    auto buffer = system.x;
    FdmBlas3::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmBlas3::l2Norm(buffer);

    for (int i = 0; i < 200; ++i) {
        FdmGaussSeidelSolver3::relax(system.A, system.b, 1.0, &system.x);

        FdmBlas3::residual(system.A, system.x, system.b, &buffer);
        double norm = FdmBlas3::l2Norm(buffer);
        EXPECT_LT(norm, norm0);

        norm0 = norm;
    }
}

TEST(FdmGaussSeidelSolver3, RelaxRedBlack) {
    FdmLinearSystem3 system;
    FdmLinearSystemSolverTestHelper3::buildTestLinearSystem(&system,
                                                            {32, 32, 32});

    auto buffer = system.x;
    FdmBlas3::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmBlas3::l2Norm(buffer);

    for (int i = 0; i < 200; ++i) {
        FdmGaussSeidelSolver3::relaxRedBlack(system.A, system.b, 1.0,
                                             &system.x);

        FdmBlas3::residual(system.A, system.x, system.b, &buffer);
        double norm = FdmBlas3::l2Norm(buffer);
        if (i > 0) {
            EXPECT_LT(norm, norm0);
        }

        norm0 = norm;
    }
}

TEST(FdmGaussSeidelSolver3, SolveCompressedLowRes) {
    FdmCompressedLinearSystem3 system;
    FdmLinearSystemSolverTestHelper3::buildTestCompressedLinearSystem(
        &system, {3, 3, 3});

    FdmGaussSeidelSolver3 solver(100, 10, 1e-9);
    solver.solveCompressed(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmGaussSeidelSolver3, SolveCompressed) {
    FdmCompressedLinearSystem3 system;
    FdmLinearSystemSolverTestHelper3::buildTestCompressedLinearSystem(
        &system, {32, 32, 32});

    auto buffer = system.x;
    FdmCompressedBlas3::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmCompressedBlas3::l2Norm(buffer);

    FdmGaussSeidelSolver3 solver(100, 10, 1e-9);
    solver.solveCompressed(&system);

    FdmCompressedBlas3::residual(system.A, system.x, system.b, &buffer);
    double norm1 = FdmCompressedBlas3::l2Norm(buffer);

    EXPECT_LT(norm1, norm0);
}
