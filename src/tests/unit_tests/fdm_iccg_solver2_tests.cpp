// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <gtest/gtest.h>
#include <jet/fdm_iccg_solver2.h>

using namespace jet;

TEST(FdmIccgSolver2, SolveLowRes) {
    FdmLinearSystem2 system;
    system.A.resize(3, 3);
    system.x.resize(3, 3);
    system.b.resize(3, 3);

    system.A.forEachIndex([&](size_t i, size_t j) {
        if (i > 0) {
            system.A(i, j).center += 1.0;
        }
        if (i < system.A.width() - 1) {
            system.A(i, j).center += 1.0;
            system.A(i, j).right -= 1.0;
        }

        if (j > 0) {
            system.A(i, j).center += 1.0;
        } else {
            system.b(i, j) += 1.0;
        }

        if (j < system.A.height() - 1) {
            system.A(i, j).center += 1.0;
            system.A(i, j).up -= 1.0;
        } else {
            system.b(i, j) -= 1.0;
        }
    });

    FdmIccgSolver2 solver(10, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmIccgSolver2, Solve) {
    FdmLinearSystem2 system;
    system.A.resize(128, 128);
    system.x.resize(128, 128);
    system.b.resize(128, 128);

    system.A.forEachIndex([&](size_t i, size_t j) {
        if (i > 0) {
            system.A(i, j).center += 1.0;
        }
        if (i < system.A.width() - 1) {
            system.A(i, j).center += 1.0;
            system.A(i, j).right -= 1.0;
        }

        if (j > 0) {
            system.A(i, j).center += 1.0;
        } else {
            system.b(i, j) += 1.0;
        }

        if (j < system.A.height() - 1) {
            system.A(i, j).center += 1.0;
            system.A(i, j).up -= 1.0;
        } else {
            system.b(i, j) -= 1.0;
        }
    });

    auto buffer = system.x;
    FdmBlas2::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmBlas2::l2Norm(buffer);

    FdmIccgSolver2 solver(200, 1e-4);
    EXPECT_TRUE(solver.solve(&system));
}
