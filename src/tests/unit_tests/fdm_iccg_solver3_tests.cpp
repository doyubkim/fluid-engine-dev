// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <gtest/gtest.h>
#include <jet/fdm_iccg_solver3.h>

using namespace jet;

TEST(FdmIccgSolver3, SolveLowRes) {
    FdmLinearSystem3 system;
    system.A.resize(3, 3, 3);
    system.x.resize(3, 3, 3);
    system.b.resize(3, 3, 3);

    system.A.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (i > 0) {
            system.A(i, j, k).center += 1.0;
        }
        if (i < system.A.width() - 1) {
            system.A(i, j, k).center += 1.0;
            system.A(i, j, k).right -= 1.0;
        }

        if (j > 0) {
            system.A(i, j, k).center += 1.0;
        } else {
            system.b(i, j, k) += 1.0;
        }

        if (j < system.A.height() - 1) {
            system.A(i, j, k).center += 1.0;
            system.A(i, j, k).up -= 1.0;
        } else {
            system.b(i, j, k) -= 1.0;
        }

        if (k > 0) {
            system.A(i, j, k).center += 1.0;
        }
        if (k < system.A.depth() - 1) {
            system.A(i, j, k).center += 1.0;
            system.A(i, j, k).front -= 1.0;
        }
    });

    FdmIccgSolver3 solver(100, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmIccgSolver3, Solve) {
    FdmLinearSystem3 system;
    system.A.resize(32, 32, 32);
    system.x.resize(32, 32, 32);
    system.b.resize(32, 32, 32);

    system.A.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (i > 0) {
            system.A(i, j, k).center += 1.0;
        }
        if (i < system.A.width() - 1) {
            system.A(i, j, k).center += 1.0;
            system.A(i, j, k).right -= 1.0;
        }

        if (j > 0) {
            system.A(i, j, k).center += 1.0;
        } else {
            system.b(i, j, k) += 1.0;
        }

        if (j < system.A.height() - 1) {
            system.A(i, j, k).center += 1.0;
            system.A(i, j, k).up -= 1.0;
        } else {
            system.b(i, j, k) -= 1.0;
        }

        if (k > 0) {
            system.A(i, j, k).center += 1.0;
        }
        if (k < system.A.depth() - 1) {
            system.A(i, j, k).center += 1.0;
            system.A(i, j, k).front -= 1.0;
        }
    });

    FdmIccgSolver3 solver(100, 1e-4);
    solver.solve(&system);

    EXPECT_TRUE(solver.solve(&system));
}
