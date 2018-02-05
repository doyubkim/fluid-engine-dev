// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_mgpcg_solver2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmMgpcgSolver2, Solve) {
    size_t levels = 6;
    FdmMgLinearSystem2 system;
    system.resizeWithCoarsest({4, 4}, levels);

    // Simple Poisson eq.
    for (size_t l = 0; l < system.numberOfLevels(); ++l) {
        double invdx = pow(0.5, l);
        FdmMatrix2& A = system.A[l];
        FdmVector2& b = system.b[l];

        system.x[l].set(0);

        A.forEachIndex([&](size_t i, size_t j) {
            if (i > 0) {
                A(i, j).center += invdx * invdx;
            }
            if (i < A.width() - 1) {
                A(i, j).center += invdx * invdx;
                A(i, j).right -= invdx * invdx;
            }

            if (j > 0) {
                A(i, j).center += invdx * invdx;
            } else {
                b(i, j) += invdx;
            }

            if (j < A.height() - 1) {
                A(i, j).center += invdx * invdx;
                A(i, j).up -= invdx * invdx;
            } else {
                b(i, j) -= invdx;
            }
        });
    }

    FdmMgpcgSolver2 solver(200, levels, 5, 5, 10, 10, 1e-4);
    EXPECT_TRUE(solver.solve(&system));
}
