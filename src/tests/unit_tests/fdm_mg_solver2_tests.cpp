// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_mg_solver2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmMgSolver2, Solve) {
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

    auto buffer = system.x[0];
    FdmBlas2::residual(system.A[0], system.x[0], system.b[0], &buffer);
    double norm0 = FdmBlas2::l2Norm(buffer);

    FdmMgSolver2 solver(levels, 5, 5, 20, 20, 1e-9);
    solver.solve(&system);

    FdmBlas2::residual(system.A[0], system.x[0], system.b[0], &buffer);
    double norm1 = FdmBlas2::l2Norm(buffer);

    EXPECT_LT(norm1, norm0);
}
