// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_mgpcg_solver3.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmMgpcgSolver3, Solve) {
    size_t levels = 4;
    FdmMgLinearSystem3 system;
    system.resizeWithCoarsest({4, 4, 4}, levels);

    // Simple Poisson eq.
    size_t n = system.A.levels.front().size().x;
    for (size_t l = 0; l < system.numberOfLevels(); ++l) {
        double invdx = pow(0.5, l);
        FdmMatrix3& A = system.A[l];
        FdmVector3& b = system.b[l];

        system.x[l].set(0);

        A.forEachIndex([&](size_t i, size_t j, size_t k) {
            if (i > 0) {
                A(i, j, k).center += invdx * invdx;
            }
            if (i < A.width() - 1) {
                A(i, j, k).center += invdx * invdx;
                A(i, j, k).right -= invdx * invdx;
            }

            if (j > 0) {
                A(i, j, k).center += invdx * invdx;
            } else {
                b(i, j, k) += invdx;
            }

            if (j < A.height() - 1) {
                A(i, j, k).center += invdx * invdx;
                A(i, j, k).up -= invdx * invdx;
            } else {
                b(i, j, k) -= invdx;
            }

            if (k > 0) {
                A(i, j, k).center += invdx * invdx;
            }
            if (k < A.depth() - 1) {
                A(i, j, k).center += invdx * invdx;
                A(i, j, k).front -= invdx * invdx;
            }
        });
    }

    auto buffer = system.x[0];
    FdmBlas3::residual(system.A[0], system.x[0], system.b[0], &buffer);
    double norm0 = FdmBlas3::l2Norm(buffer);

    FdmMgpcgSolver3 solver(50, levels, 5, 5, 10, 10, 1e-4, 1.5, false);
    EXPECT_TRUE(solver.solve(&system));
}
