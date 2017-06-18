// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <gtest/gtest.h>
#include <jet/fdm_gauss_seidel_solver2.h>

using namespace jet;

TEST(FdmGaussSeidelSolver2, SolveLowRes) {
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

    FdmGaussSeidelSolver2 solver(100, 10, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmGaussSeidelSolver2, Solve) {
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

    FdmGaussSeidelSolver2 solver(100, 10, 1e-9);
    solver.solve(&system);

    FdmBlas2::residual(system.A, system.x, system.b, &buffer);
    double norm1 = FdmBlas2::l2Norm(buffer);

    EXPECT_LT(norm1, norm0);
}

TEST(FdmGaussSeidelSolver2, Relax) {
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

    for (int i = 0; i < 200; ++i) {
        FdmGaussSeidelSolver2::relax(system.A, system.b, 1.0, &system.x);

        FdmBlas2::residual(system.A, system.x, system.b, &buffer);
        double norm = FdmBlas2::l2Norm(buffer);
        EXPECT_LT(norm, norm0);

        norm0 = norm;
    }
}

TEST(FdmGaussSeidelSolver2, RelaxRedBlack) {
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

    for (int i = 0; i < 200; ++i) {
        FdmGaussSeidelSolver2::relaxRedBlack(system.A, system.b, 1.0,
                                             &system.x);

        FdmBlas2::residual(system.A, system.x, system.b, &buffer);
        double norm = FdmBlas2::l2Norm(buffer);
        EXPECT_LT(norm, norm0);

        norm0 = norm;
    }
}
