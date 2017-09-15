// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_cg_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(FdmCgSolver3, Constructors) {
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

    FdmCgSolver3 solver(100, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}
