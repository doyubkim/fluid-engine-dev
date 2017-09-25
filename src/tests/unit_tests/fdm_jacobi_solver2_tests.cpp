// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_jacobi_solver2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmJacobiSolver2, Solve) {
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

    FdmJacobiSolver2 solver(100, 10, 1e-9);
    solver.solve(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmJacobiSolver2, SolveCompressed) {
    FdmCompressedLinearSystem2 system;
    system.coordToIndex.resize(3, 3);

    const auto acc = system.coordToIndex.constAccessor();
    Size2 size = acc.size();

    system.coordToIndex.forEachIndex([&](size_t i, size_t j) {
        const size_t cIdx = acc.index(i, j);
        const size_t lIdx = acc.index(i - 1, j);
        const size_t rIdx = acc.index(i + 1, j);
        const size_t dIdx = acc.index(i, j - 1);
        const size_t uIdx = acc.index(i, j + 1);

        system.coordToIndex[cIdx] = system.b.size();
        system.indexToCoord.append({i, j});
        double bij = 0.0;

        std::vector<double> row(1, 0.0);
        std::vector<size_t> colIdx(1, cIdx);

        if (i > 0) {
            row[0] += 1.0;
            row.push_back(-1.0);
            colIdx.push_back(lIdx);
        }
        if (i < size.x - 1) {
            row[0] += 1.0;
            row.push_back(-1.0);
            colIdx.push_back(rIdx);
        }

        if (j > 0) {
            row[0] += 1.0;
            row.push_back(-1.0);
            colIdx.push_back(dIdx);
        } else {
            bij += 1.0;
        }

        if (j < size.y - 1) {
            row[0] += 1.0;
            row.push_back(-1.0);
            colIdx.push_back(uIdx);
        } else {
            bij -= 1.0;
        }

        system.A.addRow(row, colIdx);
        system.b.append(bij);
    });

    system.x.resize(system.b.size(), 0.0);

    FdmJacobiSolver2 solver(100, 10, 1e-9);
    solver.solveCompressed(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}
