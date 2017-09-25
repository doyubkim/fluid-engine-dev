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

TEST(FdmGaussSeidelSolver2, SolveCompressedRes) {
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

    FdmGaussSeidelSolver2 solver(100, 10, 1e-9);
    solver.solveCompressed(&system);

    EXPECT_GT(solver.tolerance(), solver.lastResidual());
}

TEST(FdmGaussSeidelSolver2, SolveCompressed) {
    FdmCompressedLinearSystem2 system;
    system.coordToIndex.resize(128, 128);

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

    auto buffer = system.x;
    FdmCompressedBlas2::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmCompressedBlas2::l2Norm(buffer);

    FdmGaussSeidelSolver2 solver(100, 10, 1e-9);
    solver.solveCompressed(&system);

    FdmCompressedBlas2::residual(system.A, system.x, system.b, &buffer);
    double norm1 = FdmCompressedBlas2::l2Norm(buffer);

    EXPECT_LT(norm1, norm0);
}

TEST(FdmGaussSeidelSolver2, RelaxRedBlackCompressed) {
    FdmCompressedLinearSystem2 system;
    system.coordToIndex.resize(128, 128);

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

    auto buffer = system.x;
    FdmCompressedBlas2::residual(system.A, system.x, system.b, &buffer);
    double norm0 = FdmCompressedBlas2::l2Norm(buffer);

    for (int i = 0; i < 200; ++i) {
        FdmGaussSeidelSolver2::relaxRedBlack(
            system.A, system.b, system.indexToCoord, 1.0, &system.x);

        FdmCompressedBlas2::residual(system.A, system.x, system.b, &buffer);
        double norm = FdmCompressedBlas2::l2Norm(buffer);
        EXPECT_LT(norm, norm0);

        norm0 = norm;
    }
}
