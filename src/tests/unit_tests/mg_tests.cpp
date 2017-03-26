// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix_mxn.h>
#include <jet/mg.h>

#include <gtest/gtest.h>

using namespace jet;

namespace {

typedef Blas<double, VectorND, MatrixMxND> BlasType;

void relax(const typename BlasType::MatrixType& a,
           const typename BlasType::VectorType& b,
           unsigned int numberOfIterations, double maxTolerance,
           typename BlasType::VectorType* x,
           typename BlasType::VectorType* buffer) {
    (void)maxTolerance;
    (void)buffer;

    size_t n = a.rows();
    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        x->forEachIndex([&](size_t i) {
            double sum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    sum += a(j, i) * (*x)[j];
                }
            }
            (*x)[i] = (b[i] - sum) / a(i, i);
        });
    }
}

void rest(const typename BlasType::VectorType& finer,
          typename BlasType::VectorType* coarser) {
    size_t n = coarser->size();
    coarser->parallelForEachIndex([&](size_t i) {
        // --*--|--*--|--*--|--*--
        //  1/8   3/8   3/8   1/8
        //           to
        // -----|-----*-----|-----
        size_t _2im1 = (i > 0) ? 2 * i - 1 : 2 * i;
        size_t _2ip2 = (i + 1 < n) ? 2 * i + 2 : 2 * i;
        (*coarser)[i] = 0.125 * (finer[_2im1] + 3.0 * finer[2 * i] +
                                 3.0 * finer[2 * i + 1] + finer[_2ip2]);
    });
}

void corr(const typename BlasType::VectorType& coarser,
          typename BlasType::VectorType* finer) {
    size_t n = coarser.size();
    coarser.forEachIndex([&](size_t i) {
        // -----|-----*-----|-----
        //           to
        //  1/4   3/4   3/4   1/4
        // --*--|--*--|--*--|--*--
        size_t _2im1 = (i > 0) ? 2 * i - 1 : 2 * i;
        size_t _2ip2 = (i + 1 < n) ? 2 * i + 2 : 2 * i;
        (*finer)[_2im1] += 0.25 * coarser[i];
        (*finer)[2 * i + 0] += 0.75 * coarser[i];
        (*finer)[2 * i + 1] += 0.75 * coarser[i];
        (*finer)[_2ip2] += 0.25 * coarser[i];
    });
}
}

TEST(Mg, Solve) {
    MgMatrix<BlasType> A;
    MgVector<BlasType> x, b, tmp;
    MgParameters<BlasType> params;

    size_t n = 128;
    unsigned int levels = 6;

    // Build matrix
    A.levels.resize(levels);
    x.levels.resize(levels);
    b.levels.resize(levels);
    tmp.levels.resize(levels);
    for (unsigned int l = 0; l < levels; ++l) {
        size_t m = n >> l;
        A[l].resize(m, m, 0.0);
        x[l].resize(m, 0.0);
        b[l].resize(m, 0.0);
        tmp[l].resize(m, 0.0);
    }

    // Simple Poisson eq.
    for (unsigned int l = 0; l < levels; ++l) {
        size_t m = n >> l;
        double invdx = pow(0.5, l);
        auto& Al = A[l];
        auto& bl = b[l];

        for (size_t i = 0; i < m; ++i) {
            if (i > 0) {
                Al(i, i) += invdx * invdx;
                Al(i - 1, i) -= invdx * invdx;
                bl[i] += invdx;
            }
            if (i < m - 1) {
                Al(i, i) += invdx * invdx;
                Al(i + 1, i) -= invdx * invdx;
                bl[i] -= invdx;
            }
        }
    }

    // Test relax
    BlasType::residual(A[0], x[0], b[0], &tmp[0]);
    double r0 = BlasType::l2Norm(tmp[0]);

    relax(A[0], b[0], 100, 0.0, &x[0], &tmp[0]);

    BlasType::residual(A[0], x[0], b[0], &tmp[0]);
    double r1 = BlasType::l2Norm(tmp[0]);

    EXPECT_GT(r0, r1);

    // Reset solution
    x[0].set(0.0);

    // Now Mg
    params.maxNumberOfLevels = levels;
    params.relaxFunc = relax;
    params.restrictFunc = rest;
    params.correctFunc = corr;

    auto result = mgVCycle(A, params, &x, &b, &tmp);
    EXPECT_GT(r0, result.lastResidualNorm);
    EXPECT_GT(r1, result.lastResidualNorm);
}
