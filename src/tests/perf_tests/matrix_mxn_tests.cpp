// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <perf_tests.h>

#include <jet/matrix_mxn.h>
#include <jet/timer.h>

#include <gtest/gtest.h>

#include <random>

using namespace jet;

TEST(MatrixMxN, Mvm) {
    MatrixMxND mat;
    VectorND x;
    VectorND y;

    std::mt19937 rng{0};
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t e = 2; e < 14; ++e) {
        size_t n = (kOneSize << e);
        mat.resize(n, n);
        x.resize(n);
        y.resize(n);
        mat.forEachIndex(
            [&mat, &rng, &d](size_t i, size_t j) { mat(i, j) = d(rng); });
        x.forEachIndex([&x, &y, &rng, &d](size_t i) {
            x[i] = d(rng);
            y[i] = d(rng);
        });

        Timer timer;
        for (int iter = 0; iter < 10; ++iter) {
            y = mat * x;
        }

        JET_PRINT_INFO("MatrixMxN mat x vec for dim %zu avg. %f sec.\n", n,
                       timer.durationInSeconds() / 10.0);
    }
}

TEST(MatrixMxN, MvmRef) {
    // http://simulationcorner.net/index.php?page=fastmatrixvector
    MatrixMxND mat;
    VectorND x;
    VectorND y;

    std::mt19937 rng{0};
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t e = 2; e < 14; ++e) {
        size_t n = (kOneSize << e);
        mat.resize(n, n);
        x.resize(n);
        y.resize(n);
        mat.forEachIndex(
            [&mat, &rng, &d](size_t i, size_t j) { mat(i, j) = d(rng); });
        x.forEachIndex([&x, &y, &rng, &d](size_t i) {
            x[i] = d(rng);
            y[i] = d(rng);
        });

        Timer timer;
        for (int iter = 0; iter < 10; ++iter) {
            // register blocking 1
            double *Apos1 = mat.data();
            double *Apos2 = mat.data() + n;
            double *ypos = y.data();

            parallelFor(kZeroSize, n / 2, [Apos1, Apos2, ypos, n, &x](size_t hi) {
                size_t i = 2 * hi;
                double *Apos1_ = Apos1 + i * n;
                double *Apos2_ = Apos2 + i * n;
                double *ypos_ = ypos + i;

                double ytemp1 = 0;
                double ytemp2 = 0;
                double *xpos = x.data();

                for (size_t j = 0; j < n; j++) {
                    ytemp1 += (*Apos1_++) * (*xpos);
                    ytemp2 += (*Apos2_++) * (*xpos);

                    xpos++;
                }
                *ypos_ = ytemp1;
                ypos_++;
                *ypos_ = ytemp2;
            });
        }

        JET_PRINT_INFO("MatrixMxN mat x vec for dim %zu avg. %f sec.\n", n,
                       timer.durationInSeconds() / 10.0);
    }
}
