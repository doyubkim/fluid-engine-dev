// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array2.h>
#include <jet/array3.h>
#include <jet/constants.h>
#include <jet/serial.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using namespace jet;

TEST(Serial, Fill) {
    size_t N = 100;
    std::vector<double> a(N);

    serialFill(a.begin(), a.end(), 3.0);

    for (double val : a) {
        EXPECT_EQ(3.0, val);
    }
}

TEST(Serial, For) {
    size_t N = 100;
    std::vector<double> a(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i);
    }

    serialFor(kZeroSize, a.size(), [&a] (size_t i) {
        double expected = static_cast<double>(i);
        EXPECT_DOUBLE_EQ(expected, a[i]);
    });
}

TEST(Serial, For2D) {
    size_t nX = 100;
    size_t nY = 100;
    Array2<double> a(nX, nY);

    for (size_t j = 0; j < nY; ++j) {
        for (size_t i = 0; i < nX; ++i) {
            a(i, j) = static_cast<double>(i + j * nX);
        }
    }

    serialFor(
        kZeroSize, a.width(),
        kZeroSize, a.height(),
        [&] (size_t i, size_t j) {
        double expected = static_cast<double>(i + j * nX);
        EXPECT_DOUBLE_EQ(expected, a(i, j));
    });
}

TEST(Serial, For3D) {
    size_t nX = 100;
    size_t nY = 100;
    size_t nZ = 100;
    Array3<double> a(nX, nY, nZ);

    for (size_t k = 0; k < nZ; ++k) {
        for (size_t j = 0; j < nY; ++j) {
            for (size_t i = 0; i < nX; ++i) {
                a(i, j, k) = static_cast<double>(i + (j + k * nY) * nX);
            }
        }
    }

    serialFor(
        kZeroSize, a.width(),
        kZeroSize, a.height(),
        kZeroSize, a.depth(),
        [&] (size_t i, size_t j, size_t k) {
        double expected = static_cast<double>(i + (j + k * nY) * nX);
        EXPECT_DOUBLE_EQ(expected, a(i, j, k));
    });
}

TEST(Serial, Sort) {
    size_t N = 100;
    std::vector<double> a(N);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        a[i] = d(rng);
    }

    serialSort(a.begin(), a.end());

    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(a[i], a[i + 1]) << i;
    }

    std::vector<double> b(N);
    for (size_t i = 0; i < N; ++i) {
        b[i] = d(rng);
    }

    std::vector<double> c = b;

    serialSort(b.begin(), b.end());

    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(b[i], b[i + 1]) << i;
    }

    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) {
        idx[i] = i;
    }

    serialSort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return c[a] < c[b];
    });

    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(c[idx[i]], c[idx[i + 1]]);
    }
}
