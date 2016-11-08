// Copyright (c) 2016 Doyub Kim

#include <jet/array2.h>
#include <jet/array3.h>
#include <jet/constants.h>
#include <jet/parallel.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using namespace jet;

static unsigned int sNumCores = std::thread::hardware_concurrency();

TEST(Parallel, Fill) {
    size_t N = std::max(20u, (3 * sNumCores) / 2);
    std::vector<double> a(N);

    parallelFill(a.begin(), a.end(), 3.0);

    for (double val : a) {
        EXPECT_EQ(3.0, val);
    }
}

TEST(Parallel, For) {
    size_t N = std::max(20u, (3 * sNumCores) / 2);
    std::vector<double> a(N);

    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<double>(i);
    }

    parallelFor(kZeroSize, a.size(), [&a] (size_t i) {
        double expected = static_cast<double>(i);
        EXPECT_DOUBLE_EQ(expected, a[i]);
    });
}

TEST(Parallel, For2D) {
    size_t nX = std::max(20u, (3 * sNumCores) / 2);
    size_t nY = std::max(30u, (3 * sNumCores) / 2);
    Array2<double> a(nX, nY);

    for (size_t j = 0; j < nY; ++j) {
        for (size_t i = 0; i < nX; ++i) {
            a(i, j) = static_cast<double>(i + j * nX);
        }
    }

    parallelFor(
        kZeroSize, a.width(),
        kZeroSize, a.height(),
        [&] (size_t i, size_t j) {
        double expected = static_cast<double>(i + j * nX);
        EXPECT_DOUBLE_EQ(expected, a(i, j));
    });
}

TEST(Parallel, For3D) {
    size_t nX = std::max(20u, (3 * sNumCores) / 2);
    size_t nY = std::max(30u, (3 * sNumCores) / 2);
    size_t nZ = std::max(30u, (3 * sNumCores) / 2);
    Array3<double> a(nX, nY, nZ);

    for (size_t k = 0; k < nZ; ++k) {
        for (size_t j = 0; j < nY; ++j) {
            for (size_t i = 0; i < nX; ++i) {
                a(i, j, k) = static_cast<double>(i + (j + k * nY) * nX);
            }
        }
    }

    parallelFor(
        kZeroSize, a.width(),
        kZeroSize, a.height(),
        kZeroSize, a.depth(),
        [&] (size_t i, size_t j, size_t k) {
        double expected = static_cast<double>(i + (j + k * nY) * nX);
        EXPECT_DOUBLE_EQ(expected, a(i, j, k));
    });
}

TEST(Parallel, Sort) {
    size_t N = std::max(20u, (3 * sNumCores) / 2);
    std::vector<double> a(N);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        a[i] = d(rng);
    }

    parallelSort(a.begin(), a.end());

    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(a[i], a[i + 1]) << i;
    }

    std::vector<double> b(N);
    for (size_t i = 0; i < N; ++i) {
        b[i] = d(rng);
    }

    std::vector<double> c = b;

    parallelSort(b.begin(), b.end());

    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(b[i], b[i + 1]) << i;
    }

    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) {
        idx[i] = i;
    }

    parallelSort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return c[a] < c[b];
    });

    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(c[idx[i]], c[idx[i + 1]]);
    }
}
