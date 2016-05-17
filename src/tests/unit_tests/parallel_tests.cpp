// Copyright (c) 2016 Doyub Kim

#include <jet/parallel.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using namespace jet;

TEST(Parallel, Sort) {
    size_t N = 20;
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
