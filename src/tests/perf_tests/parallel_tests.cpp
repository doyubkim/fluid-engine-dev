// Copyright (c) 2016 Doyub Kim

#include <perf_tests.h>
#include <jet/parallel.h>
#include <jet/timer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

using namespace jet;

TEST(Parallel, For) {
    size_t N = (1 << 24) + 7;
    std::vector<double> a(N), b(N), c(N);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        a[i] = d(rng);
        b[i] = d(rng);
    }

    Timer timer;

    for (int iter = 0; iter < 20; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            c[i] = 1.0 / std::sqrt(a[i] / b[i] + 1.0);
        }
    }

    JET_PRINT_INFO(
        "serial for-loop avg. %f sec.\n",
        timer.durationInSeconds() / 20.0);

    timer.reset();

    for (int iter = 0; iter < 20; ++iter) {
        parallelFor(kZeroSize, N, [&] (size_t i) {
            c[i] = 1.0 / std::sqrt(a[i] / b[i] + 1.0);
        });
    }

    JET_PRINT_INFO(
        "parallelFor avg. %f sec.\n",
        timer.durationInSeconds() / 20.0);
}

TEST(Parallel, Sort) {
    size_t N = (1 << 20) + 7;
    std::vector<double> a(N), b(N);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        a[i] = d(rng);
        b[i] = a[i];
    }

    Timer timer;

    for (int iter = 0; iter < 20; ++iter) {
        std::sort(a.begin(), a.end());
        a = b;  // This will introduce some noise to the measurement :(
    }

    JET_PRINT_INFO(
        "std::sort avg. %f sec.\n",
        timer.durationInSeconds() / 20.0);

    timer.reset();

    for (int iter = 0; iter < 20; ++iter) {
        parallelSort(a.begin(), a.end());
        a = b;  // This will introduce some noise to the measurement :(
    }

    JET_PRINT_INFO(
        "parallelSort avg. %f sec.\n",
        timer.durationInSeconds() / 20.0);

    // Check the result
    parallelSort(a.begin(), a.end());
    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(a[i], a[i + 1]) << i;
    }
}
