// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <perf_tests.h>

#include <jet/parallel.h>
#include <jet/timer.h>

#include <gtest/gtest.h>

#ifdef COMPARE_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#endif

#include <numeric>
#include <random>

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

    JET_PRINT_INFO("serial for-loop avg. %f sec.\n",
                   timer.durationInSeconds() / 20.0);

    timer.reset();

    for (int iter = 0; iter < 20; ++iter) {
        parallelFor(kZeroSize, N, [&](size_t i) {
            c[i] = 1.0 / std::sqrt(a[i] / b[i] + 1.0);
        });
    }

    JET_PRINT_INFO("parallelFor avg. %f sec.\n",
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

    JET_PRINT_INFO("std::sort avg. %f sec.\n",
                   timer.durationInSeconds() / 20.0);

    timer.reset();

    for (int iter = 0; iter < 20; ++iter) {
        parallelSort(a.begin(), a.end());
        a = b;  // This will introduce some noise to the measurement :(
    }

    JET_PRINT_INFO("parallelSort avg. %f sec.\n",
                   timer.durationInSeconds() / 20.0);

    // Check the result
    parallelSort(a.begin(), a.end());
    for (size_t i = 0; i + 1 < a.size(); ++i) {
        EXPECT_LE(a[i], a[i + 1]) << i;
    }
}

TEST(Parallel, Reduce) {
    size_t N = (1 << 24) + 7;
    std::vector<double> a(N);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    for (size_t i = 0; i < N; ++i) {
        a[i] = d(rng);
    }

    auto func = [&](size_t start, size_t end, double init) {
        double result = init;
        for (size_t i = start; i < end; ++i) {
            result += a[i];
        }
        return result;
    };

    Timer timer;

    double sum = 0.0;
    for (int iter = 0; iter < 20; ++iter) {
        sum += std::accumulate(a.begin(), a.end(), 0.0);
    }

    JET_PRINT_INFO("serial reduce avg. %f sec.\n",
                   timer.durationInSeconds() / 20.0);

    timer.reset();

    sum = 0.0;
    for (int iter = 0; iter < 20; ++iter) {
        sum +=
            parallelReduce(kZeroSize, a.size(), 0.0, func, std::plus<double>());
    }

    JET_PRINT_INFO("parallelReduce avg. %f sec.\n",
                   timer.durationInSeconds() / 20.0);
#ifdef COMPARE_WITH_TBB
    timer.reset();

    sum = 0.0;
    for (int iter = 0; iter < 20; ++iter) {
        sum += tbb::parallel_reduce(
            tbb::blocked_range<double*>(a.data(), a.data() + a.size()), 0.0,
            [&](const tbb::blocked_range<double*>& r, double init) {
                return std::accumulate(r.begin(), r.end(), init);
            },
            std::plus<double>());
    }

    JET_PRINT_INFO("parallelReduce avg. %f sec.\n",
                   timer.durationInSeconds() / 20.0);
#endif
}
