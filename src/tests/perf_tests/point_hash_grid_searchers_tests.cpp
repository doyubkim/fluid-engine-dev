// Copyright (c) 2016 Doyub Kim

#include <perf_tests.h>
#include <jet/array1.h>
#include <jet/point_hash_grid_searcher3.h>
#include <jet/point_parallel_hash_grid_searcher3.h>
#include <jet/timer.h>
#include <gtest/gtest.h>
#include <random>

using namespace jet;

TEST(PointHashGridSearcher3, Build) {
    PointHashGridSearcher3 grid(64, 64, 64, 1.0 / 64.0);
    int N = 1 << 20;

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    Array1<Vector3D> points;
    for (int i = 0; i < N; ++i) {
        points.append(Vector3D(d(rng), d(rng), d(rng)));
    }

    Timer timer;

    for (int i = 0; i < 10; ++i) {
        grid.build(points);
    }

    JET_PRINT_INFO(
        "PointHashGridSearcher3::build avg. %f sec.\n",
        timer.durationInSeconds() / 10.0);
}

TEST(PointParallelHashGridSearcher3, Build) {
    PointParallelHashGridSearcher3 grid(64, 64, 64, 1.0 / 64.0);
    int N = 1 << 20;

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    Array1<Vector3D> points;
    for (int i = 0; i < N; ++i) {
        points.append(Vector3D(d(rng), d(rng), d(rng)));
    }

    Timer timer;

    for (int i = 0; i < 10; ++i) {
        grid.build(points);
    }

    JET_PRINT_INFO(
        "PointParallelHashGridSearcher3::build avg. %f sec.\n",
        timer.durationInSeconds() / 10.0);
}
