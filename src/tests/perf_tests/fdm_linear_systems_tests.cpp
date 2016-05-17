// Copyright (c) 2016 Doyub Kim

#include <perf_tests.h>
#include <jet/fdm_linear_system2.h>
#include <jet/fdm_linear_system3.h>
#include <jet/timer.h>
#include <gtest/gtest.h>
#include <random>

using namespace jet;

TEST(FdmBlas2, Mvm) {
    FdmMatrix2 m(1000, 1000);
    FdmVector2 a(1000, 1000), b(1000, 1000);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    m.forEachIndex([&](size_t i, size_t j) {
        m(i, j).center = d(rng);
        m(i, j).right = d(rng);
        m(i, j).up = d(rng);
        a(i, j) = d(rng);
    });

    Timer timer;

    for (int i = 0; i < 10; ++i) {
        FdmBlas2::mvm(m, a, &b);
    }

    JET_PRINT_INFO(
        "FdmBlas2::mvm avg. %f sec.\n",
        timer.durationInSeconds() / 10.0);
}

TEST(FdmBlas3, Mvm) {
    FdmMatrix3 m(200, 200, 200);
    FdmVector3 a(200, 200, 200), b(200, 200, 200);

    std::mt19937 rng;
    std::uniform_real_distribution<> d(0.0, 1.0);

    m.forEachIndex([&](size_t i, size_t j, size_t k) {
        m(i, j, k).center = d(rng);
        m(i, j, k).right = d(rng);
        m(i, j, k).up = d(rng);
        m(i, j, k).front = d(rng);
        a(i, j, k) = d(rng);
    });

    Timer timer;

    for (int i = 0; i < 10; ++i) {
        FdmBlas3::mvm(m, a, &b);
    }

    JET_PRINT_INFO(
        "FdmBlas3::mvm avg. %f sec.\n",
        timer.durationInSeconds() / 10.0);
}
