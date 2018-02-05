// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/constants.h>
#include <jet/samplers.h>
#include <gtest/gtest.h>
#include <random>

using namespace jet;

TEST(Samplers, UniformSampleCone) {
    std::mt19937 mtRand(0);
    std::uniform_real_distribution<double> dist(0, 1);

    for (int i = 0; i < 100; ++i) {
        double u1 = dist(mtRand);
        double u2 = dist(mtRand);

        Vector3D pt = uniformSampleCone(u1, u2, Vector3D(1, 0, 0), 0.5);

        double dot = pt.dot(Vector3D(1, 0, 0));
        EXPECT_LE(std::cos(0.5), dot);

        double d = pt.length();
        EXPECT_DOUBLE_EQ(1.0, d);
    }
}

TEST(Samplers, UniformSampleHemisphere) {
    std::mt19937 mtRand(0);
    std::uniform_real_distribution<double> dist(0, 1);

    for (int i = 0; i < 100; ++i) {
        double u1 = dist(mtRand);
        double u2 = dist(mtRand);

        Vector3D pt = uniformSampleHemisphere(u1, u2, Vector3D(1, 0, 0));

        double dot = pt.dot(Vector3D(1, 0, 0));
        EXPECT_LE(std::cos(kHalfPiD), dot);

        double d = pt.length();
        EXPECT_DOUBLE_EQ(1.0, d);
    }
}

TEST(Samplers, UniformSampleSphere) {
    std::mt19937 mtRand(0);
    std::uniform_real_distribution<double> dist(0, 1);

    for (int i = 0; i < 100; ++i) {
        double u1 = dist(mtRand);
        double u2 = dist(mtRand);

        Vector3D pt = uniformSampleSphere(u1, u2);

        double d = pt.length();
        EXPECT_DOUBLE_EQ(1.0, d);
    }
}

TEST(Samplers, UniformSampleDisk) {
    std::mt19937 mtRand(0);
    std::uniform_real_distribution<double> dist(0, 1);

    for (int i = 0; i < 100; ++i) {
        double u1 = dist(mtRand);
        double u2 = dist(mtRand);

        Vector2D pt = uniformSampleDisk(u1, u2);

        double d = pt.length();
        EXPECT_GE(1.0, d);
    }
}
