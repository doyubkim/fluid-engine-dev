// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/array2.h>
#include <jet/array3.h>
#include <jet/array_samplers1.h>
#include <jet/array_samplers2.h>
#include <jet/array_samplers3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(NearestArraySampler1, Sample) {
    {
        Array1<double> grid({ 1.0, 2.0, 3.0, 4.0 });
        double gridSpacing = 1.0, gridOrigin = 0.0;
        NearestArraySampler1<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(0.45);
        EXPECT_LT(std::fabs(s0 - 1.0), 1e-9);

        double s1 = sampler(1.57);
        EXPECT_LT(std::fabs(s1 - 3.0), 1e-9);

        double s2 = sampler(3.51);
        EXPECT_LT(std::fabs(s2 - 4.0), 1e-9);
    }
    {
        Array1<double> grid({ 1.0, 2.0, 3.0, 4.0 });
        double gridSpacing = 0.5, gridOrigin = -1.0;
        NearestArraySampler1<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(0.45);
        EXPECT_LT(std::fabs(s0 - 4.0), 1e-9);

        double s1 = sampler(-0.05);
        EXPECT_LT(std::fabs(s1 - 3.0), 1e-9);
    }
}

TEST(LinearArraySampler1, Sample) {
    {
        Array1<double> grid({ 1.0, 2.0, 3.0, 4.0 });
        double gridSpacing = 1.0, gridOrigin = 0.0;
        LinearArraySampler1<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(0.5);
        EXPECT_LT(std::fabs(s0 - 1.5), 1e-9);

        double s1 = sampler(1.8);
        EXPECT_LT(std::fabs(s1 - 2.8), 1e-9);

        double s2 = sampler(3.5);
        EXPECT_NEAR(4.0, s2, 1e-9);
    }
    {
        Array1<double> grid({ 1.0, 2.0, 3.0, 4.0 });
        double gridSpacing = 0.5, gridOrigin = -1.0;
        LinearArraySampler1<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(0.2);
        EXPECT_LT(std::fabs(s0 - 3.4), 1e-9);

        double s1 = sampler(-0.7);
        EXPECT_LT(std::fabs(s1 - 1.6), 1e-9);
    }
}

TEST(CubicArraySampler1, Sample) {
    Array1<double> grid({ 1.0, 2.0, 3.0, 4.0 });
    double gridSpacing = 1.0, gridOrigin = 0.0;
    CubicArraySampler1<double, double> sampler(
        grid.constAccessor(), gridSpacing, gridOrigin);

    double s0 = sampler(1.25);
    EXPECT_LT(2.0, s0);
    EXPECT_GT(3.0, s0);
}

TEST(NearestArraySampler2, Sample) {
    {
        Array2<double> grid(
            {{ 1.0, 2.0, 3.0, 4.0 },
             { 2.0, 3.0, 4.0, 5.0 },
             { 3.0, 4.0, 5.0, 6.0 },
             { 4.0, 5.0, 6.0, 7.0 },
             { 5.0, 6.0, 7.0, 8.0 }});
        Vector2D gridSpacing(1.0, 1.0), gridOrigin;
        NearestArraySampler2<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(Vector2D(0.45, 0.45));
        EXPECT_LT(std::fabs(s0 - 1.0), 1e-9);

        double s1 = sampler(Vector2D(1.57, 4.01));
        EXPECT_LT(std::fabs(s1 - 7.0), 1e-9);

        double s2 = sampler(Vector2D(3.50, 1.21));
        EXPECT_LT(std::fabs(s2 - 5.0), 1e-9);
    }
    {
        Array2<double> grid(
            {{ 1.0, 2.0, 3.0, 4.0 },
             { 2.0, 3.0, 4.0, 5.0 },
             { 3.0, 4.0, 5.0, 6.0 },
             { 4.0, 5.0, 6.0, 7.0 },
             { 5.0, 6.0, 7.0, 8.0 }});
        Vector2D gridSpacing(0.5, 0.25), gridOrigin(-1.0, -0.5);
        NearestArraySampler2<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(Vector2D(0.45, 0.4));
        EXPECT_LT(std::fabs(s0 - 8.0), 1e-9);

        double s1 = sampler(Vector2D(-0.05, 0.37));
        EXPECT_LT(std::fabs(s1 - 6.0), 1e-9);
    }
}

TEST(LinearArraySampler2, Sample) {
    {
        Array2<double> grid(
            {{ 1.0, 2.0, 3.0, 4.0 },
             { 2.0, 3.0, 4.0, 5.0 },
             { 3.0, 4.0, 5.0, 6.0 },
             { 4.0, 5.0, 6.0, 7.0 },
             { 5.0, 6.0, 7.0, 8.0 } });
        Vector2D gridSpacing(1.0, 1.0), gridOrigin;
        LinearArraySampler2<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(Vector2D(0.5, 0.5));
        EXPECT_LT(std::fabs(s0 - 2.0), 1e-9);

        double s1 = sampler(Vector2D(1.5, 4.0));
        EXPECT_LT(std::fabs(s1 - 6.5), 1e-9);
    }
    {
        Array2<double> grid(
            {{ 1.0, 2.0, 3.0, 4.0 },
             { 2.0, 3.0, 4.0, 5.0 },
             { 3.0, 4.0, 5.0, 6.0 },
             { 4.0, 5.0, 6.0, 7.0 },
             { 5.0, 6.0, 7.0, 8.0 }});
        Vector2D gridSpacing(0.5, 0.25), gridOrigin(-1.0, -0.5);
        LinearArraySampler2<double, double> sampler(
            grid.constAccessor(), gridSpacing, gridOrigin);

        double s0 = sampler(Vector2D(0.5, 0.5));
        EXPECT_LT(std::fabs(s0 - 8.0), 1e-9);

        double s1 = sampler(Vector2D(-0.5, 0.375));
        EXPECT_LT(std::fabs(s1 - 5.5), 1e-9);
    }
}

TEST(CubicArraySampler2, Sample) {
    Array2<double> grid(
        {{ 1.0, 2.0, 3.0, 4.0 },
         { 2.0, 3.0, 4.0, 5.0 },
         { 3.0, 4.0, 5.0, 6.0 },
         { 4.0, 5.0, 6.0, 7.0 },
         { 5.0, 6.0, 7.0, 8.0 }});
    Vector2D gridSpacing(1.0, 1.0), gridOrigin;
    CubicArraySampler2<double, double> sampler(
        grid.constAccessor(), gridSpacing, gridOrigin);

    double s0 = sampler(Vector2D(1.5, 2.8));
    EXPECT_LT(4.0, s0);
    EXPECT_GT(6.0, s0);
}

TEST(CubicArraySampler3, Sample) {
    Array3<double> grid(4, 4, 4);
    for (size_t k = 0; k < 4; ++k) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t i = 0; i < 4; ++i) {
                grid(i, j, k) = static_cast<double>(i + j + k);
            }
        }
    }

    Vector3D gridSpacing(1.0, 1.0, 1.0), gridOrigin;
    CubicArraySampler3<double, double> sampler(
        grid.constAccessor(), gridSpacing, gridOrigin);

    double s0 = sampler(Vector3D(1.5, 1.8, 1.2));
    EXPECT_LT(3.0, s0);
    EXPECT_GT(6.0, s0);
}
