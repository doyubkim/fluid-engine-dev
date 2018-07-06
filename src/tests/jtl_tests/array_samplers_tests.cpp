// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "jtl_tests_utils.h"

#include <jet/array.h>
#include <jet/array_samplers.h>

#include <gtest/gtest.h>

using namespace jet;

static constexpr double kEps = 1e-10;

TEST(NearestArraySampler1, Sample) {
    {
        Array1<double> grid({1.0, 2.0, 3.0, 4.0});
        double gridSpacing = 1.0, gridOrigin = 0.0;
        NearestArraySampler1<double> sampler(grid.view(), gridSpacing,
                                             gridOrigin);

        double s0 = sampler(0.45);
        EXPECT_LT(std::fabs(s0 - 1.0), 1e-9);

        double s1 = sampler(1.57);
        EXPECT_LT(std::fabs(s1 - 3.0), 1e-9);

        double s2 = sampler(3.51);
        EXPECT_LT(std::fabs(s2 - 4.0), 1e-9);
    }
    {
        Array1<double> grid({1.0, 2.0, 3.0, 4.0});
        double gridSpacing = 0.5, gridOrigin = -1.0;
        NearestArraySampler1<double> sampler(grid.view(), gridSpacing,
                                             gridOrigin);

        double s0 = sampler(0.45);
        EXPECT_LT(std::fabs(s0 - 4.0), 1e-9);

        double s1 = sampler(-0.05);
        EXPECT_LT(std::fabs(s1 - 3.0), 1e-9);
    }
}

TEST(LinearArraySampler1, Sample) {
    {
        Array1<double> grid({1.0, 2.0, 3.0, 4.0});
        double gridSpacing = 1.0, gridOrigin = 0.0;
        LinearArraySampler1<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        double s0 = sampler(0.5);
        EXPECT_LT(std::fabs(s0 - 1.5), 1e-9);

        double s1 = sampler(1.8);
        EXPECT_LT(std::fabs(s1 - 2.8), 1e-9);

        double s2 = sampler(3.5);
        EXPECT_NEAR(4.0, s2, 1e-9);
    }
    {
        Array1<double> grid({1.0, 2.0, 3.0, 4.0});
        double gridSpacing = 0.5, gridOrigin = -1.0;
        LinearArraySampler1<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        double s0 = sampler(0.2);
        EXPECT_LT(std::fabs(s0 - 3.4), 1e-9);

        double s1 = sampler(-0.7);
        EXPECT_LT(std::fabs(s1 - 1.6), 1e-9);
    }
}

TEST(CubicArraySampler1, Sample) {
    Array1<double> grid({1.0, 2.0, 3.0, 4.0});
    double gridSpacing = 1.0, gridOrigin = 0.0;
    MonotonicCatmullRomArraySampler1<double> sampler(grid.view(), gridSpacing,
                                                     gridOrigin);

    double s0 = sampler(1.25);
    EXPECT_LT(2.0, s0);
    EXPECT_GT(3.0, s0);
}

TEST(NearestArraySampler2, Sample) {
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(1.0, 1.0), gridOrigin;
        NearestArraySampler2<double> sampler(grid.view(), gridSpacing,
                                             gridOrigin);

        double s0 = sampler(Vector2D(0.45, 0.45));
        EXPECT_NEAR(1.0, s0, kEps);

        double s1 = sampler(Vector2D(1.57, 4.01));
        EXPECT_NEAR(7.0, s1, kEps);

        double s2 = sampler(Vector2D(3.50, 1.21));
        EXPECT_NEAR(5.0, s2, kEps);
    }
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(0.5, 0.25), gridOrigin(-1.0, -0.5);
        NearestArraySampler2<double> sampler(grid.view(), gridSpacing,
                                             gridOrigin);

        double s0 = sampler(Vector2D(0.45, 0.4));
        EXPECT_NEAR(8.0, s0, kEps);

        double s1 = sampler(Vector2D(-0.05, 0.37));
        EXPECT_NEAR(6.0, s1, kEps);
    }
}

TEST(LinearArraySampler2, Sample) {
    // From simple grid
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(1.0, 1.0), gridOrigin;
        LinearArraySampler2<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        double s0 = sampler(Vector2D(0.5, 0.5));
        EXPECT_NEAR(s0, 2.0, kEps);

        double s1 = sampler(Vector2D(1.5, 4.0));
        EXPECT_NEAR(s1, 6.5, kEps);
    }

    // From more complex grid
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(0.5, 0.25), gridOrigin(-1.0, -0.5);
        LinearArraySampler2<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        double s0 = sampler(Vector2D(0.5, 0.5));
        EXPECT_LT(std::fabs(s0 - 8.0), kEps);

        double s1 = sampler(Vector2D(-0.5, 0.375));
        EXPECT_LT(std::fabs(s1 - 5.5), kEps);
    }
}

TEST(LinearArraySampler2, GetCoordinatesAndWeights) {
    // From simple grid
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(1.0, 1.0), gridOrigin;
        LinearArraySampler2<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        std::array<Vector2UZ, 4> indices;
        std::array<double, 4> weights;
        sampler.getCoordinatesAndWeights(Vector2D(0.5, 0.5), indices, weights);
        EXPECT_EQ(indices[0], Vector2UZ(0, 0));
        EXPECT_EQ(indices[1], Vector2UZ(1, 0));
        EXPECT_EQ(indices[2], Vector2UZ(0, 1));
        EXPECT_EQ(indices[3], Vector2UZ(1, 1));
        EXPECT_NEAR(weights[0], 0.25, kEps);
        EXPECT_NEAR(weights[1], 0.25, kEps);
        EXPECT_NEAR(weights[2], 0.25, kEps);
        EXPECT_NEAR(weights[3], 0.25, kEps);

        sampler.getCoordinatesAndWeights(Vector2D(1.3, 4.0), indices, weights);
        EXPECT_EQ(indices[0], Vector2UZ(1, 3));
        EXPECT_EQ(indices[1], Vector2UZ(2, 3));
        EXPECT_EQ(indices[2], Vector2UZ(1, 4));
        EXPECT_EQ(indices[3], Vector2UZ(2, 4));
        EXPECT_NEAR(weights[0], 0.0, kEps);
        EXPECT_NEAR(weights[1], 0.0, kEps);
        EXPECT_NEAR(weights[2], 0.7, kEps);
        EXPECT_NEAR(weights[3], 0.3, kEps);
    }

    // From more complex grid
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(0.5, 0.25), gridOrigin(-1.0, -0.5);
        LinearArraySampler2<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        std::array<Vector2UZ, 4> indices;
        std::array<double, 4> weights;
        sampler.getCoordinatesAndWeights(Vector2D(0.5, 0.5), indices, weights);
        EXPECT_EQ(indices[0], Vector2UZ(2, 3));
        EXPECT_EQ(indices[1], Vector2UZ(3, 3));
        EXPECT_EQ(indices[2], Vector2UZ(2, 4));
        EXPECT_EQ(indices[3], Vector2UZ(3, 4));
        EXPECT_NEAR(weights[0], 0.0, kEps);
        EXPECT_NEAR(weights[1], 0.0, kEps);
        EXPECT_NEAR(weights[2], 0.0, kEps);
        EXPECT_NEAR(weights[3], 1.0, kEps);

        sampler.getCoordinatesAndWeights(Vector2D(-0.5, 0.375), indices,
                                         weights);
        EXPECT_EQ(indices[0], Vector2UZ(1, 3));
        EXPECT_EQ(indices[1], Vector2UZ(2, 3));
        EXPECT_EQ(indices[2], Vector2UZ(1, 4));
        EXPECT_EQ(indices[3], Vector2UZ(2, 4));
        EXPECT_NEAR(weights[0], 0.5, kEps);
        EXPECT_NEAR(weights[1], 0.0, kEps);
        EXPECT_NEAR(weights[2], 0.5, kEps);
        EXPECT_NEAR(weights[3], 0.0, kEps);
    }
}

TEST(LinearArraySampler2, GetCoordinatesAndGradientWeights) {
    // From simple grid
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(1.0, 1.0), gridOrigin;
        LinearArraySampler2<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        std::array<Vector2UZ, 4> indices;
        std::array<Vector2D, 4> weights;
        sampler.getCoordinatesAndGradientWeights(Vector2D(0.5, 0.5), indices,
                                                 weights);
        EXPECT_EQ(indices[0], Vector2UZ(0, 0));
        EXPECT_EQ(indices[1], Vector2UZ(1, 0));
        EXPECT_EQ(indices[2], Vector2UZ(0, 1));
        EXPECT_EQ(indices[3], Vector2UZ(1, 1));
        EXPECT_VECTOR2_NEAR(weights[0], Vector2D(-0.5, -0.5), kEps);
        EXPECT_VECTOR2_NEAR(weights[1], Vector2D(+0.5, -0.5), kEps);
        EXPECT_VECTOR2_NEAR(weights[2], Vector2D(-0.5, +0.5), kEps);
        EXPECT_VECTOR2_NEAR(weights[3], Vector2D(+0.5, +0.5), kEps);

        sampler.getCoordinatesAndGradientWeights(Vector2D(1.3, 4.0), indices,
                                                 weights);
        EXPECT_EQ(indices[0], Vector2UZ(1, 3));
        EXPECT_EQ(indices[1], Vector2UZ(2, 3));
        EXPECT_EQ(indices[2], Vector2UZ(1, 4));
        EXPECT_EQ(indices[3], Vector2UZ(2, 4));
        EXPECT_VECTOR2_NEAR(weights[0], Vector2D(0.0, -0.7), kEps);
        EXPECT_VECTOR2_NEAR(weights[1], Vector2D(0.0, -0.3), kEps);
        EXPECT_VECTOR2_NEAR(weights[2], Vector2D(-1.0, 0.7), kEps);
        EXPECT_VECTOR2_NEAR(weights[3], Vector2D(1.0, 0.3), kEps);
    }

    // From more complex grid
    {
        Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                             {2.0, 3.0, 4.0, 5.0},
                             {3.0, 4.0, 5.0, 6.0},
                             {4.0, 5.0, 6.0, 7.0},
                             {5.0, 6.0, 7.0, 8.0}});
        Vector2D gridSpacing(0.5, 0.25), gridOrigin(-1.0, -0.5);
        LinearArraySampler2<double> sampler(grid.view(), gridSpacing,
                                            gridOrigin);

        std::array<Vector2UZ, 4> indices;
        std::array<Vector2D, 4> weights;
        sampler.getCoordinatesAndGradientWeights(Vector2D(0.5, 0.5), indices,
                                                 weights);
        EXPECT_EQ(indices[0], Vector2UZ(2, 3));
        EXPECT_EQ(indices[1], Vector2UZ(3, 3));
        EXPECT_EQ(indices[2], Vector2UZ(2, 4));
        EXPECT_EQ(indices[3], Vector2UZ(3, 4));
        EXPECT_VECTOR2_NEAR(weights[0], Vector2D(0.0, 0.0), kEps);
        EXPECT_VECTOR2_NEAR(weights[1], Vector2D(0.0, -4.0), kEps);
        EXPECT_VECTOR2_NEAR(weights[2], Vector2D(-2.0, 0.0), kEps);
        EXPECT_VECTOR2_NEAR(weights[3], Vector2D(2.0, 4.0), kEps);

        sampler.getCoordinatesAndGradientWeights(Vector2D(-0.5, 0.375), indices,
                                                 weights);
        EXPECT_EQ(indices[0], Vector2UZ(1, 3));
        EXPECT_EQ(indices[1], Vector2UZ(2, 3));
        EXPECT_EQ(indices[2], Vector2UZ(1, 4));
        EXPECT_EQ(indices[3], Vector2UZ(2, 4));  // fx=0 fy=0.5
        EXPECT_VECTOR2_NEAR(weights[0], Vector2D(-1.0, -4.0), kEps);
        EXPECT_VECTOR2_NEAR(weights[1], Vector2D(1.0, 0.0), kEps);
        EXPECT_VECTOR2_NEAR(weights[2], Vector2D(-1.0, 4.0), kEps);
        EXPECT_VECTOR2_NEAR(weights[3], Vector2D(1.0, 0.0), kEps);
    }
}

TEST(CubicArraySampler2, Sample) {
    Array2<double> grid({{1.0, 2.0, 3.0, 4.0},
                         {2.0, 3.0, 4.0, 5.0},
                         {3.0, 4.0, 5.0, 6.0},
                         {4.0, 5.0, 6.0, 7.0},
                         {5.0, 6.0, 7.0, 8.0}});
    Vector2D gridSpacing(1.0, 1.0), gridOrigin;
    MonotonicCatmullRomArraySampler2<double> sampler(grid.view(), gridSpacing,
                                                     gridOrigin);

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
    MonotonicCatmullRomArraySampler3<double> sampler(grid.view(), gridSpacing,
                                                     gridOrigin);

    double s0 = sampler(Vector3D(1.5, 1.8, 1.2));
    EXPECT_LT(3.0, s0);
    EXPECT_GT(6.0, s0);
}
