// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/sph_kernels2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(SphStdKernel2, Constructors) {
    SphStdKernel2 kernel;
    EXPECT_DOUBLE_EQ(0.0, kernel.h);

    SphStdKernel2 kernel2(3.0);
    EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphStdKernel2, KernelFunction) {
    SphStdKernel2 kernel(10.0);

    double prevValue = kernel(0.0);

    for (int i = 1; i <= 10; ++i) {
        double value = kernel(static_cast<double>(i));
        EXPECT_LT(value, prevValue);
    }
}

TEST(SphStdKernel2, FirstDerivative) {
    SphStdKernel2 kernel(10.0);

    double value0 = kernel.firstDerivative(0.0);
    double value1 = kernel.firstDerivative(5.0);
    double value2 = kernel.firstDerivative(10.0);
    EXPECT_DOUBLE_EQ(0.0, value0);
    EXPECT_DOUBLE_EQ(0.0, value2);

    // Compare with finite difference
    const double e = 0.001;
    double fdm = (kernel(5.0 + e) - kernel(5.0 - e)) / (2.0 * e);
    EXPECT_NEAR(fdm, value1, 1e-10);
}

TEST(SphStdKernel2, SecondDerivative) {
    SphStdKernel2 kernel(10.0);

    double value0 = kernel.secondDerivative(0.0);
    double value1 = kernel.secondDerivative(5.0);
    double value2 = kernel.secondDerivative(10.0);

    const double h4 = pow(10.0, 4.0);
    EXPECT_DOUBLE_EQ(-24.0 / (kPiD * h4), value0);
    EXPECT_DOUBLE_EQ(0.0, value2);

    // Compare with finite difference
    const double e = 0.001;
    double fdm = (kernel(5.0 + e) - 2.0 * kernel(5.0) + kernel(5.0 - e)) / (e * e);
    EXPECT_NEAR(fdm, value1, 1e-10);
}

TEST(SphStdKernel2, Gradient) {
    SphStdKernel2 kernel(10.0);

    Vector2D value0 = kernel.gradient(0.0, Vector2D(1, 0));
    EXPECT_DOUBLE_EQ(0.0, value0.x);
    EXPECT_DOUBLE_EQ(0.0, value0.y);

    Vector2D value1 = kernel.gradient(5.0, Vector2D(0, 1));
    EXPECT_DOUBLE_EQ(0.0, value1.x);
    EXPECT_LT(0.0, value1.y);

    Vector2D value2 = kernel.gradient(Vector2D(0, 5));
    EXPECT_EQ(value1, value2);
}

TEST(SphSpikyKernel2, Constructors) {
    SphSpikyKernel2 kernel;
    EXPECT_DOUBLE_EQ(0.0, kernel.h);

    SphSpikyKernel2 kernel2(3.0);
    EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphSpikyKernel2, KernelFunction) {
    SphSpikyKernel2 kernel(10.0);

    double prevValue = kernel(0.0);

    for (int i = 1; i <= 10; ++i) {
        double value = kernel(static_cast<double>(i));
        EXPECT_LT(value, prevValue);
    }
}

TEST(SphSpikyKernel2, FirstDerivative) {
    SphSpikyKernel2 kernel(10.0);

    double value0 = kernel.firstDerivative(0.0);
    double value1 = kernel.firstDerivative(5.0);
    double value2 = kernel.firstDerivative(10.0);
    EXPECT_LT(value0, value1);
    EXPECT_LT(value1, value2);

    // Compare with finite difference
    const double e = 0.001;
    double fdm = (kernel(5.0 + e) - kernel(5.0 - e)) / (2.0 * e);
    EXPECT_NEAR(fdm, value1, 1e-10);
}

TEST(SphSpikyKernel2, Gradient) {
    SphSpikyKernel2 kernel(10.0);

    Vector2D value0 = kernel.gradient(0.0, Vector2D(1, 0));
    EXPECT_LT(0.0, value0.x);
    EXPECT_DOUBLE_EQ(0.0, value0.y);

    Vector2D value1 = kernel.gradient(5.0, Vector2D(0, 1));
    EXPECT_DOUBLE_EQ(0.0, value1.x);
    EXPECT_LT(0.0, value1.y);

    Vector2D value2 = kernel.gradient(Vector2D(0, 5));
    EXPECT_EQ(value1, value2);
}

TEST(SphSpikyKernel2, SecondDerivative) {
    SphSpikyKernel2 kernel(10.0);

    double value0 = kernel.secondDerivative(0.0);
    double value1 = kernel.secondDerivative(5.0);
    double value2 = kernel.secondDerivative(10.0);

    const double h4 = pow(10.0, 4.0);
    EXPECT_DOUBLE_EQ(60.0 / (kPiD * h4), value0);
    EXPECT_DOUBLE_EQ(0.0, value2);

    // Compare with finite difference
    const double e = 0.001;
    double fdm = (kernel(5.0 + e) - 2.0 * kernel(5.0) + kernel(5.0 - e)) / (e * e);
    EXPECT_NEAR(fdm, value1, 1e-10);
}
