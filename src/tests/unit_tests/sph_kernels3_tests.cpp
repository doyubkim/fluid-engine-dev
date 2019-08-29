// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/sph_kernels3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(SphStdKernel3, Constructors) {
    SphStdKernel3 kernel;
    EXPECT_DOUBLE_EQ(0.0, kernel.h);

    SphStdKernel3 kernel2(3.0);
    EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphStdKernel3, KernelFunction) {
    SphStdKernel3 kernel(10.0);

    double prevValue = kernel(0.0);

    for (int i = 1; i <= 10; ++i) {
        double value = kernel(static_cast<double>(i));
        EXPECT_LT(value, prevValue);
    }
}

TEST(SphStdKernel3, FirstDerivative) {
    SphStdKernel3 kernel(10.0);

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

TEST(SphStdKernel3, SecondDerivative) {
    SphStdKernel3 kernel(10.0);

    double value0 = kernel.secondDerivative(0.0);
    double value1 = kernel.secondDerivative(5.0);
    double value2 = kernel.secondDerivative(10.0);

    const double h5 = pow(10.0, 5.0);
    EXPECT_DOUBLE_EQ(-945.0 / (32.0 * kPiD * h5), value0);
    EXPECT_DOUBLE_EQ(0.0, value2);

    // Compare with finite difference
    const double e = 0.001;
    double fdm = (kernel(5.0 + e) - 2.0 * kernel(5.0) + kernel(5.0 - e)) / (e * e);
    EXPECT_NEAR(fdm, value1, 1e-10);
}

TEST(SphStdKernel3, Gradient) {
    SphStdKernel3 kernel(10.0);

    Vector3D value0 = kernel.gradient(0.0, Vector3D(1, 0, 0));
    EXPECT_DOUBLE_EQ(0.0, value0.x);
    EXPECT_DOUBLE_EQ(0.0, value0.y);
    EXPECT_DOUBLE_EQ(0.0, value0.z);

    Vector3D value1 = kernel.gradient(5.0, Vector3D(0, 1, 0));
    EXPECT_DOUBLE_EQ(0.0, value1.x);
    EXPECT_LT(0.0, value1.y);
    EXPECT_DOUBLE_EQ(0.0, value1.z);

    Vector3D value2 = kernel.gradient(Vector3D(0, 5, 0));
    EXPECT_EQ(value1, value2);
}

TEST(SphSpikyKernel3, Constructors) {
    SphSpikyKernel3 kernel;
    EXPECT_DOUBLE_EQ(0.0, kernel.h);

    SphSpikyKernel3 kernel2(3.0);
    EXPECT_DOUBLE_EQ(3.0, kernel2.h);
}

TEST(SphSpikyKernel3, KernelFunction) {
    SphSpikyKernel3 kernel(10.0);

    double prevValue = kernel(0.0);

    for (int i = 1; i <= 10; ++i) {
        double value = kernel(static_cast<double>(i));
        EXPECT_LT(value, prevValue);
    }
}

TEST(SphSpikyKernel3, FirstDerivative) {
    SphSpikyKernel3 kernel(10.0);

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

TEST(SphSpikyKernel3, Gradient) {
    SphSpikyKernel3 kernel(10.0);

    Vector3D value0 = kernel.gradient(0.0, Vector3D(1, 0, 0));
    EXPECT_LT(0.0, value0.x);
    EXPECT_DOUBLE_EQ(0.0, value0.y);
    EXPECT_DOUBLE_EQ(0.0, value0.z);

    Vector3D value1 = kernel.gradient(5.0, Vector3D(0, 1, 0));
    EXPECT_DOUBLE_EQ(0.0, value1.x);
    EXPECT_LT(0.0, value1.y);
    EXPECT_DOUBLE_EQ(0.0, value1.z);

    Vector3D value2 = kernel.gradient(Vector3D(0, 5, 0));
    EXPECT_EQ(value1, value2);
}

TEST(SphSpikyKernel3, SecondDerivative) {
    SphSpikyKernel3 kernel(10.0);

    double value0 = kernel.secondDerivative(0.0);
    double value1 = kernel.secondDerivative(5.0);
    double value2 = kernel.secondDerivative(10.0);

    const double h5 = pow(10.0, 5.0);
    EXPECT_DOUBLE_EQ(90.0 / (kPiD * h5), value0);
    EXPECT_DOUBLE_EQ(0.0, value2);

    // Compare with finite difference
    const double e = 0.001;
    double fdm = (kernel(5.0 + e) - 2.0 * kernel(5.0) + kernel(5.0 - e)) / (e * e);
    EXPECT_NEAR(fdm, value1, 1e-10);
}
