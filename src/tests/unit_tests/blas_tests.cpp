// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/blas.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Blas, Set) {
    Vector3D vec;

    Blas<double, Vector3D, Matrix3x3D>::set(3.14, &vec);

    EXPECT_DOUBLE_EQ(3.14, vec.x);
    EXPECT_DOUBLE_EQ(3.14, vec.y);
    EXPECT_DOUBLE_EQ(3.14, vec.z);

    Vector3D vec2(5.1, 3.7, 8.2);
    Blas<double, Vector3D, Matrix3x3D>::set(vec2, &vec);

    EXPECT_DOUBLE_EQ(5.1, vec.x);
    EXPECT_DOUBLE_EQ(3.7, vec.y);
    EXPECT_DOUBLE_EQ(8.2, vec.z);

    Matrix3x3D mat;

    Blas<double, Vector3D, Matrix3x3D>::set(0.414, &mat);

    for (int i = 0; i < 9; ++i) {
        double elem = mat[i];
        EXPECT_DOUBLE_EQ(0.414, elem);
    }

    Matrix3x3D mat2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    Blas<double, Vector3D, Matrix3x3D>::set(mat2, &mat);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }
}

TEST(Blas, Dot) {
    Vector3D vec(1.0, 2.0, 3.0), vec2(4.0, 5.0, 6.0);
    double result = Blas<double, Vector3D, Matrix3x3D>::dot(vec, vec2);
    EXPECT_DOUBLE_EQ(32.0, result);
}

TEST(Blas, Axpy) {
    Vector3D result;
    Blas<double, Vector3D, Matrix3x3D>::axpy(
        2.5, Vector3D(1, 2, 3), Vector3D(4, 5, 6), &result);

    EXPECT_DOUBLE_EQ(6.5, result.x);
    EXPECT_DOUBLE_EQ(10.0, result.y);
    EXPECT_DOUBLE_EQ(13.5, result.z);
}

TEST(Blas, Mvm) {
    Matrix3x3D mat(1, 2, 3, 4, 5, 6, 7, 8, 9);

    Vector3D result;
    Blas<double, Vector3D, Matrix3x3D>::mvm(
        mat, Vector3D(1, 2, 3), &result);

    EXPECT_DOUBLE_EQ(14.0, result.x);
    EXPECT_DOUBLE_EQ(32.0, result.y);
    EXPECT_DOUBLE_EQ(50.0, result.z);
}

TEST(Blas, Residual) {
    Matrix3x3D mat(1, 2, 3, 4, 5, 6, 7, 8, 9);

    Vector3D result;
    Blas<double, Vector3D, Matrix3x3D>::residual(
        mat, Vector3D(1, 2, 3), Vector3D(4, 5, 6), &result);

    EXPECT_DOUBLE_EQ(-10.0, result.x);
    EXPECT_DOUBLE_EQ(-27.0, result.y);
    EXPECT_DOUBLE_EQ(-44.0, result.z);
}

TEST(Blas, L2Norm) {
    Vector3D vec(-1.0, 2.0, -3.0);
    double result = Blas<double, Vector3D, Matrix3x3D>::l2Norm(vec);
    EXPECT_DOUBLE_EQ(std::sqrt(14.0), result);
}

TEST(Blas, LInfNorm) {
    Vector3D vec(-1.0, 2.0, -3.0);
    double result = Blas<double, Vector3D, Matrix3x3D>::lInfNorm(vec);
    EXPECT_DOUBLE_EQ(3.0, result);
}
