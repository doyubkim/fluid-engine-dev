// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix3x3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Matrix3x3, Constructors) {
    Matrix3x3D mat;
    EXPECT_TRUE(mat == Matrix3x3D(1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0));

    Matrix3x3D mat2(3.1);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(3.1, mat2[i]);
    }

    Matrix3x3D mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat3[i]);
    }

    Matrix3x3D mat4({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat4[i]);
    }

    Matrix3x3D mat5(mat4);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat5[i]);
    }

    double arr[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    Matrix3x3D mat6(arr);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat6[i]);
    }
}

TEST(Matrix3x3, SetMethods) {
    Matrix3x3D mat;

    mat.set(3.1);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(3.1, mat[i]);
    }

    mat.set(0.0);
    mat.set(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.set({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.set(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    double arr[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    mat.set(arr);
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.setDiagonal(3.1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(3.1, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat.set(0.0);
    mat.setOffDiagonal(4.2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i != j) {
                EXPECT_DOUBLE_EQ(4.2, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat.set(0.0);
    mat.setRow(0, Vector3D(1.0, 2.0, 3.0));
    mat.setRow(1, Vector3D(4.0, 5.0, 6.0));
    mat.setRow(2, Vector3D(7.0, 8.0, 9.0));
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.setColumn(0, Vector3D(1.0, 4.0, 7.0));
    mat.setColumn(1, Vector3D(2.0, 5.0, 8.0));
    mat.setColumn(2, Vector3D(3.0, 6.0, 9.0));
    for (int i = 0; i < 9; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }
}

TEST(Matrix3x3, BasicGetters) {
    Matrix3x3D mat(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
               mat2(1.01, 2.01, 2.99, 4.0, 4.99, 6.001, 7.0003, 8.0, 8.99),
               mat3;

    EXPECT_TRUE(mat.isSimilar(mat2, 0.02));
    EXPECT_FALSE(mat.isSimilar(mat2, 0.001));

    EXPECT_TRUE(mat.isSquare());

    EXPECT_EQ(3u, mat.rows());
    EXPECT_EQ(3u, mat.cols());
}

TEST(Matrix3x3, BinaryOperators) {
    Matrix3x3D mat(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0), mat2;
    Vector3D vec;

    mat2 = mat.add(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(11.0, -6.0, 9.0, -4.0, 7.0, -2.0, 5.0, 0.0, 3.0)));

    mat2 = mat.add(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(10.0, -6.0, 10.0, -2.0, 10.0, 2.0, 10.0, 6.0, 10.0)));

    mat2 = mat.sub(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(7.0, -10.0, 5.0, -8.0, 3.0, -6.0, 1.0, -4.0, -1.0)));

    mat2 = mat.sub(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(8.0, -10.0, 4.0, -10.0, 0.0, -10.0, -4.0, -10.0, -8.0)));

    mat2 = mat.mul(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(18.0, -16.0, 14.0, -12.0, 10.0, -8.0, 6.0, -4.0, 2.0)));

    vec = mat.mul(Vector3D(1, 2, 3));
    EXPECT_TRUE(vec.isSimilar(Vector3D(14.0, -8.0, 2.0)));

    mat2 = mat.mul(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(26.0, 34.0, 42.0, -14.0, -19.0, -24.0, 2.0, 4.0, 6.0)))
        << mat2(0, 0) << ' ' << mat2(0, 1) << ' ' << mat2(0, 2) << "\n"
        << mat2(1, 0) << ' ' << mat2(1, 1) << ' ' << mat2(1, 2) << "\n"
        << mat2(2, 0) << ' ' << mat2(2, 1) << ' ' << mat2(2, 2) << "\n";

    mat2 = mat.div(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(4.5, -4.0, 3.5, -3.0, 2.5, -2.0, 1.5, -1.0, 0.5)));


    mat2 = mat.radd(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(11.0, -6.0, 9.0, -4.0, 7.0, -2.0, 5.0, 0.0, 3.0)));

    mat2 = mat.radd(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(10.0, -6.0, 10.0, -2.0, 10.0, 2.0, 10.0, 6.0, 10.0)));

    mat2 = mat.rsub(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(-7.0, 10.0, -5.0, 8.0, -3.0, 6.0, -1.0, 4.0, 1.0)));

    mat2 = mat.rsub(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(-8.0, 10.0, -4.0, 10.0, 0.0, 10.0, 4.0, 10.0, 8.0)));

    mat2 = mat.rmul(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(18.0, -16.0, 14.0, -12.0, 10.0, -8.0, 6.0, -4.0, 2.0)));

    mat2 = mat.rmul(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(6.0, -4.0, 2.0, 24.0, -19.0, 14.0, 42.0, -34.0, 26.0)))
        << mat2(0, 0) << ' ' << mat2(0, 1) << ' ' << mat2(0, 2) << "\n"
        << mat2(1, 0) << ' ' << mat2(1, 1) << ' ' << mat2(1, 2) << "\n"
        << mat2(2, 0) << ' ' << mat2(2, 1) << ' ' << mat2(2, 2) << "\n";

    mat2 = mat.rdiv(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(
            2.0/9.0, -0.25, 2.0/7.0, -1.0/3.0, 0.4, -0.5, 2.0/3.0, -1.0, 2.0)));
}

TEST(Matrix3x3, AugmentedOperators) {
    Matrix3x3D mat(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);

    mat.iadd(2.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(11.0, -6.0, 9.0, -4.0, 7.0, -2.0, 5.0, 0.0, 3.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat.iadd(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(10.0, -6.0, 10.0, -2.0, 10.0, 2.0, 10.0, 6.0, 10.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat.isub(2.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(7.0, -10.0, 5.0, -8.0, 3.0, -6.0, 1.0, -4.0, -1.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat.isub(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(8.0, -10.0, 4.0, -10.0, 0.0, -10.0, -4.0, -10.0, -8.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat.imul(2.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(18.0, -16.0, 14.0, -12.0, 10.0, -8.0, 6.0, -4.0, 2.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat.imul(Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(26.0, 34.0, 42.0, -14.0, -19.0, -24.0, 2.0, 4.0, 6.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat.idiv(2.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(4.5, -4.0, 3.5, -3.0, 2.5, -2.0, 1.5, -1.0, 0.5)));
}

TEST(Matrix3x3, Modifiers) {
    Matrix3x3D mat(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);

    mat.transpose();
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(9.0, -6.0, 3.0, -8.0, 5.0, -2.0, 7.0, -4.0, 1.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 2.0);
    mat.invert();
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(-2.0/3.0, -2.0/3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0)));
}

TEST(Matrix3x3, ComplexGetters) {
    Matrix3x3D mat(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0), mat2;

    EXPECT_DOUBLE_EQ(5.0, mat.sum());

    EXPECT_DOUBLE_EQ(5.0 / 9.0, mat.avg());

    EXPECT_DOUBLE_EQ(-8.0, mat.min());

    EXPECT_DOUBLE_EQ(9.0, mat.max());

    EXPECT_DOUBLE_EQ(1.0, mat.absmin());

    EXPECT_DOUBLE_EQ(9.0, mat.absmax());

    EXPECT_DOUBLE_EQ(15.0, mat.trace());

    EXPECT_DOUBLE_EQ(0.0, mat.determinant());

    mat2 = mat.diagonal();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(9.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0)));

    mat2 = mat.offDiagonal();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(0.0, -8.0, 7.0, -6.0, 0.0, -4.0, 3.0, -2.0, 0.0)));

    mat2 = mat.strictLowerTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(0.0, 0.0, 0.0,
                   -6.0, 0.0, 0.0,
                   3.0, -2.0, 0.0)));

    mat2 = mat.strictUpperTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(0.0, -8.0, 7.0,
                   0.0, 0.0, -4.0,
                   0.0, 0.0, 0.0)));

    mat2 = mat.lowerTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(9.0, 0.0, 0.0,
                   -6.0, 5.0, 0.0,
                   3.0, -2.0, 1.0)));

    mat2 = mat.upperTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(9.0, -8.0, 7.0,
                   0.0, 5.0, -4.0,
                   0.0, 0.0, 1.0)));

    mat2 = mat.transposed();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(9.0, -6.0, 3.0, -8.0, 5.0, -2.0, 7.0, -4.0, 1.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 2.0);
    mat2 = mat.inverse();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(-2.0/3.0, -2.0/3.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    Matrix3x3F mat3 = mat.castTo<float>();
    EXPECT_TRUE(mat3.isSimilar(
        Matrix3x3F(9.f, -8.f, 7.f, -6.f, 5.f, -4.f, 3.f, -2.f, 1.f)));
}

TEST(Matrix3x3, SetterOperatorOverloadings) {
    Matrix3x3D mat(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0), mat2;

    mat2 = mat;
    EXPECT_TRUE(mat2.isSimilar(
        Matrix3x3D(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0)));

    mat += 2.0;
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(11.0, -6.0, 9.0, -4.0, 7.0, -2.0, 5.0, 0.0, 3.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat += Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(10.0, -6.0, 10.0, -2.0, 10.0, 2.0, 10.0, 6.0, 10.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat -= 2.0;
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(7.0, -10.0, 5.0, -8.0, 3.0, -6.0, 1.0, -4.0, -1.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat -= Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(8.0, -10.0, 4.0, -10.0, 0.0, -10.0, -4.0, -10.0, -8.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat *= 2.0;
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(18.0, -16.0, 14.0, -12.0, 10.0, -8.0, 6.0, -4.0, 2.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat *= Matrix3x3D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(26.0, 34.0, 42.0, -14.0, -19.0, -24.0, 2.0, 4.0, 6.0)));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    mat /= 2.0;
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(4.5, -4.0, 3.5, -3.0, 2.5, -2.0, 1.5, -1.0, 0.5)));
}

TEST(Matrix3x3, GetterOperatorOverloadings) {
    Matrix3x3D mat(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);

    EXPECT_DOUBLE_EQ(9.0, mat[0]);
    EXPECT_DOUBLE_EQ(-8.0, mat[1]);
    EXPECT_DOUBLE_EQ(7.0, mat[2]);
    EXPECT_DOUBLE_EQ(-6.0, mat[3]);
    EXPECT_DOUBLE_EQ(5.0, mat[4]);
    EXPECT_DOUBLE_EQ(-4.0, mat[5]);
    EXPECT_DOUBLE_EQ(3.0, mat[6]);
    EXPECT_DOUBLE_EQ(-2.0, mat[7]);
    EXPECT_DOUBLE_EQ(1.0, mat[8]);

    mat[0] = -9.0;
    mat[1] = 8.0;
    mat[2] = -7.0;
    mat[3] = 6.0;
    mat[4] = -5.0;
    mat[5] = 4.0;
    mat[6] = -3.0;
    mat[7] = 2.0;
    mat[8] = -1.0;
    EXPECT_DOUBLE_EQ(-9.0, mat[0]);
    EXPECT_DOUBLE_EQ(8.0, mat[1]);
    EXPECT_DOUBLE_EQ(-7.0, mat[2]);
    EXPECT_DOUBLE_EQ(6.0, mat[3]);
    EXPECT_DOUBLE_EQ(-5.0, mat[4]);
    EXPECT_DOUBLE_EQ(4.0, mat[5]);
    EXPECT_DOUBLE_EQ(-3.0, mat[6]);
    EXPECT_DOUBLE_EQ(2.0, mat[7]);
    EXPECT_DOUBLE_EQ(-1.0, mat[8]);

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    EXPECT_DOUBLE_EQ(9.0, mat[0]);
    EXPECT_DOUBLE_EQ(-8.0, mat[1]);
    EXPECT_DOUBLE_EQ(7.0, mat[2]);
    EXPECT_DOUBLE_EQ(-6.0, mat[3]);
    EXPECT_DOUBLE_EQ(5.0, mat[4]);
    EXPECT_DOUBLE_EQ(-4.0, mat[5]);
    EXPECT_DOUBLE_EQ(3.0, mat[6]);
    EXPECT_DOUBLE_EQ(-2.0, mat[7]);
    EXPECT_DOUBLE_EQ(1.0, mat[8]);

    mat(0, 0) = -9.0;
    mat(0, 1) = 8.0;
    mat(0, 2) = -7.0;
    mat(1, 0) = 6.0;
    mat(1, 1) = -5.0;
    mat(1, 2) = 4.0;
    mat(2, 0) = -3.0;
    mat(2, 1) = 2.0;
    mat(2, 2) = -1.0;
    EXPECT_DOUBLE_EQ(-9.0, mat[0]);
    EXPECT_DOUBLE_EQ(8.0, mat[1]);
    EXPECT_DOUBLE_EQ(-7.0, mat[2]);
    EXPECT_DOUBLE_EQ(6.0, mat[3]);
    EXPECT_DOUBLE_EQ(-5.0, mat[4]);
    EXPECT_DOUBLE_EQ(4.0, mat[5]);
    EXPECT_DOUBLE_EQ(-3.0, mat[6]);
    EXPECT_DOUBLE_EQ(2.0, mat[7]);
    EXPECT_DOUBLE_EQ(-1.0, mat[8]);
    EXPECT_DOUBLE_EQ(-9.0, mat(0, 0));
    EXPECT_DOUBLE_EQ(8.0, mat(0, 1));
    EXPECT_DOUBLE_EQ(-7.0, mat(0, 2));
    EXPECT_DOUBLE_EQ(6.0, mat(1, 0));
    EXPECT_DOUBLE_EQ(-5.0, mat(1, 1));
    EXPECT_DOUBLE_EQ(4.0, mat(1, 2));
    EXPECT_DOUBLE_EQ(-3.0, mat(2, 0));
    EXPECT_DOUBLE_EQ(2.0, mat(2, 1));
    EXPECT_DOUBLE_EQ(-1.0, mat(2, 2));

    mat.set(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0);
    EXPECT_TRUE(
        mat == Matrix3x3D(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0));

    mat.set(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
    EXPECT_TRUE(
        mat != Matrix3x3D(9.0, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0));
}

TEST(Matrix3x3, Helpers) {
    Matrix3x3D mat = Matrix3x3D::makeZero();
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));

    mat = Matrix3x3D::makeIdentity();
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(1.0, 0.0, 0.0,
                   0.0, 1.0, 0.0,
                   0.0, 0.0, 1.0)));

    mat = Matrix3x3D::makeScaleMatrix(3.0, -4.0, 2.4);
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(3.0, 0.0, 0.0,
                   0.0, -4.0, 0.0,
                   0.0, 0.0, 2.4)));

    mat = Matrix3x3D::makeScaleMatrix(Vector3D(-2.0, 5.0, 3.5));
    EXPECT_TRUE(mat.isSimilar(
        Matrix3x3D(-2.0, 0.0, 0.0,
                   0.0, 5.0, 0.0,
                   0.0, 0.0, 3.5)));

    mat = Matrix3x3D::makeRotationMatrix(
        Vector3D(-1.0/3.0, 2.0/3.0, 2.0/3.0), -74.0 / 180.0 * kPiD);
    EXPECT_TRUE(mat.isSimilar(Matrix3x3D(
        0.36, 0.48, -0.8,
        -0.8, 0.60, 0.0,
        0.48, 0.64, 0.6), 0.05))
        << mat(0, 0) << ' ' << mat(0, 1) << ' ' << mat(0, 2) << "\n"
        << mat(1, 0) << ' ' << mat(1, 1) << ' ' << mat(1, 2) << "\n"
        << mat(2, 0) << ' ' << mat(2, 1) << ' ' << mat(2, 2) << "\n";
}
