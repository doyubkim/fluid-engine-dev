// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix4x4.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Matrix4x4, Constructors) {
    Matrix4x4D mat;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(1.0, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    Matrix4x4D mat2(3.1);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(3.1, mat2[i]);
    }

    Matrix4x4D mat3(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat3[i]);
    }

    Matrix4x4D mat4(
        {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat4[i]);
    }

    Matrix4x4D mat5(mat4);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat5[i]);
    }

    double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Matrix4x4D mat6(arr);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat6[i]);
    }
}

TEST(Matrix4x4, SetMethods) {
    Matrix4x4D mat;

    mat.set(3.1);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(3.1, mat[i]);
    }

    mat.set(0.0);
    mat.set(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.set({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.set(Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    double arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    mat.set(arr);
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.setDiagonal(3.1);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(3.1, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat.set(0.0);
    mat.setOffDiagonal(4.2);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i != j) {
                EXPECT_DOUBLE_EQ(4.2, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat.set(0.0);
    mat.setRow(0, Vector4D(1, 2, 3, 4));
    mat.setRow(1, Vector4D(5, 6, 7, 8));
    mat.setRow(2, Vector4D(9, 10, 11, 12));
    mat.setRow(3, Vector4D(13, 14, 15, 16));
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.setColumn(0, Vector4D(1, 5, 9, 13));
    mat.setColumn(1, Vector4D(2, 6, 10, 14));
    mat.setColumn(2, Vector4D(3, 7, 11, 15));
    mat.setColumn(3, Vector4D(4, 8, 12, 16));
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }
}

TEST(Matrix4x4, BasicGetters) {
    Matrix4x4D mat(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
               mat2(1.01, 2.01, 2.99, 4.0, 4.99, 6.001, 7.0003, 8.0, 8.99,
                    10.01, 11, 11.99, 13.01, 14.001, 14.999, 16),
               mat3;

    EXPECT_TRUE(mat.isSimilar(mat2, 0.02));
    EXPECT_FALSE(mat.isSimilar(mat2, 0.001));

    EXPECT_TRUE(mat.isSquare());

    EXPECT_EQ(4u, mat.rows());
    EXPECT_EQ(4u, mat.cols());
}

TEST(Matrix4x4, BinaryOperators) {
    Matrix4x4D mat(-16, 15, -14, 13,
                   -12, 11, -10, 9,
                   -8, 7, -6, 5,
                   -6, 3, -2, 1),
               mat2;
    Vector4D vec;

    mat2 = mat.add(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-14, 17, -12, 15},
         {-10, 13, -8, 11},
         {-6, 9, -4, 7},
         {-4, 5, 0, 3}}));

    mat2 = mat.add(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat2.isSimilar(
        {{-15, 17, -11, 17},
         {-7, 17, -3, 17},
         {1, 17, 5, 17},
         {7, 17, 13, 17}}));

    mat2 = mat.sub(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-18, 13, -16, 11},
         {-14, 9, -12, 7},
         {-10, 5, -8, 3},
         {-8, 1, -4, -1}}));

    mat2 = mat.sub(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat2.isSimilar(
        {{-17, 13, -17, 9},
         {-17, 5, -17, 1},
         {-17, -3, -17, -7},
         {-19, -11, -17, -15}}));

    mat2 = mat.mul(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-32, 30, -28, 26},
         {-24, 22, -20, 18},
         {-16, 14, -12, 10},
         {-12, 6, -4, 2}}));

    vec = mat.mul(Vector4D(1, 2, 3, 4));
    EXPECT_TRUE(vec.isSimilar(Vector4D(24, 16, 8, -2)));

    mat2 = mat.mul(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat2.isSimilar(
        {{102, 100, 98, 96},
         {70, 68, 66, 64},
         {38, 36, 34, 32},
         {4, 0, -4, -8}}));

    mat2 = mat.div(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-8.0, 15.0/2.0, -7.0, 13.0/2.0},
         {-6.0, 11.0/2.0, -5.0, 9.0/2.0},
         {-4.0, 7.0/2.0, -3.0, 5.0/2.0},
         {-3.0, 3.0/2.0, -1.0, 1.0/2.0}}));


    mat2 = mat.radd(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-14, 17, -12, 15},
         {-10, 13, -8, 11},
         {-6, 9, -4, 7},
         {-4, 5, 0, 3}}));

    mat2 = mat.radd(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat2.isSimilar(
        {{-15, 17, -11, 17},
         {-7, 17, -3, 17},
         {1, 17, 5, 17},
         {7, 17, 13, 17}}));

    mat2 = mat.rsub(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{18, -13, 16, -11},
         {14, -9, 12, -7},
         {10, -5, 8, -3},
         {8, -1, 4, 1}}));

    mat2 = mat.rsub(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat2.isSimilar(
        {{17, -13, 17, -9},
         {17, -5, 17, -1},
         {17, 3, 17, 7},
         {19, 11, 17, 15}}));

    mat2 = mat.rmul(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-32, 30, -28, 26},
         {-24, 22, -20, 18},
         {-16, 14, -12, 10},
         {-12, 6, -4, 2}}));

    mat2 = mat.rmul(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat2.isSimilar(
        {{-88, 70, -60, 50},
         {-256, 214, -188, 162},
         {-424, 358, -316, 274},
         {-592, 502, -444, 386}}));

    mat2 = mat.rdiv(2.0);
    EXPECT_TRUE(mat2.isSimilar(
        {{-1.0/8.0, 2.0/15.0, -1.0/7.0, 2.0/13.0},
         {-1.0/6.0, 2.0/11.0, -1.0/5.0, 2.0/9.0},
         {-1.0/4.0, 2.0/7.0, -1.0/3.0, 2.0/5.0},
         {-1.0/3.0, 2.0/3.0, -1.0, 2.0}}));
}

TEST(Matrix4x4, AugmentedOperators) {
    Matrix4x4D mat(
        -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);

    mat.iadd(2.0);
    EXPECT_TRUE(mat.isSimilar(
        {{-14, 17, -12, 15},
         {-10, 13, -8, 11},
         {-6, 9, -4, 7},
         {-4, 5, 0, 3}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat.iadd(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat.isSimilar(
        {{-15, 17, -11, 17},
         {-7, 17, -3, 17},
         {1, 17, 5, 17},
         {7, 17, 13, 17}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat.isub(2.0);
    EXPECT_TRUE(mat.isSimilar(
        {{-18, 13, -16, 11},
         {-14, 9, -12, 7},
         {-10, 5, -8, 3},
         {-8, 1, -4, -1}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat.isub(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat.isSimilar(
        {{-17, 13, -17, 9},
         {-17, 5, -17, 1},
         {-17, -3, -17, -7},
         {-19, -11, -17, -15}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat.imul(2.0);
    EXPECT_TRUE(mat.isSimilar(
        {{-32, 30, -28, 26},
         {-24, 22, -20, 18},
         {-16, 14, -12, 10},
         {-12, 6, -4, 2}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat.imul(
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    EXPECT_TRUE(mat.isSimilar(
        {{102, 100, 98, 96},
         {70, 68, 66, 64},
         {38, 36, 34, 32},
         {4, 0, -4, -8}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat.idiv(2.0);
    EXPECT_TRUE(mat.isSimilar(
        {{-8.0, 15.0/2.0, -7.0, 13.0/2.0},
         {-6.0, 11.0/2.0, -5.0, 9.0/2.0},
         {-4.0, 7.0/2.0, -3.0, 5.0/2.0},
         {-3.0, 3.0/2.0, -1.0, 1.0/2.0}}));
}

TEST(Matrix4x4, Modifiers) {
    Matrix4x4D mat(
        -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);

    mat.transpose();
    EXPECT_TRUE(mat.isSimilar(
        {{-16, -12, -8, -6},
         {15, 11, 7, 3},
         {-14, -10, -6, -2},
         {13, 9, 5, 1}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 6, -6, 3, -2, 2);
    mat.invert();
    EXPECT_TRUE(mat.isSimilar(
        {{-1.0/2.0, 1.0/2.0, 1.0/2.0, -1.0/2.0},
         {-5.0/2.0, 5.0/2.0, 2.0, -1.0},
         {-5.0/4.0, 1.0/4.0, 5.0/2.0, -1.0/2.0},
         {1.0, -2.0, 1.0, 0.0}}));
}

TEST(Matrix4x4, ComplexGetters) {
    Matrix4x4D mat(
        -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1),
               mat2;

    EXPECT_DOUBLE_EQ(-8.0, mat.sum());

    EXPECT_DOUBLE_EQ(-0.5, mat.avg());

    EXPECT_DOUBLE_EQ(-16.0, mat.min());

    EXPECT_DOUBLE_EQ(15.0, mat.max());

    EXPECT_DOUBLE_EQ(1.0, mat.absmin());

    EXPECT_DOUBLE_EQ(-16.0, mat.absmax());

    EXPECT_DOUBLE_EQ(-10.0, mat.trace());

    EXPECT_DOUBLE_EQ(0.0, mat.determinant());

    mat2 = mat.diagonal();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(
            -16, 0, 0, 0, 0, 11, 0, 0, 0, 0, -6, 0, 0, 0, 0, 1)));

    mat2 = mat.offDiagonal();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(
            0, 15, -14, 13, -12, 0, -10, 9, -8, 7, 0, 5, -4, 3, -2, 0)));

    mat2 = mat.strictLowerTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(0, 0, 0, 0,
                   -12, 0, 0, 0,
                   -8, 7, 0, 0,
                   -4, 3, -2, 0)));

    mat2 = mat.strictUpperTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(0, 15, -14, 13,
                   0, 0, -10, 9,
                   0, 0, 0, 5,
                   0, 0, 0, 0)));

    mat2 = mat.lowerTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(-16, 0, 0, 0,
                   -12, 11, 0, 0,
                   -8, 7, -6, 0,
                   -4, 3, -2, 1)));

    mat2 = mat.upperTri();
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(-16, 15, -14, 13,
                   0, 11, -10, 9,
                   0, 0, -6, 5,
                   0, 0, 0, 1)));

    mat2 = mat.transposed();
    EXPECT_TRUE(mat2.isSimilar(
        {{-16, -12, -8, -4},
         {15, 11, 7, 3},
         {-14, -10, -6, -2},
         {13, 9, 5, 1}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 6, -6, 3, -2, 2);
    mat2 = mat.inverse();
    EXPECT_TRUE(mat2.isSimilar(
        {{-1.0/2.0, 1.0/2.0, 1.0/2.0, -1.0/2.0},
         {-5.0/2.0, 5.0/2.0, 2.0, -1.0},
         {-5.0/4.0, 1.0/4.0, 5.0/2.0, -1.0/2.0},
         {1.0, -2.0, 1.0, 0.0}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    Matrix4x4F mat3 = mat.castTo<float>();
    EXPECT_TRUE(mat3.isSimilar(
        Matrix4x4F(
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1)));
}

TEST(Matrix4x4, SetterOperatorOverloadings) {
    Matrix4x4D mat(
        -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1),
               mat2;

    mat2 = mat;
    EXPECT_TRUE(mat2.isSimilar(
        Matrix4x4D(
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1)));

    mat += 2.0;
    EXPECT_TRUE(mat.isSimilar(
        {{-14, 17, -12, 15},
         {-10, 13, -8, 11},
         {-6, 9, -4, 7},
         {-4, 5, 0, 3}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat +=
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    EXPECT_TRUE(mat.isSimilar(
        {{-15, 17, -11, 17},
         {-7, 17, -3, 17},
         {1, 17, 5, 17},
         {7, 17, 13, 17}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat -= 2.0;
    EXPECT_TRUE(mat.isSimilar(
        {{-18, 13, -16, 11},
         {-14, 9, -12, 7},
         {-10, 5, -8, 3},
         {-8, 1, -4, -1}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat -=
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    EXPECT_TRUE(mat.isSimilar(
        {{-17, 13, -17, 9},
         {-17, 5, -17, 1},
         {-17, -3, -17, -7},
         {-19, -11, -17, -15}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat *= 2.0;
    EXPECT_TRUE(mat.isSimilar(
        {{-32, 30, -28, 26},
         {-24, 22, -20, 18},
         {-16, 14, -12, 10},
         {-12, 6, -4, 2}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat *=
        Matrix4x4D(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    EXPECT_TRUE(mat.isSimilar(
        {{102, 100, 98, 96},
         {70, 68, 66, 64},
         {38, 36, 34, 32},
         {4, 0, -4, -8}}));

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -6, 3, -2, 1);
    mat /= 2.0;
    EXPECT_TRUE(mat.isSimilar(
        {{-8.0, 15.0/2.0, -7.0, 13.0/2.0},
         {-6.0, 11.0/2.0, -5.0, 9.0/2.0},
         {-4.0, 7.0/2.0, -3.0, 5.0/2.0},
         {-3.0, 3.0/2.0, -1.0, 1.0/2.0}}));
}

TEST(Matrix4x4, GetterOperatorOverloadings) {
    Matrix4x4D mat(
        -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1);

    double sign = -1.0;
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(sign * (16 - i), mat[i]);
        sign *= -1.0;

        mat[i] *= -1.0;
    }

    sign = 1.0;
    for (int i = 0; i < 16; ++i) {
        EXPECT_DOUBLE_EQ(sign * (16 - i), mat[i]);
        sign *= -1.0;
    }

    sign = 1.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double ans = sign * (16 - (4 * i + j));
            EXPECT_DOUBLE_EQ(ans, mat(i, j));
            sign *= -1.0;
        }
    }

    mat.set(-16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1);
    EXPECT_TRUE(
        mat == Matrix4x4D(
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1));

    mat.set(16, -15, 14, -13, 12, -11, 10, -9, -8, 7, -6, 5, -4, 3, -2, 1);
    EXPECT_TRUE(
        mat != Matrix4x4D(
            -16, 15, -14, 13, -12, 11, -10, 9, -8, 7, -6, 5, -4, 3, -2, 1));
}

TEST(Matrix4x4, Helpers) {
    Matrix4x4D mat = Matrix4x4D::makeZero();
    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(0.0, mat[i]);
    }

    mat = Matrix4x4D::makeIdentity();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(1.0, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat = Matrix4x4D::makeScaleMatrix(3.0, -4.0, 2.4);
    EXPECT_TRUE(mat.isSimilar(
        Matrix4x4D(3.0, 0.0, 0.0, 0.0,
                   0.0, -4.0, 0.0, 0.0,
                   0.0, 0.0, 2.4, 0.0,
                   0.0, 0.0, 0.0, 1.0)));

    mat = Matrix4x4D::makeScaleMatrix(Vector3D(-2.0, 5.0, 3.5));
    EXPECT_TRUE(mat.isSimilar(
        Matrix4x4D(-2.0, 0.0, 0.0, 0.0,
                   0.0, 5.0, 0.0, 0.0,
                   0.0, 0.0, 3.5, 0.0,
                   0.0, 0.0, 0.0, 1.0)));

    mat = Matrix4x4D::makeRotationMatrix(
        Vector3D(-1.0/3.0, 2.0/3.0, 2.0/3.0), -74.0 / 180.0 * kPiD);
    EXPECT_TRUE(mat.isSimilar(Matrix4x4D(
        0.36, 0.48, -0.8, 0,
        -0.8, 0.60, 0.0, 0,
        0.48, 0.64, 0.6, 0,
        0, 0, 0, 1), 0.05));

    mat = Matrix4x4D::makeTranslationMatrix(Vector3D(-2.0, 5.0, 3.5));
    EXPECT_TRUE(mat.isSimilar(
        Matrix4x4D(1.0, 0.0, 0.0, -2.0,
                   0.0, 1.0, 0.0, 5.0,
                   0.0, 0.0, 1.0, 3.5,
                   0.0, 0.0, 0.0, 1.0)));
}
