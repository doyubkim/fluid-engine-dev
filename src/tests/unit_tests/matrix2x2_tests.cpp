// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix2x2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Matrix2x2, Constructors) {
    Matrix2x2D mat;
    EXPECT_TRUE(mat == Matrix2x2D(1.0, 0.0, 0.0, 1.0));

    Matrix2x2D mat2(3.1);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(3.1, mat2[i]);
    }

    Matrix2x2D mat3(1.0, 2.0, 3.0, 4.0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat3[i]);
    }

    Matrix2x2D mat4({{1.0, 2.0}, {3.0, 4.0}});
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat4[i]);
    }

    Matrix2x2D mat5(mat4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat5[i]);
    }

    double arr[4] = {1.0, 2.0, 3.0, 4.0};
    Matrix2x2D mat6(arr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat6[i]);
    }
}

TEST(Matrix2x2, SetMethods) {
    Matrix2x2D mat;

    mat.set(3.1);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(3.1, mat[i]);
    }

    mat.set(0.0);
    mat.set(1.0, 2.0, 3.0, 4.0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.set({{1.0, 2.0}, {3.0, 4.0}});
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.set(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    double arr[4] = {1.0, 2.0, 3.0, 4.0};
    mat.set(arr);
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.setDiagonal(3.1);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(3.1, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat.set(0.0);
    mat.setOffDiagonal(4.2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (i != j) {
                EXPECT_DOUBLE_EQ(4.2, mat(i, j));
            } else {
                EXPECT_DOUBLE_EQ(0.0, mat(i, j));
            }
        }
    }

    mat.set(0.0);
    mat.setRow(0, Vector2D(1.0, 2.0));
    mat.setRow(1, Vector2D(3.0, 4.0));
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }

    mat.set(0.0);
    mat.setColumn(0, Vector2D(1.0, 3.0));
    mat.setColumn(1, Vector2D(2.0, 4.0));
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(i + 1), mat[i]);
    }
}

TEST(Matrix2x2, BasicGetters) {
    Matrix2x2D mat(1.0, 2.0, 3.0, 4.0), mat2(1.01, 2.01, 2.99, 4.0), mat3;

    EXPECT_TRUE(mat.isSimilar(mat2, 0.02));
    EXPECT_FALSE(mat.isSimilar(mat2, 0.001));

    EXPECT_TRUE(mat.isSquare());

    EXPECT_EQ(2u, mat.rows());
    EXPECT_EQ(2u, mat.cols());
}

TEST(Matrix2x2, BinaryOperators) {
    Matrix2x2D mat(-4.0, 3.0, -2.0, 1.0), mat2;
    Vector2D vec;

    mat2 = mat.add(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-2.0, 5.0, 0.0, 3.0)));

    mat2 = mat.add(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-3.0, 5.0, 1.0, 5.0)));

    mat2 = mat.sub(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-6.0, 1.0, -4.0, -1.0)));

    mat2 = mat.sub(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-5.0, 1.0, -5.0, -3.0)));

    mat2 = mat.mul(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-8.0, 6.0, -4.0, 2.0)));

    vec = mat.mul(Vector2D(1, 2));
    EXPECT_TRUE(vec.isSimilar(Vector2D(2.0, 0.0)));

    mat2 = mat.mul(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(5.0, 4.0, 1.0, 0.0)));

    mat2 = mat.div(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-2.0, 1.5, -1.0, 0.5)));


    mat2 = mat.radd(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-2.0, 5.0, 0.0, 3.0)));

    mat2 = mat.radd(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-3.0, 5.0, 1.0, 5.0)));

    mat2 = mat.rsub(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(6.0, -1.0, 4.0, 1.0)));

    mat2 = mat.rsub(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(5.0, -1.0, 5.0, 3.0)));

    mat2 = mat.rmul(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-8.0, 6.0, -4.0, 2.0)));

    mat2 = mat.rmul(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-8.0, 5.0, -20.0, 13.0)));

    mat2 = mat.rdiv(2.0);
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-0.5, 2.0/3.0, -1.0, 2.0)));
}

TEST(Matrix2x2, AugmentedOperators) {
    Matrix2x2D mat(-4.0, 3.0, -2.0, 1.0);

    mat.iadd(2.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-2.0, 5.0, 0.0, 3.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.iadd(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-3.0, 5.0, 1.0, 5.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.isub(2.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-6.0, 1.0, -4.0, -1.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.isub(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-5.0, 1.0, -5.0, -3.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.imul(2.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-8.0, 6.0, -4.0, 2.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.imul(Matrix2x2D(1.0, 2.0, 3.0, 4.0));
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(5.0, 4.0, 1.0, 0.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.idiv(2.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-2.0, 1.5, -1.0, 0.5)));
}

TEST(Matrix2x2, Modifiers) {
    Matrix2x2D mat(-4.0, 3.0, -2.0, 1.0);

    mat.transpose();
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-4.0, -2.0, 3.0, 1.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat.invert();
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(0.5, -1.5, 1.0, -2.0)));
}

TEST(Matrix2x2, ComplexGetters) {
    Matrix2x2D mat(-4.0, 3.0, -2.0, 1.0), mat2;

    EXPECT_DOUBLE_EQ(-2.0, mat.sum());

    EXPECT_DOUBLE_EQ(-0.5, mat.avg());

    EXPECT_DOUBLE_EQ(-4.0, mat.min());

    EXPECT_DOUBLE_EQ(3.0, mat.max());

    EXPECT_DOUBLE_EQ(1.0, mat.absmin());

    EXPECT_DOUBLE_EQ(-4.0, mat.absmax());

    EXPECT_DOUBLE_EQ(-3.0, mat.trace());

    EXPECT_DOUBLE_EQ(2.0, mat.determinant());

    mat2 = mat.diagonal();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-4.0, 0.0, 0.0, 1.0)));

    mat2 = mat.offDiagonal();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(0.0, 3.0, -2.0, 0.0)));

    mat2 = mat.strictLowerTri();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(0.0, 0.0, -2.0, 0.0)));

    mat2 = mat.strictUpperTri();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(0.0, 3.0, 0.0, 0.0)));

    mat2 = mat.lowerTri();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-4.0, 0.0, -2.0, 1.0)));

    mat2 = mat.upperTri();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-4.0, 3.0, 0.0, 1.0)));

    mat2 = mat.transposed();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-4.0, -2.0, 3.0, 1.0)));

    mat2 = mat.inverse();
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(0.5, -1.5, 1.0, -2.0)));

    Matrix2x2F mat3 = mat.castTo<float>();
    EXPECT_TRUE(mat3.isSimilar(Matrix2x2F(-4.f, 3.f, -2.f, 1.f)));
}

TEST(Matrix2x2, SetterOperatorOverloadings) {
    Matrix2x2D mat(-4.0, 3.0, -2.0, 1.0), mat2;

    mat2 = mat;
    EXPECT_TRUE(mat2.isSimilar(Matrix2x2D(-4.0, 3.0, -2.0, 1.0)));

    mat += 2.0;
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-2.0, 5.0, 0.0, 3.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat += Matrix2x2D(1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-3.0, 5.0, 1.0, 5.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat -= 2.0;
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-6.0, 1.0, -4.0, -1.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat -= Matrix2x2D(1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-5.0, 1.0, -5.0, -3.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat *= 2.0;
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-8.0, 6.0, -4.0, 2.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat *= Matrix2x2D(1.0, 2.0, 3.0, 4.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(5.0, 4.0, 1.0, 0.0)));

    mat.set(-4.0, 3.0, -2.0, 1.0);
    mat /= 2.0;
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-2.0, 1.5, -1.0, 0.5)));
}

TEST(Matrix2x2, GetterOperatorOverloadings) {
    Matrix2x2D mat(-4.0, 3.0, -2.0, 1.0);

    EXPECT_DOUBLE_EQ(-4.0, mat[0]);
    EXPECT_DOUBLE_EQ(3.0, mat[1]);
    EXPECT_DOUBLE_EQ(-2.0, mat[2]);
    EXPECT_DOUBLE_EQ(1.0, mat[3]);

    mat[0] = 4.0;
    mat[1] = -3.0;
    mat[2] = 2.0;
    mat[3] = -1.0;
    EXPECT_DOUBLE_EQ(4.0, mat[0]);
    EXPECT_DOUBLE_EQ(-3.0, mat[1]);
    EXPECT_DOUBLE_EQ(2.0, mat[2]);
    EXPECT_DOUBLE_EQ(-1.0, mat[3]);

    mat.set(-4.0, 3.0, -2.0, 1.0);
    EXPECT_DOUBLE_EQ(-4.0, mat(0, 0));
    EXPECT_DOUBLE_EQ(3.0, mat(0, 1));
    EXPECT_DOUBLE_EQ(-2.0, mat(1, 0));
    EXPECT_DOUBLE_EQ(1.0, mat(1, 1));

    mat(0, 0) = 4.0;
    mat(0, 1) = -3.0;
    mat(1, 0) = 2.0;
    mat(1, 1) = -1.0;
    EXPECT_DOUBLE_EQ(4.0, mat[0]);
    EXPECT_DOUBLE_EQ(-3.0, mat[1]);
    EXPECT_DOUBLE_EQ(2.0, mat[2]);
    EXPECT_DOUBLE_EQ(-1.0, mat[3]);

    mat.set(-4.0, 3.0, -2.0, 1.0);
    EXPECT_TRUE(mat == Matrix2x2D(-4.0, 3.0, -2.0, 1.0));

    mat.set(4.0, 3.0, 2.0, 1.0);
    EXPECT_TRUE(mat != Matrix2x2D(-4.0, 3.0, -2.0, 1.0));
}

TEST(Matrix2x2, Helpers) {
    Matrix2x2D mat = Matrix2x2D::makeZero();
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(0.0, 0.0, 0.0, 0.0)));

    mat = Matrix2x2D::makeIdentity();
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(1.0, 0.0, 0.0, 1.0)));

    mat = Matrix2x2D::makeScaleMatrix(3.0, -4.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(3.0, 0.0, 0.0, -4.0)));

    mat = Matrix2x2D::makeScaleMatrix(Vector2D(-2.0, 5.0));
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(-2.0, 0.0, 0.0, 5.0)));

    mat = Matrix2x2D::makeRotationMatrix(kPiD / 3.0);
    EXPECT_TRUE(mat.isSimilar(Matrix2x2D(
        0.5, -std::sqrt(3.0) / 2.0, std::sqrt(3.0) / 2.0, 0.5)));
}
