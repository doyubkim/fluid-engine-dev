// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix.h>
#include <jet/vector.h>

#include <gtest/gtest.h>

#include <iostream>

using namespace jet;

namespace jet {

template <typename T, size_t M, size_t N>
std::ostream& operator<<(std::ostream& os, const Matrix<T, M, N>& mat) {
    for (size_t i = 0; i < mat.rows(); ++i) {
        for (size_t j = 0; j < mat.cols(); ++j) {
            os << mat(i, j);
            if (j + 1 < mat.cols()) {
                os << std::string(", ");
            }
        }
        os << std::endl;
    }
    return os;
}
}  // namespace jet

TEST(Matrix, Constructors) {
    Matrix<double, 2, 3> mat;

    EXPECT_EQ(2u, mat.rows());
    EXPECT_EQ(3u, mat.cols());

    for (double elem : mat) {
        EXPECT_DOUBLE_EQ(0.0, elem);
    }

    Matrix<double, 2, 3> mat2(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(i + 1.0, mat2[i]);
    }

    Matrix<double, 2, 3> mat3 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(i + 1.0, mat3[i]);
    }

    Matrix<double, 2, 3> mat4(mat3);

    for (int i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(i + 1.0, mat4[i]);
    }
}

TEST(Matrix, BasicSetters) {
    Matrix<double, 4, 2> mat;
    mat.set(5.0);
    EXPECT_EQ(4u, mat.rows());
    EXPECT_EQ(2u, mat.cols());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(5.0, mat[i]);
    }

    mat.set(7.0);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(7.0, mat[i]);
    }

    mat.set({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}});
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(i + 1.0, mat[i]);
    }

    Matrix<double, 4, 2> mat2;
    mat2.set(mat);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(i + 1.0, mat2[i]);
    }

    mat.setDiagonal(10.0);
    for (size_t i = 0; i < 8; ++i) {
        if (i == 0 || i == 3) {
            EXPECT_EQ(10.0, mat[i]);
        } else {
            EXPECT_EQ(mat2[i], mat[i]);
        }
    }

    mat.setOffDiagonal(-1.0);
    for (size_t i = 0; i < 8; ++i) {
        if (i == 0 || i == 3) {
            EXPECT_EQ(10.0, mat[i]);
        } else {
            EXPECT_EQ(-1.0, mat[i]);
        }
    }

    Vector<double, 2> row = {2.0, 3.0};
    mat.setRow(2, row);
    for (size_t i = 0; i < 8; ++i) {
        if (i == 0 || i == 3) {
            EXPECT_EQ(10.0, mat[i]);
        } else if (i == 4) {
            EXPECT_EQ(2.0, mat[i]);
        } else if (i == 5) {
            EXPECT_EQ(3.0, mat[i]);
        } else {
            EXPECT_EQ(-1.0, mat[i]);
        }
    }

    mat.set({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}});
    mat2.set({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}});
    EXPECT_TRUE(mat.isEqual(mat2));

    mat2.set({{1.01, 2.01}, {3.01, 4.01}, {4.99, 5.99}, {6.99, 7.99}});
    EXPECT_TRUE(mat.isSimilar(mat2, 0.02));
    EXPECT_FALSE(mat.isSimilar(mat2, 0.005));

    EXPECT_FALSE(mat.isSquare());
}

TEST(Matrix, BinaryOperatorMethod) {
    const Matrix<double, 2, 3> matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    Matrix<double, 2, 3> matB = matA.add(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, matB[i]);
    }

    Matrix<double, 2, 3> matC = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};
    matB = matA.add(matC);
    Matrix<double, 2, 3> ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_TRUE(ans.isEqual(matB));

    matB = matA.sub(1.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, matB[i]);
    }

    matB = matA.sub(matC);
    ans = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_TRUE(ans.isEqual(matB));

    matB = matA.mul(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), matB[i]);
    }

    Matrix<double, 3, 2> matD = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    auto matE = matA.mul(matD);
    EXPECT_EQ(13.0, matE(0, 0));
    EXPECT_EQ(41.0, matE(0, 1));
    EXPECT_EQ(34.0, matE(1, 0));
    EXPECT_EQ(89.0, matE(1, 1));

    matB = matA.div(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, matB[i]);
    }

    matB = matA.radd(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, matB[i]);
    }

    matB = matA.rsub(1.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(0.5 - i, matB[i]);
    }

    matC = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};
    matB = matA.rsub(matC);
    ans = {{2.0, -3.0, -1.0}, {5.0, -3.0, 2.0}};
    EXPECT_EQ(ans, matB);

    matB = matA.rmul(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), matB[i]);
    }

    matD = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    auto matF = matD.rmul(matA);
    EXPECT_EQ(13.0, matF(0, 0));
    EXPECT_EQ(41.0, matF(0, 1));
    EXPECT_EQ(34.0, matF(1, 0));
    EXPECT_EQ(89.0, matF(1, 1));

    matB = matA.rdiv(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 / (i + 1.0), matB[i]);
    }
}

TEST(Matrix, AugmentedOperatorMethod) {
    const Matrix<double, 2, 3> matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const Matrix<double, 2, 3> matB = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};

    Matrix<double, 2, 3> mat = matA;
    mat.iadd(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, mat[i]);
    }

    mat = matA;
    mat.iadd(matB);
    Matrix<double, 2, 3> ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat.isub(1.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, mat[i]) << i;
    }

    mat = matA;
    mat.isub(matB);
    ans = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat.imul(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), mat[i]);
    }

    Matrix<double, 5, 5> matA2;
    Matrix<double, 5, 5> matC2;
    for (int i = 0; i < 25; ++i) {
        matA2[i] = i + 1.0;
        matC2[i] = 25.0 - i;
    }
    matA2.imul(matC2);

    const Matrix<double, 5, 5> ans2 = {
        {175.0, 160.0, 145.0, 130.0, 115.0},
        {550.0, 510.0, 470.0, 430.0, 390.0},
        {925.0, 860.0, 795.0, 730.0, 665.0},
        {1300.0, 1210.0, 1120.0, 1030.0, 940.0},
        {1675.0, 1560.0, 1445.0, 1330.0, 1215.0}};
    EXPECT_EQ(ans2, matA2);

    mat = matA;
    mat.idiv(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, mat[i]);
    }
}

TEST(Matrix, ComplexGetters) {
    const Matrix<double, 2, 3> matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    EXPECT_EQ(21.0, matA.sum());
    EXPECT_DOUBLE_EQ(21.0 / 6.0, matA.avg());

    const Matrix<double, 2, 3> matB = {{3.0, -1.0, 2.0}, {-9.0, 2.0, 8.0}};
    EXPECT_EQ(-9.0, matB.min());
    EXPECT_EQ(8.0, matB.max());
    EXPECT_EQ(-1.0, matB.absmin());
    EXPECT_EQ(-9.0, matB.absmax());

    const Matrix<double, 5, 5> matC = {{3.0, -1.0, 2.0, 4.0, 5.0},
                                       {-9.0, 2.0, 8.0, -1.0, 2.0},
                                       {4.0, 3.0, 6.0, 7.0, -5.0},
                                       {-2.0, 6.0, 7.0, 1.0, 0.0},
                                       {4.0, 2.0, 3.0, 3.0, -9.0}};
    EXPECT_EQ(3.0, matC.trace());

    EXPECT_DOUBLE_EQ(-6380.0, matC.determinant());

    Matrix<double, 2, 3> mat = matA.diagonal();
    Matrix<double, 2, 3> ans = {{1.0, 0.0, 0.0}, {0.0, 5.0, 0.0}};
    EXPECT_EQ(ans, mat);

    mat = matA.offDiagonal();
    ans = {{0.0, 2.0, 3.0}, {4.0, 0.0, 6.0}};
    EXPECT_EQ(ans, mat);

    const auto matCStrictLowerTri = matC.strictLowerTri();
    Matrix<double, 5, 5> ansStrictLowerTri = {{0.0, 0.0, 0.0, 0.0, 0.0},
           {-9.0, 0.0, 0.0, 0.0, 0.0},
           {4.0, 3.0, 0.0, 0.0, 0.0},
           {-2.0, 6.0, 7.0, 0.0, 0.0},
           {4.0, 2.0, 3.0, 3.0, 0.0}};
    EXPECT_EQ(ansStrictLowerTri, matCStrictLowerTri);

    const auto matCStrictUpperTri = matC.strictUpperTri();
    Matrix<double, 5, 5> ansStrictUpperTri = {{0.0, -1.0, 2.0, 4.0, 5.0},
           {0.0, 0.0, 8.0, -1.0, 2.0},
           {0.0, 0.0, 0.0, 7.0, -5.0},
           {0.0, 0.0, 0.0, 0.0, 0.0},
           {0.0, 0.0, 0.0, 0.0, 0.0}};
    EXPECT_EQ(ansStrictUpperTri, matCStrictUpperTri);

    const auto matCLowerTri = matC.lowerTri();
    Matrix<double, 5, 5> ansLowerTri = {{3.0, 0.0, 0.0, 0.0, 0.0},
           {-9.0, 2.0, 0.0, 0.0, 0.0},
           {4.0, 3.0, 6.0, 0.0, 0.0},
           {-2.0, 6.0, 7.0, 1.0, 0.0},
           {4.0, 2.0, 3.0, 3.0, -9.0}};
    EXPECT_EQ(ansLowerTri, matCLowerTri);

    const auto matUpperTri = matC.upperTri();
    Matrix<double, 5, 5> ansUpperTri = {{3.0, -1.0, 2.0, 4.0, 5.0},
           {0.0, 2.0, 8.0, -1.0, 2.0},
           {0.0, 0.0, 6.0, 7.0, -5.0},
           {0.0, 0.0, 0.0, 1.0, 0.0},
           {0.0, 0.0, 0.0, 0.0, -9.0}};
    EXPECT_EQ(ansUpperTri, matUpperTri);

    const Matrix<float, 5, 5> matF = matC.castTo<float>();
    const Matrix<float, 5, 5> ansF = {{3.f, -1.f, 2.f, 4.f, 5.f},
                                      {-9.f, 2.f, 8.f, -1.f, 2.f},
                                      {4.f, 3.f, 6.f, 7.f, -5.f},
                                      {-2.f, 6.f, 7.f, 1.f, 0.f},
                                      {4.f, 2.f, 3.f, 3.f, -9.f}};
    EXPECT_EQ(ansF, matF);

    const Matrix<double, 3, 2> matT = matA.transposed();
    const Matrix<double, 3, 2> ansT = {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};
    EXPECT_EQ(ansT, matT);

    Matrix<double, 5, 5> matI = matC;
    Matrix<double, 5, 5> mat2I = matI.inverse();
    Matrix<double, 5, 5> ansI = {
        {0.260345, -0.0484326, -0.300157, 0.109404, 0.300627},
        {-0.215517, -0.138715, 0.188871, 0.167712, -0.255486},
        {0.294828, 0.108307, -0.315831, 0.0498433, 0.363323},
        {-0.25, -0.0227273, 0.477273, -0.136364, -0.409091},
        {0.0827586, -0.0238245, -0.0376176, 0.0570533, -0.0495298}};
    EXPECT_TRUE(mat2I.isSimilar(ansI, 1e-6));
}

TEST(Matrix, Modifiers) {
    Matrix<double, 5, 5> mat = {{3.0, -1.0, 2.0, 4.0, 5.0},
                                {-9.0, 2.0, 8.0, -1.0, 2.0},
                                {4.0, 3.0, 6.0, 7.0, -5.0},
                                {-2.0, 6.0, 7.0, 1.0, 0.0},
                                {4.0, 2.0, 3.0, 3.0, -9.0}};
    mat.transpose();

    Matrix<double, 5, 5> ans = {{3.0, -9.0, 4.0, -2.0, 4.0},
                                {-1.0, 2.0, 3.0, 6.0, 2.0},
                                {2.0, 8.0, 6.0, 7.0, 3.0},
                                {4.0, -1.0, 7.0, 1.0, 3.0},
                                {5.0, 2.0, -5.0, 0.0, -9.0}};
    EXPECT_EQ(ans, mat);

    mat = {{3.0, -1.0, 2.0, 4.0, 5.0},
           {-9.0, 2.0, 8.0, -1.0, 2.0},
           {4.0, 3.0, 6.0, 7.0, -5.0},
           {-2.0, 6.0, 7.0, 1.0, 0.0},
           {4.0, 2.0, 3.0, 3.0, -9.0}};
    mat.invert();
    ans = {
        {151 / 580.0, -309 / 6380.0, -383 / 1276.0, 349 / 3190.0, 959 / 3190.0},
        {-25 / 116.0, -177 / 1276.0, 241 / 1276.0, 107 / 638.0, -163 / 638.0},
        {171 / 580.0, 691 / 6380.0, -403 / 1276.0, 159 / 3190.0, 1159 / 3190.0},
        {-1 / 4.0, -1 / 44.0, 21 / 44.0, -3 / 22.0, -9 / 22.0},
        {12 / 145.0, -38 / 1595.0, -12 / 319.0, 91 / 1595.0, -79 / 1595.0}};
    EXPECT_TRUE(mat.isSimilar(ans, 1e-9));
}

TEST(Matrix, SetterOperators) {
    const Matrix<double, 2, 3> matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const Matrix<double, 2, 3> matB = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};

    Matrix<double, 2, 3> mat = matA;
    mat += 3.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, mat[i]);
    }

    mat = matA;
    mat += matB;
    Matrix<double, 2, 3> ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat -= 1.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, mat[i]) << i;
    }

    mat = matA;
    mat -= matB;
    ans = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat *= 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), mat[i]);
    }

    Matrix<double, 5, 5> matA2;
    Matrix<double, 5, 5> matC2;
    for (int i = 0; i < 25; ++i) {
        matA2[i] = i + 1.0;
        matC2[i] = 25.0 - i;
    }
    matA2 *= matC2;

    const Matrix<double, 5, 5> ans2 = {
        {175.0, 160.0, 145.0, 130.0, 115.0},
        {550.0, 510.0, 470.0, 430.0, 390.0},
        {925.0, 860.0, 795.0, 730.0, 665.0},
        {1300.0, 1210.0, 1120.0, 1030.0, 940.0},
        {1675.0, 1560.0, 1445.0, 1330.0, 1215.0}};
    EXPECT_EQ(ans2, matA2);

    mat = matA;
    mat /= 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, mat[i]);
    }
}

TEST(Matrix, GetterOperator) {
    Matrix<double, 2, 4> mat, mat2;
    mat.set({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
    double cnt = 1.0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(cnt, mat(i, j));
            cnt += 1.0;
        }
    }

    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(i + 1.0, mat[i]);
    }

    mat.set({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
    mat2.set({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
    EXPECT_EQ(mat, mat2);
}

TEST(Matrix, Builders) {
    const Matrix<double, 3, 4> mat = Matrix<double, 3, 4>::makeZero();
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(0.0, mat[i]);
    }

    const Matrix<double, 5, 5> mat2 = Matrix<double, 5, 5>::makeIdentity();
    for (size_t i = 0; i < 25; ++i) {
        if (i % 6 == 0) {
            EXPECT_EQ(1.0, mat2[i]);
        } else {
            EXPECT_EQ(0.0, mat2[i]);
        }
    }
}
