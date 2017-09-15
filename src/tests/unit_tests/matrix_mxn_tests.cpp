// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix_mxn.h>

#include <gtest/gtest.h>

#include <algorithm>

using namespace jet;

namespace {

template <typename T>
inline std::ostream& operator<<(std::ostream& strm, const MatrixMxN<T>& m) {
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            strm << m(i, j) << ' ';
        }
        strm << '\n';
    }
    return strm;
}
}

TEST(MatrixMxN, Constructors) {
    MatrixMxND mat;
    EXPECT_EQ(0u, mat.rows());
    EXPECT_EQ(0u, mat.cols());

    MatrixMxND mat3(4, 2, 5.0);
    EXPECT_EQ(4u, mat3.rows());
    EXPECT_EQ(2u, mat3.cols());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(5.0, mat3[i]);
    }

    MatrixMxND mat4 = {
        {1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}};
    EXPECT_EQ(3u, mat4.rows());
    EXPECT_EQ(4u, mat4.cols());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(i + 1.0, mat4[i]);
    }

    MatrixMxND mat5(mat4);
    EXPECT_EQ(3u, mat5.rows());
    EXPECT_EQ(4u, mat5.cols());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(mat4[i], mat5[i]);
    }

    double ans6[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    MatrixMxND mat6(3, 2, ans6);
    EXPECT_EQ(3u, mat6.rows());
    EXPECT_EQ(2u, mat6.cols());
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(ans6[i], mat6[i]);
    }
}

TEST(MatrixMxN, BasicSetters) {
    MatrixMxND mat;
    mat.resize(4, 2, 5.0);
    EXPECT_EQ(4u, mat.rows());
    EXPECT_EQ(2u, mat.cols());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(5.0, mat[i]);
    }

    mat.set(7.0);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(7.0, mat[i]);
    }

    mat.set({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
    EXPECT_EQ(2u, mat.rows());
    EXPECT_EQ(4u, mat.cols());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(i + 1.0, mat[i]);
    }

    MatrixMxND mat2;
    mat2.set(mat);
    EXPECT_EQ(2u, mat2.rows());
    EXPECT_EQ(4u, mat2.cols());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(i + 1.0, mat2[i]);
    }

    double arr[] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    mat.set(4, 2, arr);
    EXPECT_EQ(4u, mat.rows());
    EXPECT_EQ(2u, mat.cols());
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(arr[i], mat[i]);
    }

    mat.setDiagonal(10.0);
    for (size_t i = 0; i < 8; ++i) {
        if (i == 0 || i == 3) {
            EXPECT_EQ(10.0, mat[i]);
        } else {
            EXPECT_EQ(arr[i], mat[i]);
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

    VectorND row = {2.0, 3.0};
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

    mat.set({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
    mat2.set({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}});
    EXPECT_TRUE(mat.isEqual(mat2));

    mat2.set({{1.01, 2.01, 3.01, 4.01}, {4.99, 5.99, 6.99, 7.99}});
    EXPECT_TRUE(mat.isSimilar(mat2, 0.02));
    EXPECT_FALSE(mat.isSimilar(mat2, 0.005));

    EXPECT_FALSE(mat.isSquare());
    mat.set({{1.0, 2.0}, {3.0, 4.0}});
    EXPECT_TRUE(mat.isSquare());
}

TEST(MatrixMxN, BinaryOperatorMethod) {
    const MatrixMxND matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    MatrixMxND matB = matA.add(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, matB[i]);
    }

    MatrixMxND matC = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};
    matB = matA.add(matC);
    MatrixMxND ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
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

    matC = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    matB = matA.mul(matC);
    ans = {{13.0, 41.0}, {34.0, 89.0}};
    EXPECT_TRUE(ans.isEqual(matB));

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

    matC = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    matB = matC.rmul(matA);
    ans = {{13.0, 41.0}, {34.0, 89.0}};
    EXPECT_EQ(ans, matB);

    matB = matA.rdiv(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 / (i + 1.0), matB[i]);
    }
}

TEST(MatrixMxN, AugmentedOperatorMethod) {
    const MatrixMxND matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const MatrixMxND matB = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};

    MatrixMxND mat = matA;
    mat.iadd(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, mat[i]);
    }

    mat = matA;
    mat.iadd(matB);
    MatrixMxND ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
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

    mat = matA;
    const MatrixMxND matC = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    mat.imul(matC);
    EXPECT_EQ(2u, mat.rows());
    EXPECT_EQ(2u, mat.cols());
    ans = {{13.0, 41.0}, {34.0, 89.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat.idiv(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, mat[i]);
    }
}

TEST(MatrixMxN, ComplexGetters) {
    const MatrixMxND matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    EXPECT_EQ(21.0, matA.sum());
    EXPECT_DOUBLE_EQ(21.0 / 6.0, matA.avg());

    const MatrixMxND matB = {{3.0, -1.0, 2.0}, {-9.0, 2.0, 8.0}};
    EXPECT_EQ(-9.0, matB.min());
    EXPECT_EQ(8.0, matB.max());
    EXPECT_EQ(-1.0, matB.absmin());
    EXPECT_EQ(-9.0, matB.absmax());

    const MatrixMxND matC = {
        {3.0, -1.0, 2.0}, {-9.0, 2.0, 8.0}, {4.0, 3.0, 6.0}};
    EXPECT_EQ(11.0, matC.trace());

    EXPECT_DOUBLE_EQ(-192.0, matC.determinant());

    MatrixMxND mat = matA.diagonal();
    MatrixMxND ans = {{1.0, 0.0, 0.0}, {0.0, 5.0, 0.0}};
    EXPECT_EQ(ans, mat);

    mat = matA.offDiagonal();
    ans = {{0.0, 2.0, 3.0}, {4.0, 0.0, 6.0}};
    EXPECT_EQ(ans, mat);

    mat = matC.strictLowerTri();
    ans = {{0.0, 0.0, 0.0}, {-9.0, 0.0, 0.0}, {4.0, 3.0, 0.0}};
    EXPECT_EQ(ans, mat);

    mat = matC.strictUpperTri();
    ans = {{0.0, -1.0, 2.0}, {0.0, 0.0, 8.0}, {0.0, 0.0, 0.0}};
    EXPECT_EQ(ans, mat);

    mat = matC.lowerTri();
    ans = {{3.0, 0.0, 0.0}, {-9.0, 2.0, 0.0}, {4.0, 3.0, 6.0}};
    EXPECT_EQ(ans, mat);

    mat = matC.upperTri();
    ans = {{3.0, -1.0, 2.0}, {0.0, 2.0, 8.0}, {0.0, 0.0, 6.0}};
    EXPECT_EQ(ans, mat);

    MatrixMxNF matF = matC.castTo<float>();
    MatrixMxNF ansF = {{3.f, -1.f, 2.f}, {-9.f, 2.f, 8.f}, {4.f, 3.f, 6.f}};
    EXPECT_EQ(ansF, matF);

    mat = matA.transposed();
    ans = {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};
    EXPECT_EQ(ans, mat);

    mat = {{1.0, 2.0, 3.0}, {2.0, 5.0, 3.0}, {1.0, 0.0, 8.0}};
    MatrixMxND mat2 = mat.inverse();
    ans = {{-40.0, 16.0, 9.0}, {13.0, -5.0, -3.0}, {5.0, -2.0, -1.0}};
    EXPECT_TRUE(mat2.isSimilar(ans, 1e-9));

    mat = {{1.0, 2.0, 3.0}, {0.0, 1.0, 4.0}, {5.0, 6.0, 0.0}};
    mat2 = mat.inverse();
    ans = {{-24.0, 18.0, 5.0}, {20.0, -15.0, -4.0}, {-5.0, 4.0, 1.0}};
    EXPECT_TRUE(mat2.isSimilar(ans, 1e-9));

    mat = {{0.0, 1.0, 4.0}, {1.0, 2.0, 3.0}, {5.0, 6.0, 0.0}};
    mat2 = mat.inverse();
    ans = {{18.0, -24.0, 5.0}, {-15.0, 20.0, -4.0}, {4.0, -5.0, 1.0}};
    EXPECT_TRUE(mat2.isSimilar(ans, 1e-9));
}

TEST(MatrixMxN, Modifiers) {
    MatrixMxND mat = {{9.0, -8.0, 7.0}, {-6.0, 5.0, -4.0}, {3.0, -2.0, 1.0}};
    mat.transpose();

    MatrixMxND ans = {{9.0, -6.0, 3.0}, {-8.0, 5.0, -2.0}, {7.0, -4.0, 1.0}};
    EXPECT_EQ(ans, mat);

    mat = {{1.0, 2.0, 3.0}, {2.0, 5.0, 3.0}, {1.0, 0.0, 8.0}};
    mat.invert();
    ans = {{-40.0, 16.0, 9.0}, {13.0, -5.0, -3.0}, {5.0, -2.0, -1.0}};
    EXPECT_TRUE(mat.isSimilar(ans, 1e-9));
}

TEST(MatrixMxN, SetterOperators) {
    const MatrixMxND matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const MatrixMxND matB = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};

    MatrixMxND mat = matA;
    mat += 3.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, mat[i]);
    }

    mat = matA;
    mat += matB;
    MatrixMxND ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
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

    mat = matA;
    const MatrixMxND matC = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    mat *= matC;
    EXPECT_EQ(2u, mat.rows());
    EXPECT_EQ(2u, mat.cols());
    ans = {{13.0, 41.0}, {34.0, 89.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat /= 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, mat[i]);
    }
}

TEST(MatrixMxN, GetterOperator) {
    MatrixMxND mat, mat2;
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

TEST(MatrixMxN, Builders) {
    MatrixMxND mat = MatrixMxND::makeZero(3, 4);
    EXPECT_EQ(3u, mat.rows());
    EXPECT_EQ(4u, mat.cols());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(0.0, mat[i]);
    }

    mat = MatrixMxND::makeIdentity(5);
    EXPECT_EQ(5u, mat.rows());
    EXPECT_EQ(5u, mat.cols());
    for (size_t i = 0; i < 25; ++i) {
        if (i % 6 == 0) {
            EXPECT_EQ(1.0, mat[i]);
        } else {
            EXPECT_EQ(0.0, mat[i]);
        }
    }
}
