// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix_csr.h>
#include <jet/matrix_mxn.h>
#include <jet/vector_n.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(MatrixCsr, Constructors) {
    const MatrixCsrD emptyMat;

    EXPECT_EQ(0u, emptyMat.rows());
    EXPECT_EQ(0u, emptyMat.cols());
    EXPECT_EQ(0u, emptyMat.numberOfNonZeros());
    EXPECT_EQ(emptyMat.nonZeroBegin(), emptyMat.nonZeroEnd());
    EXPECT_EQ(1, emptyMat.rowPointersEnd() - emptyMat.rowPointersBegin());
    EXPECT_EQ(emptyMat.columnIndicesBegin(), emptyMat.columnIndicesEnd());

    const MatrixCsrD matInitLst = {
        {1.0, 0.0, 0.0, -3.0}, {0.0, 3.0, -5.0, 1.0}, {-4.0, 0.0, 1.0, 5.0}};
    EXPECT_EQ(3u, matInitLst.rows());
    EXPECT_EQ(4u, matInitLst.cols());
    EXPECT_EQ(8u, matInitLst.numberOfNonZeros());

    auto iterInitLst = matInitLst.nonZeroBegin();
    EXPECT_EQ(1.0, iterInitLst[0]);
    EXPECT_EQ(-3.0, iterInitLst[1]);
    EXPECT_EQ(3.0, iterInitLst[2]);
    EXPECT_EQ(-5.0, iterInitLst[3]);
    EXPECT_EQ(1.0, iterInitLst[4]);
    EXPECT_EQ(-4.0, iterInitLst[5]);
    EXPECT_EQ(1.0, iterInitLst[6]);
    EXPECT_EQ(5.0, iterInitLst[7]);

    const MatrixMxND matDense = {
        {1.0, 0.01, 0.0, -3.0}, {0.01, 3.0, -5.0, 1.0}, {-4.0, 0.01, 1.0, 5.0}};
    const MatrixCsrD matSparse(matDense, 0.02);
    EXPECT_EQ(3u, matSparse.rows());
    EXPECT_EQ(4u, matSparse.cols());
    EXPECT_EQ(8u, matSparse.numberOfNonZeros());

    auto iterSparse = matSparse.nonZeroBegin();
    EXPECT_EQ(1.0, iterSparse[0]);
    EXPECT_EQ(-3.0, iterSparse[1]);
    EXPECT_EQ(3.0, iterSparse[2]);
    EXPECT_EQ(-5.0, iterSparse[3]);
    EXPECT_EQ(1.0, iterSparse[4]);
    EXPECT_EQ(-4.0, iterSparse[5]);
    EXPECT_EQ(1.0, iterSparse[6]);
    EXPECT_EQ(5.0, iterSparse[7]);

    MatrixCsrD matCopied = matSparse;
    EXPECT_EQ(3u, matCopied.rows());
    EXPECT_EQ(4u, matCopied.cols());
    EXPECT_EQ(8u, matCopied.numberOfNonZeros());

    auto iterCopied = matCopied.nonZeroBegin();
    EXPECT_EQ(1.0, iterCopied[0]);
    EXPECT_EQ(-3.0, iterCopied[1]);
    EXPECT_EQ(3.0, iterCopied[2]);
    EXPECT_EQ(-5.0, iterCopied[3]);
    EXPECT_EQ(1.0, iterCopied[4]);
    EXPECT_EQ(-4.0, iterCopied[5]);
    EXPECT_EQ(1.0, iterCopied[6]);
    EXPECT_EQ(5.0, iterCopied[7]);

    const MatrixCsrD matMoved = std::move(matCopied);
    EXPECT_EQ(3u, matMoved.rows());
    EXPECT_EQ(4u, matMoved.cols());
    EXPECT_EQ(8u, matMoved.numberOfNonZeros());

    auto iterMovied = matMoved.nonZeroBegin();
    EXPECT_EQ(1.0, iterMovied[0]);
    EXPECT_EQ(-3.0, iterMovied[1]);
    EXPECT_EQ(3.0, iterMovied[2]);
    EXPECT_EQ(-5.0, iterMovied[3]);
    EXPECT_EQ(1.0, iterMovied[4]);
    EXPECT_EQ(-4.0, iterMovied[5]);
    EXPECT_EQ(1.0, iterMovied[6]);
    EXPECT_EQ(5.0, iterMovied[7]);

    EXPECT_EQ(0u, matCopied.rows());
    EXPECT_EQ(0u, matCopied.cols());
    EXPECT_EQ(0u, matCopied.numberOfNonZeros());
    EXPECT_EQ(matCopied.nonZeroBegin(), matCopied.nonZeroEnd());
    EXPECT_EQ(matCopied.rowPointersBegin(), matCopied.rowPointersEnd());
    EXPECT_EQ(matCopied.columnIndicesBegin(), matCopied.columnIndicesEnd());
}

TEST(MatrixCsr, BasicSetters) {
    // Compress initializer list
    const std::initializer_list<std::initializer_list<double>> initLst = {
        {1.0, 0.01, 0.0, -3.0}, {0.01, 3.0, -5.0, 1.0}, {-4.0, 0.01, 1.0, 5.0}};
    MatrixCsrD matInitLst;
    matInitLst.compress(initLst, 0.02);
    EXPECT_EQ(3u, matInitLst.rows());
    EXPECT_EQ(4u, matInitLst.cols());
    EXPECT_EQ(8u, matInitLst.numberOfNonZeros());

    auto iterInitLst = matInitLst.nonZeroBegin();
    EXPECT_EQ(1.0, iterInitLst[0]);
    EXPECT_EQ(-3.0, iterInitLst[1]);
    EXPECT_EQ(3.0, iterInitLst[2]);
    EXPECT_EQ(-5.0, iterInitLst[3]);
    EXPECT_EQ(1.0, iterInitLst[4]);
    EXPECT_EQ(-4.0, iterInitLst[5]);
    EXPECT_EQ(1.0, iterInitLst[6]);
    EXPECT_EQ(5.0, iterInitLst[7]);

    // Set scalar
    matInitLst.set(42.0);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(42.0, iterInitLst[i]);
    }

    // Compress dense matrix
    const MatrixMxND matDense = {
        {1.0, 0.01, 0.0, -3.0}, {0.01, 3.0, -5.0, 1.0}, {-4.0, 0.01, 1.0, 5.0}};
    MatrixCsrD matSparse;
    matSparse.compress(matDense, 0.02);
    EXPECT_EQ(3u, matSparse.rows());
    EXPECT_EQ(4u, matSparse.cols());
    EXPECT_EQ(8u, matSparse.numberOfNonZeros());

    auto iterSparse = matSparse.nonZeroBegin();
    EXPECT_EQ(1.0, iterSparse[0]);
    EXPECT_EQ(-3.0, iterSparse[1]);
    EXPECT_EQ(3.0, iterSparse[2]);
    EXPECT_EQ(-5.0, iterSparse[3]);
    EXPECT_EQ(1.0, iterSparse[4]);
    EXPECT_EQ(-4.0, iterSparse[5]);
    EXPECT_EQ(1.0, iterSparse[6]);
    EXPECT_EQ(5.0, iterSparse[7]);

    // Set other CSR mat
    matInitLst.set(matSparse);
    iterInitLst = matInitLst.nonZeroBegin();

    EXPECT_EQ(1.0, iterInitLst[0]);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(iterSparse[i], iterInitLst[i]);
    }

    // Add/set element
    MatrixCsrD matAddElem;
    matAddElem.addElement(0, 0, 1.0);
    matAddElem.setElement(0, 3, -3.0);
    matAddElem.addElement(1, 1, 3.0);
    matAddElem.setElement(1, 2, -5.0);
    matAddElem.addElement(1, 3, 1.0);
    matAddElem.setElement(2, 0, -4.0);
    matAddElem.addElement(2, 2, 1.0);
    matAddElem.setElement(2, 3, 5.0);

    EXPECT_EQ(3u, matAddElem.rows());
    EXPECT_EQ(4u, matAddElem.cols());
    EXPECT_EQ(8u, matAddElem.numberOfNonZeros());

    auto iterAddElem = matAddElem.nonZeroBegin();
    EXPECT_EQ(1.0, iterAddElem[0]);
    EXPECT_EQ(-3.0, iterAddElem[1]);
    EXPECT_EQ(3.0, iterAddElem[2]);
    EXPECT_EQ(-5.0, iterAddElem[3]);
    EXPECT_EQ(1.0, iterAddElem[4]);
    EXPECT_EQ(-4.0, iterAddElem[5]);
    EXPECT_EQ(1.0, iterAddElem[6]);
    EXPECT_EQ(5.0, iterAddElem[7]);

    matAddElem.setElement(1, 3, 7.0);
    EXPECT_EQ(7.0, iterAddElem[4]);

    // Add element in random order
    MatrixCsrD matAddElemRandom;
    matAddElemRandom.addElement(2, 2, 1.0);
    matAddElemRandom.addElement(0, 3, -3.0);
    matAddElemRandom.addElement(2, 0, -4.0);
    matAddElemRandom.addElement(1, 1, 3.0);
    matAddElemRandom.addElement(2, 3, 5.0);
    matAddElemRandom.addElement(1, 3, 1.0);
    matAddElemRandom.addElement(1, 2, -5.0);
    matAddElemRandom.addElement(0, 0, 1.0);

    EXPECT_EQ(3u, matAddElemRandom.rows());
    EXPECT_EQ(4u, matAddElemRandom.cols());
    EXPECT_EQ(8u, matAddElemRandom.numberOfNonZeros());

    auto iterAddElemRandom = matAddElemRandom.nonZeroBegin();
    EXPECT_EQ(1.0, iterAddElemRandom[0]);
    EXPECT_EQ(-3.0, iterAddElemRandom[1]);
    EXPECT_EQ(3.0, iterAddElemRandom[2]);
    EXPECT_EQ(-5.0, iterAddElemRandom[3]);
    EXPECT_EQ(1.0, iterAddElemRandom[4]);
    EXPECT_EQ(-4.0, iterAddElemRandom[5]);
    EXPECT_EQ(1.0, iterAddElemRandom[6]);
    EXPECT_EQ(5.0, iterAddElemRandom[7]);

    // Add row
    MatrixCsrD matAddRow;
    matAddRow.addRow({1.0, -3.0}, {0, 3});
    matAddRow.addRow({3.0, -5.0, 1.0}, {1, 2, 3});
    matAddRow.addRow({-4.0, 1.0, 5.0}, {0, 2, 3});

    EXPECT_EQ(3u, matAddRow.rows());
    EXPECT_EQ(4u, matAddRow.cols());
    EXPECT_EQ(8u, matAddRow.numberOfNonZeros());

    auto iterAddRow = matAddRow.nonZeroBegin();
    EXPECT_EQ(1.0, iterAddRow[0]);
    EXPECT_EQ(-3.0, iterAddRow[1]);
    EXPECT_EQ(3.0, iterAddRow[2]);
    EXPECT_EQ(-5.0, iterAddRow[3]);
    EXPECT_EQ(1.0, iterAddRow[4]);
    EXPECT_EQ(-4.0, iterAddRow[5]);
    EXPECT_EQ(1.0, iterAddRow[6]);
    EXPECT_EQ(5.0, iterAddRow[7]);

    // Clear
    matAddRow.clear();
    EXPECT_EQ(0u, matAddRow.rows());
    EXPECT_EQ(0u, matAddRow.cols());
    EXPECT_EQ(0u, matAddRow.numberOfNonZeros());
    EXPECT_EQ(1u, matAddRow.rowPointersEnd() - matAddRow.rowPointersBegin());
    EXPECT_EQ(0u, matAddRow.rowPointersBegin()[0]);
}

TEST(MatrixCsr, BinaryOperatorMethods) {
    const MatrixCsrD matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const MatrixCsrD addResult1 = matA.add(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, addResult1.nonZero(i));
    }

    const MatrixCsrD matC = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};
    const MatrixCsrD addResult2 = matA.add(matC);
    const MatrixCsrD addAns1 = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_TRUE(addAns1.isEqual(addResult2));

    const MatrixCsrD matD = {{3.0, 0.0, 2.0}, {0.0, 2.0, 0.0}};
    const MatrixCsrD addResult3 = matA.add(matD);
    const MatrixCsrD addAns2 = {{4.0, 2.0, 5.0}, {4.0, 7.0, 6.0}};
    EXPECT_TRUE(addAns2.isEqual(addResult3));

    const MatrixCsrD matE = {{3.0, 0.0, 2.0}, {0.0, 0.0, 0.0}};
    const MatrixCsrD addResult4 = matA.add(matE);
    const MatrixCsrD addAns3 = {{4.0, 2.0, 5.0}, {4.0, 5.0, 6.0}};
    EXPECT_TRUE(addAns3.isEqual(addResult4));

    const MatrixCsrD subResult1 = matA.sub(1.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, subResult1.nonZero(i));
    }

    const MatrixCsrD subResult2 = matA.sub(matC);
    const MatrixCsrD ans2 = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_TRUE(ans2.isSimilar(subResult2));

    const MatrixCsrD matB = matA.mul(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), matB.nonZero(i));
    }

    const VectorND vecA = {-1.0, 9.0, 8.0};
    const VectorND vecB = matA.mul(vecA);
    const VectorND ansV = {41.0, 89.0};
    EXPECT_TRUE(ansV.isEqual(vecB));

    const MatrixCsrD matF = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    const MatrixCsrD matG = matA.mul(matF);
    const MatrixCsrD ans3 = {{13.0, 41.0}, {34.0, 89.0}};
    EXPECT_TRUE(ans3.isEqual(matG));

    const MatrixCsrD matH = matA.div(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, matH.nonZero(i));
    }

    const MatrixCsrD matI = matA.radd(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, matI.nonZero(i));
    }

    const MatrixCsrD matJ = matA.rsub(1.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(0.5 - i, matJ.nonZero(i));
    }

    const MatrixCsrD matK = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};
    const MatrixCsrD matL = matA.rsub(matK);
    const MatrixCsrD ans4 = {{2.0, -3.0, -1.0}, {5.0, -3.0, 2.0}};
    EXPECT_EQ(ans4, matL);

    const MatrixCsrD matM = matA.rmul(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), matM.nonZero(i));
    }

    const MatrixCsrD matP = matA.rdiv(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 / (i + 1.0), matP.nonZero(i));
    }
}

TEST(MatrixCsr, AugmentedMethods) {
    const MatrixCsrD matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const MatrixCsrD matB = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};

    MatrixCsrD mat = matA;
    mat.iadd(3.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, mat.nonZero(i));
    }

    mat = matA;
    mat.iadd(matB);
    MatrixCsrD ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat.isub(1.5);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, mat.nonZero(i));
    }

    mat = matA;
    mat.isub(matB);
    ans = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat.imul(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), mat.nonZero(i));
    }

    MatrixCsrD matA2;
    MatrixCsrD matC2;
    int cnt = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            matA2.setElement(i, j, cnt + 1.0);
            matC2.setElement(i, j, 25.0 - cnt);
            ++cnt;
        }
    }
    matA2.imul(matC2);

    MatrixCsrD ans2 = {{175.0, 160.0, 145.0, 130.0, 115.0},
                       {550.0, 510.0, 470.0, 430.0, 390.0},
                       {925.0, 860.0, 795.0, 730.0, 665.0},
                       {1300.0, 1210.0, 1120.0, 1030.0, 940.0},
                       {1675.0, 1560.0, 1445.0, 1330.0, 1215.0}};
    EXPECT_EQ(ans2, matA2);

    mat = matA;
    mat.idiv(2.0);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, mat.nonZero(i));
    }
}

TEST(MatrixCsr, ComplexGetters) {
    const MatrixCsrD matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};

    EXPECT_EQ(21.0, matA.sum());
    EXPECT_DOUBLE_EQ(21.0 / 6.0, matA.avg());

    const MatrixCsrD matB = {{3.0, -1.0, 2.0}, {-9.0, 2.0, 8.0}};
    EXPECT_EQ(-9.0, matB.min());
    EXPECT_EQ(8.0, matB.max());
    EXPECT_EQ(-1.0, matB.absmin());
    EXPECT_EQ(-9.0, matB.absmax());

    const MatrixCsrD matC = {{3.0, -1.0, 2.0, 4.0, 5.0},
                             {-9.0, 2.0, 8.0, -1.0, 2.0},
                             {4.0, 3.0, 6.0, 7.0, -5.0},
                             {-2.0, 6.0, 7.0, 1.0, 0.0},
                             {4.0, 2.0, 3.0, 3.0, -9.0}};
    EXPECT_EQ(3.0, matC.trace());

    const MatrixCsrF matF = matC.castTo<float>();
    const MatrixCsrF ansF = {{3.f, -1.f, 2.f, 4.f, 5.f},
                             {-9.f, 2.f, 8.f, -1.f, 2.f},
                             {4.f, 3.f, 6.f, 7.f, -5.f},
                             {-2.f, 6.f, 7.f, 1.f, 0.f},
                             {4.f, 2.f, 3.f, 3.f, -9.f}};
    EXPECT_EQ(ansF, matF);
}

TEST(MatrixCsr, SetterOperators) {
    const MatrixCsrD matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const MatrixCsrD matB = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};

    MatrixCsrD mat = matA;
    mat += 3.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, mat.nonZero(i));
    }

    mat = matA;
    mat += matB;
    MatrixCsrD ans = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat -= 1.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, mat.nonZero(i)) << i;
    }

    mat = matA;
    mat -= matB;
    ans = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_EQ(ans, mat);

    mat = matA;
    mat *= 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), mat.nonZero(i));
    }

    MatrixCsrD matA2;
    MatrixCsrD matC2;
    int cnt = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            matA2.setElement(i, j, cnt + 1.0);
            matC2.setElement(i, j, 25.0 - cnt);
            ++cnt;
        }
    }
    matA2 *= matC2;

    const MatrixCsrD ans2 = {{175.0, 160.0, 145.0, 130.0, 115.0},
                             {550.0, 510.0, 470.0, 430.0, 390.0},
                             {925.0, 860.0, 795.0, 730.0, 665.0},
                             {1300.0, 1210.0, 1120.0, 1030.0, 940.0},
                             {1675.0, 1560.0, 1445.0, 1330.0, 1215.0}};
    EXPECT_EQ(ans2, matA2);

    mat = matA;
    mat /= 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, mat.nonZero(i));
    }
}

TEST(MatrixCsr, GetterOperators) {
    const MatrixMxND matDense = {
        {1.0, 0.0, 0.0, -3.0}, {0.0, 3.0, -5.0, 1.0}, {-4.0, 0.0, 1.0, 5.0}};
    const MatrixCsrD matSparse(matDense, 0.02);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_EQ(matDense(i, j), matSparse(i, j));
        }
    }
}

TEST(MatrixCsr, Builders) {
    const MatrixCsrD matIden = MatrixCsrD::makeIdentity(5);
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            if (i == j) {
                EXPECT_EQ(1.0, matIden(i, j));
            } else {
                EXPECT_EQ(0.0, matIden(i, j));
            }
        }
    }
}

TEST(MatrixCsr, OperatorOverloadings) {
    const MatrixCsrD matA = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const MatrixCsrD addResult1 = matA + 3.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, addResult1.nonZero(i));
    }

    const MatrixCsrD matC = {{3.0, -1.0, 2.0}, {9.0, 2.0, 8.0}};
    const MatrixCsrD addResult2 = matA + matC;
    const MatrixCsrD addAns1 = {{4.0, 1.0, 5.0}, {13.0, 7.0, 14.0}};
    EXPECT_TRUE(addAns1.isEqual(addResult2));

    const MatrixCsrD matD = {{3.0, 0.0, 2.0}, {0.0, 2.0, 0.0}};
    const MatrixCsrD addResult3 = matA + matD;
    const MatrixCsrD addAns2 = {{4.0, 2.0, 5.0}, {4.0, 7.0, 6.0}};
    EXPECT_TRUE(addAns2.isEqual(addResult3));

    const MatrixCsrD matE = {{3.0, 0.0, 2.0}, {0.0, 0.0, 0.0}};
    const MatrixCsrD addResult4 = matA + matE;
    const MatrixCsrD addAns3 = {{4.0, 2.0, 5.0}, {4.0, 5.0, 6.0}};
    EXPECT_TRUE(addAns3.isEqual(addResult4));

    const MatrixCsrD subResult1 = matA - 1.5;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i - 0.5, subResult1.nonZero(i));
    }

    const MatrixCsrD subResult2 = matA - matC;
    const MatrixCsrD ans2 = {{-2.0, 3.0, 1.0}, {-5.0, 3.0, -2.0}};
    EXPECT_TRUE(ans2.isSimilar(subResult2));

    const MatrixCsrD matB = matA * 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), matB.nonZero(i));
    }

    const VectorND vecA = {-1.0, 9.0, 8.0};
    const VectorND vecB = matA * vecA;
    const VectorND ansV = {41.0, 89.0};
    EXPECT_TRUE(ansV.isEqual(vecB));

    const MatrixCsrD matF = {{3.0, -1.0}, {2.0, 9.0}, {2.0, 8.0}};
    const MatrixCsrD matG = matA * matF;
    const MatrixCsrD ans3 = {{13.0, 41.0}, {34.0, 89.0}};
    EXPECT_TRUE(ans3.isEqual(matG));

    const MatrixCsrD matH = matA / 2.0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ((i + 1.0) / 2.0, matH.nonZero(i));
    }

    const MatrixCsrD matI = 3.5 + matA;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(i + 4.5, matI.nonZero(i));
    }

    const MatrixCsrD matJ = 1.5 - matA;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(0.5 - i, matJ.nonZero(i));
    }

    const MatrixCsrD matM = 2.0 * matA;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 * (i + 1.0), matM.nonZero(i));
    }

    const MatrixCsrD matP = 2.0 / matA;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(2.0 / (i + 1.0), matP.nonZero(i));
    }
}
