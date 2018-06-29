// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/matrix.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(Vector, Constructors) {
    Vector<double, 5> vec1;

    EXPECT_DOUBLE_EQ(0.0, vec1[0]);
    EXPECT_DOUBLE_EQ(0.0, vec1[1]);
    EXPECT_DOUBLE_EQ(0.0, vec1[2]);
    EXPECT_DOUBLE_EQ(0.0, vec1[3]);
    EXPECT_DOUBLE_EQ(0.0, vec1[4]);

    Vector<double, 5> vec2({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_DOUBLE_EQ(1.0, vec2[0]);
    EXPECT_DOUBLE_EQ(2.0, vec2[1]);
    EXPECT_DOUBLE_EQ(3.0, vec2[2]);
    EXPECT_DOUBLE_EQ(4.0, vec2[3]);
    EXPECT_DOUBLE_EQ(5.0, vec2[4]);

    Vector<double, 5> vec3(vec2);

    EXPECT_DOUBLE_EQ(1.0, vec3[0]);
    EXPECT_DOUBLE_EQ(2.0, vec3[1]);
    EXPECT_DOUBLE_EQ(3.0, vec3[2]);
    EXPECT_DOUBLE_EQ(4.0, vec3[3]);
    EXPECT_DOUBLE_EQ(5.0, vec3[4]);
}

TEST(Vector, BasicSetters) {
    Vector<double, 5> vec{1.0, 2.0, 3.0, 4.0, 5.0};
    vec.fill(0.0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(0.0, vec[i]);
    }

    vec.fill(4.0);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(4.0, vec[i]);
    }

    vec.fill([](size_t i) -> double { return i * 5.0; });
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(i * 5.0, vec[i]);
    }

    vec.fill([](size_t i, size_t j) -> double { return i + 8.0 * (j + 1); });
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(i + 8.0, vec[i]);
    }

    Vector<double, 5> vec2{5.0, 4.0, 3.0, 2.0, 1.0};
    vec.swap(vec2);
    for (int i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(5.0 - i, vec[i]);
        EXPECT_DOUBLE_EQ(i + 8.0, vec2[i]);
    }

    vec.normalize();
    double len = 0.0;
    for (int i = 0; i < 5; ++i) {
        len += vec[i] * vec[i];
    }
    EXPECT_NEAR(1.0, len, 1e-10);
}

TEST(Vector, BasicGetters) {
    Vector<double, 4> vecA = {+3.0, -1.0, +2.0, 5.0};

    EXPECT_EQ(4u, vecA.rows());
    EXPECT_EQ(1u, vecA.cols());

    const double* data = vecA.data();
    EXPECT_EQ(3.0, data[0]);
    EXPECT_EQ(-1.0, data[1]);
    EXPECT_EQ(2.0, data[2]);
    EXPECT_EQ(5.0, data[3]);

    double* data2 = vecA.data();
    data2[0] = 6.0;
    data2[1] = 2.5;
    data2[2] = -9.0;
    data2[3] = 8.0;
    EXPECT_EQ(+6.0, vecA[0]);
    EXPECT_EQ(+2.5, vecA[1]);
    EXPECT_EQ(-9.0, vecA[2]);
    EXPECT_EQ(+8.0, vecA[3]);

    const auto iter = vecA.begin();
    EXPECT_EQ(+6.0, iter[0]);
    EXPECT_EQ(+2.5, iter[1]);
    EXPECT_EQ(-9.0, iter[2]);
    EXPECT_EQ(+8.0, iter[3]);

    vecA = {+3.0, -1.0, +2.0, 5.0};
    auto iter2 = vecA.begin();
    iter2[0] = 6.0;
    iter2[1] = 2.5;
    iter2[2] = -9.0;
    iter2[3] = 8.0;
    EXPECT_EQ(+6.0, iter2[0]);
    EXPECT_EQ(+2.5, iter2[1]);
    EXPECT_EQ(-9.0, iter2[2]);
    EXPECT_EQ(+8.0, iter2[3]);

    auto d = vecA.end() - vecA.begin();
    EXPECT_EQ(4, d);
    EXPECT_EQ(7.5, vecA.sum());
    EXPECT_EQ(7.5 / 4.0, vecA.avg());
    EXPECT_EQ(-9.0, vecA.min());
    EXPECT_EQ(8.0, vecA.max());
    EXPECT_EQ(2.5, vecA.absmin());
    EXPECT_EQ(-9.0, vecA.absmax());
    EXPECT_EQ(2u, vecA.dominantAxis());
    EXPECT_EQ(1u, vecA.subminantAxis());

    auto vecB = vecA;
    Vector<double, 4> vecC = vecB.normalized();
    vecA.normalize();
    for (size_t i = 0; i < vecA.rows(); ++i) {
        EXPECT_EQ(vecA[i], vecC[i]);
    }

    vecA[0] = 6.0;
    vecA[1] = 2.5;
    vecA[2] = -9.0;
    vecA[3] = 8.0;
    double lenSqr = vecA.lengthSquared();
    EXPECT_EQ(187.25, lenSqr);

    double len = vecA.length();
    EXPECT_EQ(std::sqrt(187.25), len);

    vecA = {+3.0, -1.0, +2.0, 5.0};
    vecB = {+6.0, +2.5, -9.0, 8.0};
    double distSq = vecA.distanceSquaredTo(vecB);
    EXPECT_EQ(151.25, distSq);

    double dist = vecA.distanceTo(vecB);
    EXPECT_EQ(std::sqrt(151.25), dist);

    Vector<float, 4> vecD = vecA.castTo<float>();
    EXPECT_EQ(+3.f, vecD[0]);
    EXPECT_EQ(-1.f, vecD[1]);
    EXPECT_EQ(+2.f, vecD[2]);
    EXPECT_EQ(+5.f, vecD[3]);
    /*
        EXPECT_FALSE(vecA.isEqual(vecB));
        vecB = vecA;
        EXPECT_TRUE(vecA.isEqual(vecB));

        vecB[0] += 1e-8;
        vecB[1] -= 1e-8;
        vecB[2] += 1e-8;
        vecB[3] -= 1e-8;
        EXPECT_FALSE(vecA.isEqual(vecB));
        EXPECT_TRUE(vecA.isSimilar(vecB, 1e-7));
    */
}

TEST(Vector, BinaryOperators) {
    Vector<double, 4> vecA = {+3.0, -1.0, +2.0, 5.0};
    Vector<double, 4> vecB = {+6.0, +2.5, -9.0, 8.0};
    Vector<double, 4> vecC = vecA + vecB;

    EXPECT_EQ(+9.0, vecC[0]);
    EXPECT_EQ(+1.5, vecC[1]);
    EXPECT_EQ(-7.0, vecC[2]);
    EXPECT_EQ(13.0, vecC[3]);

    vecC = vecA + 3.0;
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(+2.0, vecC[1]);
    EXPECT_EQ(+5.0, vecC[2]);
    EXPECT_EQ(+8.0, vecC[3]);

    vecC = 2.0 + vecA;
    EXPECT_EQ(+5.0, vecC[0]);
    EXPECT_EQ(+1.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(+7.0, vecC[3]);

    vecC = vecA - vecB;
    EXPECT_EQ(-3.0, vecC[0]);
    EXPECT_EQ(-3.5, vecC[1]);
    EXPECT_EQ(11.0, vecC[2]);
    EXPECT_EQ(-3.0, vecC[3]);

    vecC = 6.0 - vecA;
    EXPECT_EQ(+3.0, vecC[0]);
    EXPECT_EQ(+7.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(+1.0, vecC[3]);

    vecC = vecA - 4.0;
    EXPECT_EQ(-1.0, vecC[0]);
    EXPECT_EQ(-5.0, vecC[1]);
    EXPECT_EQ(-2.0, vecC[2]);
    EXPECT_EQ(+1.0, vecC[3]);

    // Deprecated
    // vecC = vecA * vecB;
    vecC = elemMul(vecA, vecB);
    EXPECT_EQ(18.0, vecC[0]);
    EXPECT_EQ(-2.5, vecC[1]);
    EXPECT_EQ(-18.0, vecC[2]);
    EXPECT_EQ(40.0, vecC[3]);

    vecC = vecA * 2.0;
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(-2.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(10.0, vecC[3]);

    // Deprecated
    // vecC = vecA / vecB;
    vecC = elemDiv(vecA, vecB);
    EXPECT_EQ(+0.5, vecC[0]);
    EXPECT_EQ(-0.4, vecC[1]);
    EXPECT_EQ(-2.0 / 9.0, vecC[2]);
    EXPECT_EQ(0.625, vecC[3]);

    vecC = vecA / 0.5;
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(-2.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(10.0, vecC[3]);

    vecC = 2.0 / vecA;
    EXPECT_EQ(+2.0 / 3.0, vecC[0]);
    EXPECT_EQ(-2.0, vecC[1]);
    EXPECT_EQ(+1.0, vecC[2]);
    EXPECT_EQ(+0.4, vecC[3]);

    vecC = 3.0 / (0.5 * vecA + 2.0 * vecB);
    EXPECT_EQ(3.0 / 13.5, vecC[0]);
    EXPECT_EQ(2.0 / 3.0, vecC[1]);
    EXPECT_EQ(3.0 / -17.0, vecC[2]);
    EXPECT_EQ(3.0 / 18.5, vecC[3]);
}
