// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/vector.h>
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
    Vector<double, 5> vec;
    vec.set({1.0, 4.0, 3.0, -5.0, 2.0});

    EXPECT_EQ(5u, vec.size());
    EXPECT_EQ(1.0, vec[0]);
    EXPECT_EQ(4.0, vec[1]);
    EXPECT_EQ(3.0, vec[2]);
    EXPECT_EQ(-5.0, vec[3]);
    EXPECT_EQ(2.0, vec[4]);

    Vector<double, 5> vec2;
    vec2.set(4.0);

    vec2.set(vec);
    EXPECT_EQ(5u, vec2.size());
    EXPECT_EQ(1.0, vec2[0]);
    EXPECT_EQ(4.0, vec2[1]);
    EXPECT_EQ(3.0, vec2[2]);
    EXPECT_EQ(-5.0, vec2[3]);
    EXPECT_EQ(2.0, vec2[4]);

    Vector<double, 5> vec3;
    vec3.set(3.14);
    vec2.swap(vec3);

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(3.14, vec2[i]);
    }
    EXPECT_EQ(5u, vec3.size());
    EXPECT_EQ(1.0, vec3[0]);
    EXPECT_EQ(4.0, vec3[1]);
    EXPECT_EQ(3.0, vec3[2]);
    EXPECT_EQ(-5.0, vec3[3]);
    EXPECT_EQ(2.0, vec3[4]);

    vec3.setZero();
    for (size_t i = 0; i < vec3.size(); ++i) {
        EXPECT_EQ(0.0, vec3[i]);
    }

    vec3.set(vec);
    vec3.normalize();
    double denom = std::sqrt(55.0);
    EXPECT_EQ(1.0 / denom, vec3[0]);
    EXPECT_EQ(4.0 / denom, vec3[1]);
    EXPECT_EQ(3.0 / denom, vec3[2]);
    EXPECT_EQ(-5.0 / denom, vec3[3]);
    EXPECT_EQ(2.0 / denom, vec3[4]);
}

TEST(Vector, BasicGetters) {
    Vector<double, 4> vecA = {+3.0, -1.0, +2.0, 5.0};

    EXPECT_EQ(4u, vecA.size());

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

    const auto acc = vecA.constAccessor();
    EXPECT_EQ(4u, acc.size());
    EXPECT_EQ(+6.0, acc[0]);
    EXPECT_EQ(+2.5, acc[1]);
    EXPECT_EQ(-9.0, acc[2]);
    EXPECT_EQ(+8.0, acc[3]);

    vecA = {+3.0, -1.0, +2.0, 5.0};
    auto acc2 = vecA.accessor();
    acc2[0] = 6.0;
    acc2[1] = 2.5;
    acc2[2] = -9.0;
    acc2[3] = 8.0;
    EXPECT_EQ(+6.0, acc2[0]);
    EXPECT_EQ(+2.5, acc2[1]);
    EXPECT_EQ(-9.0, acc2[2]);
    EXPECT_EQ(+8.0, acc2[3]);

    EXPECT_EQ(+6.0, vecA.at(0));
    EXPECT_EQ(+2.5, vecA.at(1));
    EXPECT_EQ(-9.0, vecA.at(2));
    EXPECT_EQ(+8.0, vecA.at(3));

    vecA = {+3.0, -1.0, +2.0, 5.0};
    vecA.at(0) = 6.0;
    vecA.at(1) = 2.5;
    vecA.at(2) = -9.0;
    vecA.at(3) = 8.0;
    EXPECT_EQ(+6.0, vecA[0]);
    EXPECT_EQ(+2.5, vecA[1]);
    EXPECT_EQ(-9.0, vecA[2]);
    EXPECT_EQ(+8.0, vecA[3]);

    EXPECT_EQ(7.5, vecA.sum());
    EXPECT_EQ(7.5 / 4.0, vecA.avg());
    EXPECT_EQ(-9.0, vecA.min());
    EXPECT_EQ(8.0, vecA.max());
    EXPECT_EQ(2.5, vecA.absmin());
    EXPECT_EQ(-9.0, vecA.absmax());
    EXPECT_EQ(2u, vecA.dominantAxis());
    EXPECT_EQ(1u, vecA.subminantAxis());

    auto vecB = vecA;
    auto vecC = vecB.normalized();
    vecA.normalize();
    for (size_t i = 0; i < vecA.size(); ++i) {
        EXPECT_EQ(vecA[i], vecC[i]);
    }

    vecA.at(0) = 6.0;
    vecA.at(1) = 2.5;
    vecA.at(2) = -9.0;
    vecA.at(3) = 8.0;
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

    EXPECT_FALSE(vecA.isEqual(vecB));
    vecB = vecA;
    EXPECT_TRUE(vecA.isEqual(vecB));

    vecB[0] += 1e-8;
    vecB[1] -= 1e-8;
    vecB[2] += 1e-8;
    vecB[3] -= 1e-8;
    EXPECT_FALSE(vecA.isEqual(vecB));
    EXPECT_TRUE(vecA.isSimilar(vecB, 1e-7));
}

TEST(Vector, BinaryOperatorMethods) {
    Vector<double, 4> vecA = {+3.0, -1.0, +2.0, 5.0};
    Vector<double, 4> vecB = {+6.0, +2.5, -9.0, 8.0};
    Vector<double, 4> vecC = vecA.add(vecB);

    EXPECT_EQ(+9.0, vecC[0]);
    EXPECT_EQ(+1.5, vecC[1]);
    EXPECT_EQ(-7.0, vecC[2]);
    EXPECT_EQ(13.0, vecC[3]);

    vecC = vecA.add(3.0);
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(+2.0, vecC[1]);
    EXPECT_EQ(+5.0, vecC[2]);
    EXPECT_EQ(+8.0, vecC[3]);

    vecC = vecA.sub(vecB);
    EXPECT_EQ(-3.0, vecC[0]);
    EXPECT_EQ(-3.5, vecC[1]);
    EXPECT_EQ(11.0, vecC[2]);
    EXPECT_EQ(-3.0, vecC[3]);

    vecC = vecA.sub(4.0);
    EXPECT_EQ(-1.0, vecC[0]);
    EXPECT_EQ(-5.0, vecC[1]);
    EXPECT_EQ(-2.0, vecC[2]);
    EXPECT_EQ(+1.0, vecC[3]);

    vecC = vecA.mul(vecB);
    EXPECT_EQ(18.0, vecC[0]);
    EXPECT_EQ(-2.5, vecC[1]);
    EXPECT_EQ(-18.0, vecC[2]);
    EXPECT_EQ(40.0, vecC[3]);

    vecC = vecA.mul(2.0);
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(-2.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(10.0, vecC[3]);

    vecC = vecA.div(vecB);
    EXPECT_EQ(+0.5, vecC[0]);
    EXPECT_EQ(-0.4, vecC[1]);
    EXPECT_EQ(-2.0 / 9.0, vecC[2]);
    EXPECT_EQ(0.625, vecC[3]);

    vecC = vecA.div(0.5);
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(-2.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(10.0, vecC[3]);

    double d = vecA.dot(vecB);
    EXPECT_EQ(37.5, d);
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

    vecC = vecA * vecB;
    EXPECT_EQ(18.0, vecC[0]);
    EXPECT_EQ(-2.5, vecC[1]);
    EXPECT_EQ(-18.0, vecC[2]);
    EXPECT_EQ(40.0, vecC[3]);

    vecC = vecA * 2.0;
    EXPECT_EQ(+6.0, vecC[0]);
    EXPECT_EQ(-2.0, vecC[1]);
    EXPECT_EQ(+4.0, vecC[2]);
    EXPECT_EQ(10.0, vecC[3]);

    vecC = vecA / vecB;
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
