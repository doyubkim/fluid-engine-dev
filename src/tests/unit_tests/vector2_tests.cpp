// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/vector3.h>
#include <gtest/gtest.h>
#include <algorithm>

using namespace jet;

TEST(Vector2, Constructors) {
    Vector2F vec;
    EXPECT_FLOAT_EQ(0.f, vec.x);
    EXPECT_FLOAT_EQ(0.f, vec.y);

    Vector2F vec2(5.f, 3.f);
    EXPECT_FLOAT_EQ(5.f, vec2.x);
    EXPECT_FLOAT_EQ(3.f, vec2.y);

    Vector2F vec5 = { 7.f, 6.f };
    EXPECT_FLOAT_EQ(7.f, vec5.x);
    EXPECT_FLOAT_EQ(6.f, vec5.y);

    Vector2F vec6(vec5);
    EXPECT_FLOAT_EQ(7.f, vec6.x);
    EXPECT_FLOAT_EQ(6.f, vec6.y);
}

TEST(Vector2, SetMethods) {
    Vector2F vec;
    vec.set(4.f, 2.f);
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(2.f, vec.y);

    auto lst = {0.f, 5.f};
    vec.set(lst);
    EXPECT_FLOAT_EQ(0.f, vec.x);
    EXPECT_FLOAT_EQ(5.f, vec.y);

    vec.set(Vector2F(9.f, 8.f));
    EXPECT_FLOAT_EQ(9.f, vec.x);
    EXPECT_FLOAT_EQ(8.f, vec.y);
}

TEST(Vector2, BasicSetterMethods) {
    Vector2F vec(3.f, 9.f);
    vec.setZero();
    EXPECT_FLOAT_EQ(0.f, vec.x);
    EXPECT_FLOAT_EQ(0.f, vec.y);

    vec.set(4.f, 2.f);
    vec.normalize();
    float len = vec.x * vec.x + vec.y * vec.y;
    EXPECT_NEAR(len, 1.f, 1e-6);
}

TEST(Vector2, BinaryOperatorMethods) {
    Vector2F vec(3.f, 9.f);
    vec = vec.add(4.f);
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(13.f, vec.y);

    vec = vec.add(Vector2F(-2.f, 1.f));
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(14.f, vec.y);

    vec = vec.sub(8.f);
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec = vec.sub(Vector2F(-5.f, 3.f));
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);

    vec = vec.mul(2.f);
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec = vec.mul(Vector2F(3.f, -2.f));
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);

    vec = vec.div(4.f);
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);

    vec = vec.div(Vector2F(3.f, -1.f));
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);

    float d = vec.dot(Vector2F(4.f, 2.f));
    EXPECT_FLOAT_EQ(d, 10.f);

    float c = vec.cross(Vector2F(5.f, -7.f));
    EXPECT_FLOAT_EQ(c, -22.f);
}

TEST(Vector2, BinaryInverseOperatorMethods) {
    Vector2F vec(3.f, 9.f);
    vec = vec.rsub(8.f);
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(-1.f, vec.y);

    vec = vec.rsub(Vector2F(-5.f, 3.f));
    EXPECT_FLOAT_EQ(-10.f, vec.x);
    EXPECT_FLOAT_EQ(4.f, vec.y);

    vec = Vector2F(-4.f, -3.f);
    vec = vec.rdiv(12.f);
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, -4.f);

    vec = vec.rdiv(Vector2F(3.f, -16.f));
    EXPECT_FLOAT_EQ(-1.f, vec.x);
    EXPECT_FLOAT_EQ(4.f, vec.y);

    float c = vec.rcross(Vector2F(5.f, -7.f));
    EXPECT_FLOAT_EQ(c, 13.f);
}

TEST(Vector2, AugmentedOperatorMethods) {
    Vector2F vec(3.f, 9.f);
    vec.iadd(4.f);
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, 13.f);

    vec.iadd(Vector2F(-2.f, 1.f));
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, 14.f);

    vec.isub(8.f);
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec.isub(Vector2F(-5.f, 3.f));
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);

    vec.imul(2.f);
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec.imul(Vector2F(3.f, -2.f));
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);

    vec.idiv(4.f);
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);

    vec.idiv(Vector2F(3.f, -1.f));
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
}

TEST(Vector2, AtMethod) {
    Vector2F vec(8.f, 9.f);
    EXPECT_FLOAT_EQ(vec.at(0), 8.f);
    EXPECT_FLOAT_EQ(vec.at(1), 9.f);

    vec.at(0) = 7.f;
    vec.at(1) = 6.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
}

TEST(Vector2, BasicGetterMethods) {
    Vector2F vec(3.f, 7.f), vec2(-3.f, -7.f);

    float sum = vec.sum();
    EXPECT_FLOAT_EQ(sum, 10.f);

    float avg = vec.avg();
    EXPECT_FLOAT_EQ(avg, 5.f);

    float min = vec.min();
    EXPECT_FLOAT_EQ(min, 3.f);

    float max = vec.max();
    EXPECT_FLOAT_EQ(max, 7.f);

    float absmin = vec2.absmin();
    EXPECT_FLOAT_EQ(absmin, -3.f);

    float absmax = vec2.absmax();
    EXPECT_FLOAT_EQ(absmax, -7.f);

    size_t daxis = vec.dominantAxis();
    EXPECT_EQ(daxis, (size_t)1);

    size_t saxis = vec.subminantAxis();
    EXPECT_EQ(saxis, (size_t)0);

    float eps = 1e-6f;
    vec2 = vec.normalized();
    float lenSqr = vec2.x * vec2.x + vec2.y * vec2.y;
    EXPECT_NEAR(lenSqr, 1.f, eps);

    vec2.imul(2.f);
    float len = vec2.length();
    EXPECT_NEAR(len, 2.f, eps);

    lenSqr = vec2.lengthSquared();
    EXPECT_NEAR(lenSqr, 4.f, eps);
}

TEST(Vector2, BracketOperator) {
    Vector2F vec(8.f, 9.f);
    EXPECT_FLOAT_EQ(vec[0], 8.f);
    EXPECT_FLOAT_EQ(vec[1], 9.f);

    vec[0] = 7.f;
    vec[1] = 6.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
}

TEST(Vector2, AssignmentOperator) {
    Vector2F vec(5.f, 1.f);
    Vector2F vec2(3.f, 3.f);
    vec2 = vec;
    EXPECT_FLOAT_EQ(5.f, vec2.x);
    EXPECT_FLOAT_EQ(vec2.y, 1.f);
}

TEST(Vector2, AugmentedOperators) {
    Vector2F vec(3.f, 9.f);
    vec += 4.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, 13.f);

    vec += Vector2F(-2.f, 1.f);
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, 14.f);

    vec -= 8.f;
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec -= Vector2F(-5.f, 3.f);
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);

    vec *= 2.f;
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec *= Vector2F(3.f, -2.f);
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);

    vec /= 4.f;
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);

    vec /= Vector2F(3.f, -1.f);
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
}

TEST(Vector2, EqualOperator) {
    Vector2F vec, vec2(3.f, 7.f), vec3(3.f, 5.f), vec4(5.f, 1.f);
    vec = vec2;
    EXPECT_TRUE(vec == vec2);
    EXPECT_FALSE(vec == vec3);
    EXPECT_FALSE(vec != vec2);
    EXPECT_TRUE(vec != vec3);
    EXPECT_TRUE(vec != vec4);
}

TEST(Vector2, MinMaxFunction) {
    Vector2F vec(5.f, 1.f);
    Vector2F vec2(3.f, 3.f);
    Vector2F minVector = min(vec, vec2);
    Vector2F maxVector = max(vec, vec2);
    EXPECT_EQ(Vector2F(3.f, 1.f), minVector);
    EXPECT_EQ(Vector2F(5.f, 3.f), maxVector);
}

TEST(Vector2, ClampFunction) {
    Vector2F vec(2.f, 4.f), low(3.f, -1.f), high(5.f, 2.f);
    Vector2F clampedVec = clamp(vec, low, high);
    EXPECT_EQ(Vector2F(3.f, 2.f), clampedVec);
}

TEST(Vector2, CeilFloorFunction) {
    Vector2F vec(2.2f, 4.7f);
    Vector2F ceilVec = ceil(vec);
    EXPECT_EQ(Vector2F(3.f, 5.f), ceilVec);

    Vector2F floorVec = floor(vec);
    EXPECT_EQ(Vector2F(2.f, 4.f), floorVec);
}

TEST(Vector2, BinaryOperators) {
    Vector2F vec(3.f, 9.f);
    vec = vec + 4.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, 13.f);

    vec = vec + Vector2F(-2.f, 1.f);
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(vec.y, 14.f);

    vec = vec - 8.f;
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec = vec - Vector2F(-5.f, 3.f);
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);

    vec = vec * 2.f;
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);

    vec = vec * Vector2F(3.f, -2.f);
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);

    vec = vec / 4.f;
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);

    vec = vec / Vector2F(3.f, -1.f);
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);

    Vector2D v = Vector2D(2.0, 1.0).normalized();
    Vector2D normal = Vector2D(1.0, 1.0).normalized();

    Vector2D reflected = v.reflected(normal);
    Vector2D reflectedAnswer = Vector2D(-1.0, -2.0).normalized();
    EXPECT_NEAR(reflected.distanceTo(reflectedAnswer), 0.0, 1e-9);

    Vector2D projected = v.projected(normal);
    EXPECT_NEAR(projected.dot(normal), 0.0, 1e-9);

    Vector2D tangential = normal.tangential();
    EXPECT_NEAR(tangential.dot(normal), 0.0, 1e-9);
}
