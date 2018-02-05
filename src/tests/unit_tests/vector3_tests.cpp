// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/vector3.h>
#include <gtest/gtest.h>
#include <algorithm>

using namespace jet;

TEST(Vector3, Constructors) {
    Vector3F vec;
    EXPECT_FLOAT_EQ(0.f, vec.x);
    EXPECT_FLOAT_EQ(0.f, vec.y);
    EXPECT_FLOAT_EQ(0.f, vec.z);

    Vector3F vec2(5.f, 3.f, 8.f);
    EXPECT_FLOAT_EQ(5.f, vec2.x);
    EXPECT_FLOAT_EQ(3.f, vec2.y);
    EXPECT_FLOAT_EQ(8.f, vec2.z);

    Vector2F vec3(4.f, 7.f);
    Vector3F vec4(vec3, 9.f);
    EXPECT_FLOAT_EQ(4.f, vec4.x);
    EXPECT_FLOAT_EQ(7.f, vec4.y);
    EXPECT_FLOAT_EQ(9.f, vec4.z);

    Vector3F vec5 = { 7.f, 6.f, 1.f };
    EXPECT_FLOAT_EQ(7.f, vec5.x);
    EXPECT_FLOAT_EQ(6.f, vec5.y);
    EXPECT_FLOAT_EQ(1.f, vec5.z);

    Vector3F vec6(vec5);
    EXPECT_FLOAT_EQ(7.f, vec6.x);
    EXPECT_FLOAT_EQ(6.f, vec6.y);
    EXPECT_FLOAT_EQ(1.f, vec6.z);
}

TEST(Vector3, SetMethods) {
    Vector3F vec;
    vec.set(4.f, 2.f, 8.f);
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(2.f, vec.y);
    EXPECT_FLOAT_EQ(8.f, vec.z);

    vec.set(Vector2F(1.f, 3.f), 10.f);
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(10.f, vec.z);

    auto lst = {0.f, 5.f, 6.f};
    vec.set(lst);
    EXPECT_FLOAT_EQ(0.f, vec.x);
    EXPECT_FLOAT_EQ(5.f, vec.y);
    EXPECT_FLOAT_EQ(6.f, vec.z);

    vec.set(Vector3F(9.f, 8.f, 2.f));
    EXPECT_FLOAT_EQ(9.f, vec.x);
    EXPECT_FLOAT_EQ(8.f, vec.y);
    EXPECT_FLOAT_EQ(2.f, vec.z);
}

TEST(Vector3, BasicSetterMethods) {
    Vector3F vec(3.f, 9.f, 4.f);
    vec.setZero();
    EXPECT_FLOAT_EQ(0.f, vec.x);
    EXPECT_FLOAT_EQ(0.f, vec.y);
    EXPECT_FLOAT_EQ(0.f, vec.z);

    vec.set(4.f, 2.f, 8.f);
    vec.normalize();
    float len = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    EXPECT_TRUE(fabsf(len - 1.f) < 1e-6);
}

TEST(Vector3, BinaryOperatorMethods) {
    Vector3F vec(3.f, 9.f, 4.f);
    vec = vec.add(4.f);
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(13.f, vec.y);
    EXPECT_FLOAT_EQ(8.f, vec.z);

    vec = vec.add(Vector3F(-2.f, 1.f, 5.f));
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(14.f, vec.y);
    EXPECT_FLOAT_EQ(13.f, vec.z);

    vec = vec.sub(8.f);
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(5.f, vec.z);

    vec = vec.sub(Vector3F(-5.f, 3.f, 12.f));
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    vec = vec.mul(2.f);
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(-14.f, vec.z);

    vec = vec.mul(Vector3F(3.f, -2.f, 0.5f));
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    vec = vec.div(4.f);
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);
    EXPECT_FLOAT_EQ(-1.75f, vec.z);

    vec = vec.div(Vector3F(3.f, -1.f, 0.25f));
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    float d = vec.dot(Vector3F(4.f, 2.f, 1.f));
    EXPECT_FLOAT_EQ(3.f, d);

    Vector3F c = vec.cross(Vector3F(5.f, -7.f, 2.f));
    EXPECT_FLOAT_EQ(-43.f, c.x);
    EXPECT_FLOAT_EQ(-37.f, c.y);
    EXPECT_FLOAT_EQ(-22.f, c.z);
}

TEST(Vector3, BinaryInverseOperatorMethods) {
    Vector3F vec(5.f, 14.f, 13.f);
    vec = vec.rsub(8.f);
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-6.f, vec.y);
    EXPECT_FLOAT_EQ(-5.f, vec.z);

    vec = vec.rsub(Vector3F(-5.f, 3.f, -1.f));
    EXPECT_FLOAT_EQ(-8.f, vec.x);
    EXPECT_FLOAT_EQ(9.f, vec.y);
    EXPECT_FLOAT_EQ(4.f, vec.z);

    vec = Vector3F(-12.f, -9.f, 8.f);
    vec = vec.rdiv(36.f);
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(-4.f, vec.y);
    EXPECT_FLOAT_EQ(4.5f, vec.z);

    vec = vec.rdiv(Vector3F(3.f, -16.f, 18.f));
    EXPECT_FLOAT_EQ(-1.f, vec.x);
    EXPECT_FLOAT_EQ(4.f, vec.y);
    EXPECT_FLOAT_EQ(4.f, vec.z);

    Vector3F c = vec.rcross(Vector3F(5.f, -7.f, 3.f));
    EXPECT_FLOAT_EQ(-40.f, c.x);
    EXPECT_FLOAT_EQ(-23.f, c.y);
    EXPECT_FLOAT_EQ(13.f, c.z);
}

TEST(Vector3, AugmentedOperatorMethods) {
    Vector3F vec(3.f, 9.f, 4.f);
    vec.iadd(4.f);
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(13.f, vec.y);
    EXPECT_FLOAT_EQ(8.f, vec.z);

    vec.iadd(Vector3F(-2.f, 1.f, 5.f));
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(14.f, vec.y);
    EXPECT_FLOAT_EQ(13.f, vec.z);

    vec.isub(8.f);
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(5.f, vec.z);

    vec.isub(Vector3F(-5.f, 3.f, 12.f));
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    vec.imul(2.f);
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(-14.f, vec.z);

    vec.imul(Vector3F(3.f, -2.f, 0.5f));
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    vec.idiv(4.f);
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);
    EXPECT_FLOAT_EQ(-1.75f, vec.z);

    vec.idiv(Vector3F(3.f, -1.f, 0.25f));
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);
}

TEST(Vector3, AtMethods) {
    Vector3F vec(8.f, 9.f, 1.f);
    EXPECT_FLOAT_EQ(8.f, vec.at(0));
    EXPECT_FLOAT_EQ(9.f, vec.at(1));
    EXPECT_FLOAT_EQ(1.f, vec.at(2));

    vec.at(0) = 7.f;
    vec.at(1) = 6.f;
    vec.at(2) = 5.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(5.f, vec.z);
}

TEST(Vector3, BasicGetterMethods) {
    Vector3F vec(3.f, 7.f, -1.f), vec2(-3.f, -7.f, 1.f);

    float sum = vec.sum();
    EXPECT_FLOAT_EQ(9.f, sum);

    float avg = vec.avg();
    EXPECT_FLOAT_EQ(3.f, avg);

    float min = vec.min();
    EXPECT_FLOAT_EQ(-1.f, min);

    float max = vec.max();
    EXPECT_FLOAT_EQ(7.f, max);

    float absmin = vec2.absmin();
    EXPECT_FLOAT_EQ(1.f, absmin);

    float absmax = vec2.absmax();
    EXPECT_FLOAT_EQ(-7.f, absmax);

    size_t daxis = vec.dominantAxis();
    EXPECT_EQ((size_t)1, daxis);

    size_t saxis = vec.subminantAxis();
    EXPECT_EQ((size_t)2, saxis);

    float eps = 1e-6f;
    vec2 = vec.normalized();
    float lenSqr = vec2.x * vec2.x + vec2.y * vec2.y + vec2.z * vec2.z;
    EXPECT_TRUE(lenSqr - 1.f < eps);

    vec2.imul(2.f);
    float len = vec2.length();
    EXPECT_TRUE(len - 2.f < eps);

    lenSqr = vec2.lengthSquared();
    EXPECT_TRUE(lenSqr - 4.f < eps);
}

TEST(Vector3, BracketOperators) {
    Vector3F vec(8.f, 9.f, 1.f);
    EXPECT_FLOAT_EQ(8.f, vec[0]);
    EXPECT_FLOAT_EQ(9.f, vec[1]);
    EXPECT_FLOAT_EQ(1.f, vec[2]);

    vec[0] = 7.f;
    vec[1] = 6.f;
    vec[2] = 5.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(5.f, vec.z);
}

TEST(Vector3, AssignmentOperators) {
    Vector3F vec(5.f, 1.f, 0.f);
    Vector3F vec2(3.f, 3.f, 3.f);
    vec2 = vec;
    EXPECT_FLOAT_EQ(5.f, vec2.x);
    EXPECT_FLOAT_EQ(1.f, vec2.y);
    EXPECT_FLOAT_EQ(0.f, vec2.z);
}

TEST(Vector3, AugmentedOperators) {
    Vector3F vec(3.f, 9.f, -2.f);
    vec += 4.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(13.f, vec.y);
    EXPECT_FLOAT_EQ(2.f, vec.z);

    vec += Vector3F(-2.f, 1.f, 5.f);
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(14.f, vec.y);
    EXPECT_FLOAT_EQ(7.f, vec.z);

    vec -= 8.f;
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(-1.f, vec.z);

    vec -= Vector3F(-5.f, 3.f, -6.f);
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(5.f, vec.z);

    vec *= 2.f;
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(10.f, vec.z);

    vec *= Vector3F(3.f, -2.f, 0.4f);
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);
    EXPECT_FLOAT_EQ(4.f, vec.z);

    vec /= 4.f;
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);
    EXPECT_FLOAT_EQ(1.f, vec.z);

    vec /= Vector3F(3.f, -1.f, 2.f);
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(0.5f, vec.z);
}

TEST(Vector3, EqualOperatators) {
    Vector3F vec, vec2(3.f, 7.f, 4.f), vec3(3.f, 5.f, 4.f), vec4(5.f, 1.f, 2.f);
    vec = vec2;
    EXPECT_TRUE(vec == vec2);
    EXPECT_FALSE(vec == vec3);
    EXPECT_FALSE(vec != vec2);
    EXPECT_TRUE(vec != vec3);
    EXPECT_TRUE(vec != vec4);
}

TEST(Vector3, MinMaxFunctions) {
    Vector3F vec(5.f, 1.f, 0.f);
    Vector3F vec2(3.f, 3.f, 3.f);
    Vector3F minVector = min(vec, vec2);
    Vector3F maxVector = max(vec, vec2);
    EXPECT_TRUE(minVector == Vector3F(3.f, 1.f, 0.f));
    EXPECT_TRUE(maxVector == Vector3F(5.f, 3.f, 3.f));
}

TEST(Vector3, ClampFunction) {
    Vector3F vec(2.f, 4.f, 1.f), low(3.f, -1.f, 0.f), high(5.f, 2.f, 3.f);
    Vector3F clampedVec = clamp(vec, low, high);
    EXPECT_TRUE(clampedVec == Vector3F(3.f, 2.f, 1.f));
}

TEST(Vector3, CeilFloorFunctions) {
    Vector3F vec(2.2f, 4.7f, -0.2f);
    Vector3F ceilVec = ceil(vec);
    EXPECT_TRUE(ceilVec == Vector3F(3.f, 5.f, 0.f));

    Vector3F floorVec = floor(vec);
    EXPECT_TRUE(floorVec == Vector3F(2.f, 4.f, -1.f));
}

TEST(Vector3, BinaryOperators) {
    Vector3F vec(3.f, 9.f, 4.f);
    vec = vec + 4.f;
    EXPECT_FLOAT_EQ(7.f, vec.x);
    EXPECT_FLOAT_EQ(13.f, vec.y);
    EXPECT_FLOAT_EQ(8.f, vec.z);

    vec = vec + Vector3F(-2.f, 1.f, 5.f);
    EXPECT_FLOAT_EQ(5.f, vec.x);
    EXPECT_FLOAT_EQ(14.f, vec.y);
    EXPECT_FLOAT_EQ(13.f, vec.z);

    vec = vec - 8.f;
    EXPECT_FLOAT_EQ(-3.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(5.f, vec.z);

    vec = vec - Vector3F(-5.f, 3.f, 12.f);
    EXPECT_FLOAT_EQ(2.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    vec = vec * 2.f;
    EXPECT_FLOAT_EQ(4.f, vec.x);
    EXPECT_FLOAT_EQ(6.f, vec.y);
    EXPECT_FLOAT_EQ(-14.f, vec.z);

    vec = vec * Vector3F(3.f, -2.f, 0.5f);
    EXPECT_FLOAT_EQ(12.f, vec.x);
    EXPECT_FLOAT_EQ(-12.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    vec = vec / 4.f;
    EXPECT_FLOAT_EQ(3.f, vec.x);
    EXPECT_FLOAT_EQ(-3.f, vec.y);
    EXPECT_FLOAT_EQ(-1.75f, vec.z);

    vec = vec / Vector3F(3.f, -1.f, 0.25f);
    EXPECT_FLOAT_EQ(1.f, vec.x);
    EXPECT_FLOAT_EQ(3.f, vec.y);
    EXPECT_FLOAT_EQ(-7.f, vec.z);

    Vector3D v = Vector3D(2.0, 1.0, 3.0).normalized();
    Vector3D normal = Vector3D(1.0, 1.0, 1.0).normalized();

    Vector3D reflected = v.reflected(normal);
    Vector3D reflectedAnswer = Vector3D(-2.0, -3.0, -1.0).normalized();
    EXPECT_NEAR(reflected.distanceTo(reflectedAnswer), 0.0, 1e-9);

    Vector3D projected = v.projected(normal);
    EXPECT_NEAR(projected.dot(normal), 0.0, 1e-9);

    auto tangential = normal.tangential();
    EXPECT_NEAR(std::get<0>(tangential).dot(normal), 0.0, 1e-9);
    EXPECT_NEAR(std::get<1>(tangential).dot(normal), 0.0, 1e-9);
}
