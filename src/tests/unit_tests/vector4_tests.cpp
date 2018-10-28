// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/vector4.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(Vector4, BasicGetterMethods) {
    Vector4F vec(3.f, 7.f, -1.f, 11.f), vec2(-3.f, -7.f, 1.f, 4.f);
    Vector4F vec3(3.f, 1.f, -5.f, 4.f), vec4(-3.f, 2.f, 1.f, -4.f);

    float sum = vec.sum();
    EXPECT_FLOAT_EQ(20.f, sum);

    float avg = vec.avg();
    EXPECT_FLOAT_EQ(5.f, avg);

    float min = vec.min();
    EXPECT_FLOAT_EQ(-1.f, min);

    float max = vec.max();
    EXPECT_FLOAT_EQ(11.f, max);

    float absmin = vec2.absmin();
    EXPECT_FLOAT_EQ(1.f, absmin);

    float absmax = vec2.absmax();
    EXPECT_FLOAT_EQ(-7.f, absmax);

    size_t daxis = vec3.dominantAxis();
    EXPECT_EQ((size_t)2, daxis);

    size_t saxis = vec4.subminantAxis();
    EXPECT_EQ((size_t)2, saxis);

    float eps = 1e-6f;
    vec2 = vec.normalized();
    float lenSqr = vec2.x * vec2.x + vec2.y * vec2.y + vec2.z * vec2.z + vec2.w * vec2.w;
    EXPECT_TRUE(lenSqr - 1.f < eps);

    vec2.imul(2.f);
    float len = vec2.length();
    EXPECT_TRUE(len - 2.f < eps);

    lenSqr = vec2.lengthSquared();
    EXPECT_TRUE(lenSqr - 4.f < eps);
}