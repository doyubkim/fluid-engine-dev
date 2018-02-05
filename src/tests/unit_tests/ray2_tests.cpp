// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/ray2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Ray2, Constructors) {
    Ray2D ray;
    EXPECT_EQ(Vector2D(), ray.origin);
    EXPECT_EQ(Vector2D(1, 0), ray.direction);

    Ray2D ray2({1, 2}, {3, 4});
    EXPECT_EQ(Vector2D(1, 2), ray2.origin);
    EXPECT_EQ(Vector2D(3, 4).normalized(), ray2.direction);

    Ray2D ray3(ray2);
    EXPECT_EQ(Vector2D(1, 2), ray3.origin);
    EXPECT_EQ(Vector2D(3, 4).normalized(), ray3.direction);
}

TEST(Ray2, PointAt) {
    Ray2D ray;
    EXPECT_EQ(Vector2D(4.5, 0.0), ray.pointAt(4.5));
}
