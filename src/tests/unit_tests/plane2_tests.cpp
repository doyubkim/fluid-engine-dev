// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/plane2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Plane2, Builder) {
    Plane2 plane = Plane2::builder()
        .withNormal({1, 0})
        .withPoint({2, 3})
        .build();

    EXPECT_EQ(Vector2D(1, 0), plane.normal);
    EXPECT_EQ(Vector2D(2, 3), plane.point);
}
