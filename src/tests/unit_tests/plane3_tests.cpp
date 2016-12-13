// Copyright (c) 2016 Doyub Kim

#include <jet/plane3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Plane3, Builder) {
    Plane3 plane = Plane3::builder()
        .withNormal({1, 0, 0})
        .withPoint({2, 3, 4})
        .build();

    EXPECT_EQ(Vector3D(1, 0, 0), plane.normal);
    EXPECT_EQ(Vector3D(2, 3, 4), plane.point);
}
