// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/transform2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(Transform2, Constructors) {
    Transform2 t1;

    EXPECT_EQ(Vector2D(), t1.translation());
    EXPECT_EQ(0.0, t1.orientation());

    Transform2 t2({2.0, -5.0}, kQuarterPiD);

    EXPECT_EQ(Vector2D(2.0, -5.0), t2.translation());
    EXPECT_EQ(kQuarterPiD, t2.orientation());
}

TEST(Transform2, Transform) {
    Transform2 t({2.0, -5.0}, kHalfPiD);

    auto r1 = t.toWorld({4.0, 1.0});
    EXPECT_DOUBLE_EQ(1.0, r1.x);
    EXPECT_DOUBLE_EQ(-1.0, r1.y);

    auto r2 = t.toLocal(r1);
    EXPECT_DOUBLE_EQ(4.0, r2.x);
    EXPECT_DOUBLE_EQ(1.0, r2.y);

    auto r3 = t.toWorldDirection({4.0, 1.0});
    EXPECT_DOUBLE_EQ(-1.0, r3.x);
    EXPECT_DOUBLE_EQ(4.0, r3.y);

    auto r4 = t.toLocalDirection(r3);
    EXPECT_DOUBLE_EQ(4.0, r4.x);
    EXPECT_DOUBLE_EQ(1.0, r4.y);

    BoundingBox2D bbox({-2, -1}, {2, 1});
    auto r5 = t.toWorld(bbox);
    EXPECT_BOUNDING_BOX2_EQ(BoundingBox2D({1, -7}, {3, -3}), r5);

    auto r6 = t.toLocal(r5);
    EXPECT_BOUNDING_BOX2_EQ(bbox, r6);
}
