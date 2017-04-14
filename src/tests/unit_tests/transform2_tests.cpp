// Copyright (c) 2016 Doyub Kim

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
}
