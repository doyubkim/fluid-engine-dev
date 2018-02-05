// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/transform3.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(Transform3, Constructors) {
    Transform3 t1;

    EXPECT_EQ(Vector3D(), t1.translation());
    EXPECT_EQ(0.0, t1.orientation().angle());

    Transform3 t2({2.0, -5.0, 1.0}, QuaternionD({0.0, 1.0, 0.0}, kQuarterPiD));

    EXPECT_EQ(Vector3D(2.0, -5.0, 1.0), t2.translation());
    EXPECT_EQ(Vector3D(0.0, 1.0, 0.0), t2.orientation().axis());
    EXPECT_DOUBLE_EQ(kQuarterPiD, t2.orientation().angle());
}

TEST(Transform3, Transform) {
    Transform3 t({2.0, -5.0, 1.0}, QuaternionD({0.0, 1.0, 0.0}, kHalfPiD));

    auto r1 = t.toWorld({4.0, 1.0, -3.0});
    EXPECT_NEAR(-1.0, r1.x, 1e-9);
    EXPECT_NEAR(-4.0, r1.y, 1e-9);
    EXPECT_NEAR(-3.0, r1.z, 1e-9);

    auto r2 = t.toLocal(r1);
    EXPECT_NEAR(4.0, r2.x, 1e-9);
    EXPECT_NEAR(1.0, r2.y, 1e-9);
    EXPECT_NEAR(-3.0, r2.z, 1e-9);

    auto r3 = t.toWorldDirection({4.0, 1.0, -3.0});
    EXPECT_NEAR(-3.0, r3.x, 1e-9);
    EXPECT_NEAR(1.0, r3.y, 1e-9);
    EXPECT_NEAR(-4.0, r3.z, 1e-9);

    auto r4 = t.toLocalDirection(r3);
    EXPECT_NEAR(4.0, r4.x, 1e-9);
    EXPECT_NEAR(1.0, r4.y, 1e-9);
    EXPECT_NEAR(-3.0, r4.z, 1e-9);

    BoundingBox3D bbox({-2, -1, -3}, {2, 1, 3});
    auto r5 = t.toWorld(bbox);
    EXPECT_BOUNDING_BOX3_NEAR(BoundingBox3D({-1, -6, -1}, {5, -4, 3}), r5, 1e-9);

    auto r6 = t.toLocal(r5);
    EXPECT_BOUNDING_BOX3_NEAR(bbox, r6, 1e-9);
}
