// Copyright (c) 2020 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/box2.h>

#include "unit_tests_utils.h"

using namespace jet;

TEST(Box2, Constructors) {
    {
        Box2 box;

        EXPECT_EQ(Vector2D(), box.bound.lowerCorner);
        EXPECT_EQ(Vector2D(1, 1), box.bound.upperCorner);
    }

    {
        Box2 box(Vector2D(-1, 2), Vector2D(5, 3));

        EXPECT_EQ(Vector2D(-1, 2), box.bound.lowerCorner);
        EXPECT_EQ(Vector2D(5, 3), box.bound.upperCorner);
    }

    {
        Box2 box(BoundingBox2D(Vector2D(-1, 2), Vector2D(5, 3)));

        box.isNormalFlipped = true;
        EXPECT_TRUE(box.isNormalFlipped);
        EXPECT_EQ(Vector2D(-1, 2), box.bound.lowerCorner);
        EXPECT_EQ(Vector2D(5, 3), box.bound.upperCorner);
    }
}

TEST(Box2, ClosestPoint) {
    Box2 box(Vector2D(-1, 2), Vector2D(5, 3));

    Vector2D result0 = box.closestPoint(Vector2D(-2, 4));
    EXPECT_EQ(Vector2D(-1, 3), result0);

    Vector2D result1 = box.closestPoint(Vector2D(1, 5));
    EXPECT_EQ(Vector2D(1, 3), result1);

    Vector2D result2 = box.closestPoint(Vector2D(9, 5));
    EXPECT_EQ(Vector2D(5, 3), result2);

    Vector2D result3 = box.closestPoint(Vector2D(-2, 2.4));
    EXPECT_EQ(Vector2D(-1, 2.4), result3);

    Vector2D result4 = box.closestPoint(Vector2D(1, 2.6));
    EXPECT_EQ(Vector2D(1, 3), result4);

    Vector2D result5 = box.closestPoint(Vector2D(9, 2.2));
    EXPECT_EQ(Vector2D(5, 2.2), result5);

    Vector2D result6 = box.closestPoint(Vector2D(-2, 1));
    EXPECT_EQ(Vector2D(-1, 2), result6);

    Vector2D result7 = box.closestPoint(Vector2D(1, 0));
    EXPECT_EQ(Vector2D(1, 2), result7);

    Vector2D result8 = box.closestPoint(Vector2D(9, -1));
    EXPECT_EQ(Vector2D(5, 2), result8);
}

TEST(Box2, ClosestDistance) {
    Box2 box(Vector2D(-1, 2), Vector2D(5, 3));

    double result0 = box.closestDistance(Vector2D(-2, 4));
    EXPECT_DOUBLE_EQ(Vector2D(-2, 4).distanceTo(Vector2D(-1, 3)), result0);

    double result1 = box.closestDistance(Vector2D(1, 5));
    EXPECT_DOUBLE_EQ(Vector2D(1, 5).distanceTo(Vector2D(1, 3)), result1);

    double result2 = box.closestDistance(Vector2D(9, 5));
    EXPECT_DOUBLE_EQ(Vector2D(9, 5).distanceTo(Vector2D(5, 3)), result2);

    double result3 = box.closestDistance(Vector2D(-2, 2.4));
    EXPECT_DOUBLE_EQ(Vector2D(-2, 2.4).distanceTo(Vector2D(-1, 2.4)), result3);

    double result4 = box.closestDistance(Vector2D(1, 2.6));
    EXPECT_DOUBLE_EQ(Vector2D(1, 2.6).distanceTo(Vector2D(1, 3)), result4);

    double result5 = box.closestDistance(Vector2D(9, 2.2));
    EXPECT_DOUBLE_EQ(Vector2D(9, 2.2).distanceTo(Vector2D(5, 2.2)), result5);

    double result6 = box.closestDistance(Vector2D(-2, 1));
    EXPECT_DOUBLE_EQ(Vector2D(-2, 1).distanceTo(Vector2D(-1, 2)), result6);

    double result7 = box.closestDistance(Vector2D(1, 0));
    EXPECT_DOUBLE_EQ(Vector2D(1, 0).distanceTo(Vector2D(1, 2)), result7);

    double result8 = box.closestDistance(Vector2D(9, -1));
    EXPECT_DOUBLE_EQ(Vector2D(9, -1).distanceTo(Vector2D(5, 2)), result8);
}

TEST(Box2, Intersects) {
    Box2 box(Vector2D(-1, 2), Vector2D(5, 3));

    bool result0 =
        box.intersects(Ray2D(Vector2D(1, 4), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result0);

    bool result1 =
        box.intersects(Ray2D(Vector2D(1, 2.5), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result1);

    bool result2 =
        box.intersects(Ray2D(Vector2D(1, 1), Vector2D(-1, -1).normalized()));
    EXPECT_FALSE(result2);
}

TEST(Box2, ClosestIntersection) {
    Box2 box(Vector2D(-1, 2), Vector2D(5, 3),
             Transform2(Vector2D(1.0, -3.0), 0));

    SurfaceRayIntersection2 result0 = box.closestIntersection(
        Ray2D(Vector2D(2, 1), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result0.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(2), result0.distance);
    EXPECT_VECTOR2_EQ(Vector2D(1, 0), result0.point);
    EXPECT_VECTOR2_EQ(Vector2D(0, 1), result0.normal);

    SurfaceRayIntersection2 result1 = box.closestIntersection(
        Ray2D(Vector2D(2, -0.5), Vector2D(-1, -1).normalized()));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(0.5), result1.distance);
    EXPECT_VECTOR2_EQ(Vector2D(1.5, -1), result1.point);
    EXPECT_VECTOR2_EQ(Vector2D(0, -1), result1.normal);

    SurfaceRayIntersection2 result2 = box.closestIntersection(
        Ray2D(Vector2D(2, -2), Vector2D(-1, -1).normalized()));
    EXPECT_FALSE(result2.isIntersecting);
}

TEST(Box2, BoundingBox) {
    Box2 box(Vector2D(-1, 2), Vector2D(5, 3));
    BoundingBox2D boundingBox = box.boundingBox();

    EXPECT_EQ(Vector2D(-1, 2), boundingBox.lowerCorner);
    EXPECT_EQ(Vector2D(5, 3), boundingBox.upperCorner);
}

TEST(Box2, ClosestNormal) {
    Box2 box(Vector2D(-1, 2), Vector2D(5, 3));
    box.isNormalFlipped = true;

    Vector2D result0 = box.closestNormal(Vector2D(-2, 2));
    EXPECT_EQ(Vector2D(1, -0), result0);

    Vector2D result1 = box.closestNormal(Vector2D(3, 5));
    EXPECT_EQ(Vector2D(0, -1), result1);

    Vector2D result2 = box.closestNormal(Vector2D(9, 3));
    EXPECT_EQ(Vector2D(-1, 0), result2);

    Vector2D result3 = box.closestNormal(Vector2D(4, 1));
    EXPECT_EQ(Vector2D(0, 1), result3);
}

TEST(Box2, Builder) {
    Box2 box = Box2::builder()
                   .withLowerCorner({-3.0, -2.0})
                   .withUpperCorner({5.0, 4.0})
                   .build();

    EXPECT_EQ(Vector2D(-3.0, -2.0), box.bound.lowerCorner);
    EXPECT_EQ(Vector2D(5.0, 4.0), box.bound.upperCorner);
}
