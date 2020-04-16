// Copyright (c) 2020 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/box3.h>

#include "unit_tests_utils.h"

using namespace jet;

TEST(Box3, Constructors) {
    {
        Box3 box;

        EXPECT_FALSE(box.isNormalFlipped);
        EXPECT_EQ(Vector3D(), box.bound.lowerCorner);
        EXPECT_EQ(Vector3D(1, 1, 1), box.bound.upperCorner);
    }

    {
        Box3 box(Vector3D(-1, 2, 1), Vector3D(5, 3, 4));

        EXPECT_FALSE(box.isNormalFlipped);
        EXPECT_EQ(Vector3D(-1, 2, 1), box.bound.lowerCorner);
        EXPECT_EQ(Vector3D(5, 3, 4), box.bound.upperCorner);
    }

    {
        Box3 box(BoundingBox3D(Vector3D(-1, 2, 1), Vector3D(5, 3, 4)));

        box.isNormalFlipped = true;
        EXPECT_TRUE(box.isNormalFlipped);
        EXPECT_EQ(Vector3D(-1, 2, 1), box.bound.lowerCorner);
        EXPECT_EQ(Vector3D(5, 3, 4), box.bound.upperCorner);
    }
}

TEST(Box3, ClosestPoint) {
    Box3 box(Vector3D(-1, 2, 1), Vector3D(5, 3, 4));

    Vector3D result0 = box.closestPoint(Vector3D(-2, 4, 5));
    EXPECT_EQ(Vector3D(-1, 3, 4), result0);

    Vector3D result1 = box.closestPoint(Vector3D(1, 5, 0));
    EXPECT_EQ(Vector3D(1, 3, 1), result1);

    Vector3D result2 = box.closestPoint(Vector3D(9, 5, 7));
    EXPECT_EQ(Vector3D(5, 3, 4), result2);

    Vector3D result3 = box.closestPoint(Vector3D(-2, 2.4, 3));
    EXPECT_EQ(Vector3D(-1, 2.4, 3), result3);

    Vector3D result4 = box.closestPoint(Vector3D(1, 2.6, 1.1));
    EXPECT_EQ(Vector3D(1, 2.6, 1), result4);

    Vector3D result5 = box.closestPoint(Vector3D(9, 2.2, -1));
    EXPECT_EQ(Vector3D(5, 2.2, 1), result5);

    Vector3D result6 = box.closestPoint(Vector3D(-2, 1, 1.1));
    EXPECT_EQ(Vector3D(-1, 2, 1.1), result6);

    Vector3D result7 = box.closestPoint(Vector3D(1, 0, 3.5));
    EXPECT_EQ(Vector3D(1, 2, 3.5), result7);

    Vector3D result8 = box.closestPoint(Vector3D(9, -1, -3));
    EXPECT_EQ(Vector3D(5, 2, 1), result8);
}

TEST(Box3, ClosestDistance) {
    Box3 box(Vector3D(-1, 2, 1), Vector3D(5, 3, 4));

    double result0 = box.closestDistance(Vector3D(-2, 4, 5));
    EXPECT_DOUBLE_EQ(Vector3D(-1, 3, 4).distanceTo(Vector3D(-2, 4, 5)),
                     result0);

    double result1 = box.closestDistance(Vector3D(1, 5, 0));
    EXPECT_DOUBLE_EQ(Vector3D(1, 3, 1).distanceTo(Vector3D(1, 5, 0)), result1);

    double result2 = box.closestDistance(Vector3D(9, 5, 7));
    EXPECT_DOUBLE_EQ(Vector3D(5, 3, 4).distanceTo(Vector3D(9, 5, 7)), result2);

    double result3 = box.closestDistance(Vector3D(-2, 2.4, 3));
    EXPECT_DOUBLE_EQ(Vector3D(-1, 2.4, 3).distanceTo(Vector3D(-2, 2.4, 3)),
                     result3);

    double result4 = box.closestDistance(Vector3D(1, 2.6, 1.1));
    EXPECT_DOUBLE_EQ(Vector3D(1, 2.6, 1).distanceTo(Vector3D(1, 2.6, 1.1)),
                     result4);

    double result5 = box.closestDistance(Vector3D(9, 2.2, -1));
    EXPECT_DOUBLE_EQ(Vector3D(5, 2.2, 1).distanceTo(Vector3D(9, 2.2, -1)),
                     result5);

    double result6 = box.closestDistance(Vector3D(-2, 1, 1.1));
    EXPECT_DOUBLE_EQ(Vector3D(-1, 2, 1.1).distanceTo(Vector3D(-2, 1, 1.1)),
                     result6);

    double result7 = box.closestDistance(Vector3D(1, 0, 3.5));
    EXPECT_DOUBLE_EQ(Vector3D(1, 2, 3.5).distanceTo(Vector3D(1, 0, 3.5)),
                     result7);

    double result8 = box.closestDistance(Vector3D(9, -1, -3));
    EXPECT_DOUBLE_EQ(Vector3D(5, 2, 1).distanceTo(Vector3D(9, -1, -3)),
                     result8);
}

TEST(Box3, Intersects) {
    Box3 box(Vector3D(-1, 2, 3), Vector3D(5, 3, 7));

    bool result0 = box.intersects(
        Ray3D(Vector3D(1, 4, 5), Vector3D(-1, -1, -1).normalized()));
    EXPECT_TRUE(result0);

    bool result1 = box.intersects(
        Ray3D(Vector3D(1, 2.5, 6), Vector3D(-1, -1, 1).normalized()));
    EXPECT_TRUE(result1);

    bool result2 = box.intersects(
        Ray3D(Vector3D(1, 1, 2), Vector3D(-1, -1, -1).normalized()));
    EXPECT_FALSE(result2);
}

TEST(Box3, ClosestIntersection) {
    Box3 box(Vector3D(-1, 2, 3), Vector3D(5, 3, 7),
             Transform3(Vector3D(1, -3, 2), QuaternionD()));

    SurfaceRayIntersection3 result0 = box.closestIntersection(
        Ray3D(Vector3D(2, 1, 7), Vector3D(-1, -1, -1).normalized()));
    EXPECT_TRUE(result0.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(3), result0.distance);
    EXPECT_VECTOR3_EQ(Vector3D(1, 0, 6), result0.point);
    EXPECT_VECTOR3_EQ(Vector3D(0, 1, 0), result0.normal);

    SurfaceRayIntersection3 result1 = box.closestIntersection(
        Ray3D(Vector3D(2, -0.5, 8), Vector3D(-1, -1, 1).normalized()));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(std::sqrt(0.75), result1.distance);
    EXPECT_VECTOR3_EQ(Vector3D(1.5, -1, 8.5), result1.point);
    EXPECT_VECTOR3_EQ(Vector3D(0, -1, 0), result1.normal);

    SurfaceRayIntersection3 result2 = box.closestIntersection(
        Ray3D(Vector3D(2, -2, 4), Vector3D(-1, -1, -1).normalized()));
    EXPECT_FALSE(result2.isIntersecting);
}

TEST(Box3, BoundingBox) {
    Box3 box(Vector3D(-1, 2, 3), Vector3D(5, 3, 7));
    BoundingBox3D boundingBox = box.boundingBox();

    EXPECT_EQ(Vector3D(-1, 2, 3), boundingBox.lowerCorner);
    EXPECT_EQ(Vector3D(5, 3, 7), boundingBox.upperCorner);
}

TEST(Box3, ClosestNormal) {
    Box3 box(Vector3D(-1, 2, 1), Vector3D(5, 3, 4));
    box.isNormalFlipped = true;

    Vector3D result0 = box.closestNormal(Vector3D(-2, 2, 3));
    EXPECT_EQ(Vector3D(1, 0, 0), result0);

    Vector3D result1 = box.closestNormal(Vector3D(3, 5, 2));
    EXPECT_EQ(Vector3D(0, -1, 0), result1);

    Vector3D result2 = box.closestNormal(Vector3D(9, 3, 4));
    EXPECT_EQ(Vector3D(-1, 0, 0), result2);

    Vector3D result3 = box.closestNormal(Vector3D(4, 1, 1));
    EXPECT_EQ(Vector3D(0, 1, 0), result3);

    Vector3D result4 = box.closestNormal(Vector3D(4, 2.5, -1));
    EXPECT_EQ(Vector3D(0, 0, 1), result4);

    Vector3D result5 = box.closestNormal(Vector3D(4, 2, 9));
    EXPECT_EQ(Vector3D(0, 0, -1), result5);
}
