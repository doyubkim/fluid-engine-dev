// Copyright (c) 2016 Doyub Kim

#include <jet/cylinder3.h>
#include <gtest/gtest.h>
#include <limits>

using namespace jet;

TEST(Cylinder3, Constructors) {
    Cylinder3 cyl1;
    EXPECT_FALSE(cyl1.isNormalFlipped);
    EXPECT_DOUBLE_EQ(0.0, cyl1.center.x);
    EXPECT_DOUBLE_EQ(0.0, cyl1.center.y);
    EXPECT_DOUBLE_EQ(0.0, cyl1.center.z);
    EXPECT_DOUBLE_EQ(1.0, cyl1.radius);
    EXPECT_DOUBLE_EQ(1.0, cyl1.height);
    EXPECT_DOUBLE_EQ(-1.0, cyl1.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl1.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl1.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(1.0, cyl1.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(0.5, cyl1.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(1.0, cyl1.boundingBox().upperCorner.z);

    Cylinder3 cyl2(Vector3D(1, 2, 3), 4.0, 5.0);
    EXPECT_FALSE(cyl2.isNormalFlipped);
    EXPECT_DOUBLE_EQ(1.0, cyl2.center.x);
    EXPECT_DOUBLE_EQ(2.0, cyl2.center.y);
    EXPECT_DOUBLE_EQ(3.0, cyl2.center.z);
    EXPECT_DOUBLE_EQ(4.0, cyl2.radius);
    EXPECT_DOUBLE_EQ(5.0, cyl2.height);
    EXPECT_DOUBLE_EQ(-3.0, cyl2.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl2.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl2.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, cyl2.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(4.5, cyl2.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(7.0, cyl2.boundingBox().upperCorner.z);

    cyl2.isNormalFlipped = true;
    Cylinder3 cyl3(cyl2);
    EXPECT_TRUE(cyl3.isNormalFlipped);
    EXPECT_DOUBLE_EQ(1.0, cyl3.center.x);
    EXPECT_DOUBLE_EQ(2.0, cyl3.center.y);
    EXPECT_DOUBLE_EQ(3.0, cyl3.center.z);
    EXPECT_DOUBLE_EQ(4.0, cyl3.radius);
    EXPECT_DOUBLE_EQ(5.0, cyl3.height);
    EXPECT_DOUBLE_EQ(-3.0, cyl3.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl3.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl3.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, cyl3.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(4.5, cyl3.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(7.0, cyl3.boundingBox().upperCorner.z);
}

TEST(Cylinder3, ClosestPoint) {
    Cylinder3 cyl(Vector3D(1, 2, 3), 4.0, 6.0);

    Vector3D result1 = cyl.closestPoint({7, 2, 3});
    EXPECT_DOUBLE_EQ(5.0, result1.x);
    EXPECT_DOUBLE_EQ(2.0, result1.y);
    EXPECT_DOUBLE_EQ(3.0, result1.z);

    Vector3D result2 = cyl.closestPoint({1, 6, 2});
    EXPECT_DOUBLE_EQ(1.0, result2.x);
    EXPECT_DOUBLE_EQ(5.0, result2.y);
    EXPECT_DOUBLE_EQ(2.0, result2.z);

    Vector3D result3 = cyl.closestPoint({6, -5, 3});
    EXPECT_DOUBLE_EQ(5.0, result3.x);
    EXPECT_DOUBLE_EQ(-1.0, result3.y);
    EXPECT_DOUBLE_EQ(3.0, result3.z);
}

TEST(Cylinder3, ClosestDistance) {
    Cylinder3 cyl(Vector3D(1, 2, 3), 4.0, 6.0);

    double result1 = cyl.closestDistance({7, 2, 3});
    EXPECT_DOUBLE_EQ(Vector3D(5, 2, 3).distanceTo({7, 2, 3}), result1);

    double result2 = cyl.closestDistance({1, 6, 2});
    EXPECT_DOUBLE_EQ(Vector3D(1, 5, 2).distanceTo({1, 6, 2}), result2);

    double result3 = cyl.closestDistance({6, -5, 3});
    EXPECT_DOUBLE_EQ(Vector3D(5, -1, 3).distanceTo({6, -5, 3}), result3);
}

TEST(Cylinder3, Intersects) {
    Cylinder3 cyl(Vector3D(1, 2, 3), 4.0, 6.0);

    EXPECT_TRUE(cyl.intersects(Ray3D({7, 2, 3}, {-1, 0, 0})));

    EXPECT_TRUE(cyl.intersects(Ray3D({1, 6, 2}, {0, -1, 0})));

    EXPECT_TRUE(cyl.intersects(Ray3D({1, 2, 3}, {0, 1, 0})));

    EXPECT_FALSE(cyl.intersects(Ray3D({6, -5, 3}, {0, 0, 1})));
}

TEST(Cylinder3, closestIntersection) {
    Cylinder3 cyl(Vector3D(1, 2, 3), 4.0, 6.0);

    auto result1 = cyl.closestIntersection(Ray3D({7, 2, 3}, {-1, 0, 0}));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(2.0, result1.t);

    auto result2 = cyl.closestIntersection(Ray3D({1, 6, 2}, {0, -1, 0}));
    EXPECT_TRUE(result2.isIntersecting);
    EXPECT_DOUBLE_EQ(1.0, result2.t);

    auto result3 = cyl.closestIntersection(Ray3D({1, 2, 3}, {0, 1, 0}));
    EXPECT_TRUE(result3.isIntersecting);
    EXPECT_DOUBLE_EQ(3.0, result3.t);

    auto result4 = cyl.closestIntersection(Ray3D({6, -5, 3}, {0, 0, 1}));
    EXPECT_FALSE(result4.isIntersecting);
}

TEST(Cylinder3, BoundingBox) {
    Cylinder3 cyl(Vector3D(1, 2, 3), 4.0, 6.0);
    BoundingBox3D bbox = cyl.boundingBox();
    EXPECT_DOUBLE_EQ(-3.0, bbox.lowerCorner.x);
    EXPECT_DOUBLE_EQ(-1.0, bbox.lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, bbox.lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, bbox.upperCorner.x);
    EXPECT_DOUBLE_EQ(5.0, bbox.upperCorner.y);
    EXPECT_DOUBLE_EQ(7.0, bbox.upperCorner.z);
}

TEST(Cylinder3, ClosestNormal) {
    Cylinder3 cyl(Vector3D(1, 2, 3), 4.0, 6.0);
    cyl.isNormalFlipped = true;

    Vector3D result1 = cyl.closestNormal({7, 2, 3});
    EXPECT_DOUBLE_EQ(-1.0, result1.x);
    EXPECT_DOUBLE_EQ(0.0, result1.y);
    EXPECT_DOUBLE_EQ(0.0, result1.z);

    Vector3D result2 = cyl.closestNormal({1, 6, 2});
    EXPECT_DOUBLE_EQ(0.0, result2.x);
    EXPECT_DOUBLE_EQ(-1.0, result2.y);
    EXPECT_DOUBLE_EQ(0.0, result2.z);

    Vector3D result3 = cyl.closestNormal({6, -1.5, 3});
    EXPECT_DOUBLE_EQ(-1.0, result3.x);
    EXPECT_DOUBLE_EQ(0.0, result3.y);
    EXPECT_DOUBLE_EQ(0.0, result3.z);

    Vector3D result4 = cyl.closestNormal({3, 0, 3});
    EXPECT_DOUBLE_EQ(0.0, result4.x);
    EXPECT_DOUBLE_EQ(1.0, result4.y);
    EXPECT_DOUBLE_EQ(0.0, result4.z);
}

TEST(Cylinder3, Builder) {
    Cylinder3 cyl2 = Cylinder3::builder()
        .withCenter({1, 2, 3})
        .withRadius(4.0)
        .withHeight(5.0)
        .build();

    EXPECT_FALSE(cyl2.isNormalFlipped);
    EXPECT_DOUBLE_EQ(1.0, cyl2.center.x);
    EXPECT_DOUBLE_EQ(2.0, cyl2.center.y);
    EXPECT_DOUBLE_EQ(3.0, cyl2.center.z);
    EXPECT_DOUBLE_EQ(4.0, cyl2.radius);
    EXPECT_DOUBLE_EQ(5.0, cyl2.height);
    EXPECT_DOUBLE_EQ(-3.0, cyl2.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(-0.5, cyl2.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(-1.0, cyl2.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(5.0, cyl2.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(4.5, cyl2.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(7.0, cyl2.boundingBox().upperCorner.z);
}
