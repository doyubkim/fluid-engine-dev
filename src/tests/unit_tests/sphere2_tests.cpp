// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/sphere2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(Sphere2, Constructors) {
    Sphere2 sph1;
    EXPECT_DOUBLE_EQ(0.0, sph1.center.x);
    EXPECT_DOUBLE_EQ(0.0, sph1.center.y);
    EXPECT_DOUBLE_EQ(1.0, sph1.radius);

    Sphere2 sph2({3.0, -1.0}, 5.0);
    EXPECT_DOUBLE_EQ(3.0, sph2.center.x);
    EXPECT_DOUBLE_EQ(-1.0, sph2.center.y);
    EXPECT_DOUBLE_EQ(5.0, sph2.radius);

    sph2.isNormalFlipped = true;

    Sphere2 sph3(sph2);
    EXPECT_DOUBLE_EQ(3.0, sph3.center.x);
    EXPECT_DOUBLE_EQ(-1.0, sph3.center.y);
    EXPECT_DOUBLE_EQ(5.0, sph3.radius);
    EXPECT_TRUE(sph3.isNormalFlipped);
}

TEST(Sphere2, ClosestPoint) {
    Sphere2 sph({3.0, -1.0}, 5.0);

    auto result1 = sph.closestPoint({10.0, -1.0});
    EXPECT_DOUBLE_EQ(8.0, result1.x);
    EXPECT_DOUBLE_EQ(-1.0, result1.y);

    auto result2 = sph.closestPoint({3.0, -10.0});
    EXPECT_DOUBLE_EQ(3.0, result2.x);
    EXPECT_DOUBLE_EQ(-6.0, result2.y);

    auto result3 = sph.closestPoint({3.0, 3.0});
    EXPECT_DOUBLE_EQ(3.0, result3.x);
    EXPECT_DOUBLE_EQ(4.0, result3.y);
}

TEST(Sphere2, ClosestDistance) {
    Sphere2 sph({3.0, -1.0}, 5.0);

    double result1 = sph.closestDistance({10.0, -1.0});
    EXPECT_DOUBLE_EQ(2.0, result1);

    double result2 = sph.closestDistance({3.0, -10.0});
    EXPECT_DOUBLE_EQ(4.0, result2);

    double result3 = sph.closestDistance({3.0, 3.0});
    EXPECT_DOUBLE_EQ(1.0, result3);
}

TEST(Sphere2, Intersects) {
    Sphere2 sph({3.0, -1.0}, 5.0);
    sph.isNormalFlipped = true;

    bool result1 = sph.intersects(Ray2D({10.0, -1.0}, {-1.0, 0.0}));
    EXPECT_TRUE(result1);

    bool result2 = sph.intersects(Ray2D({3.0, -10.0}, {0.0, -1.0}));
    EXPECT_FALSE(result2);

    bool result3 = sph.intersects(Ray2D({3.0, 3.0}, {1.0, 0.0}));
    EXPECT_TRUE(result3);
}

TEST(Sphere2, ClosestIntersection) {
    Sphere2 sph({3.0, -1.0}, 5.0);
    sph.isNormalFlipped = true;

    auto result1 = sph.closestIntersection(Ray2D({10.0, -1.0}, {-1.0, 0.0}));
    EXPECT_TRUE(result1.isIntersecting);
    EXPECT_DOUBLE_EQ(2.0, result1.distance);
    EXPECT_DOUBLE_EQ(8.0, result1.point.x);
    EXPECT_DOUBLE_EQ(-1.0, result1.point.y);

    auto result2 = sph.closestIntersection(Ray2D({3.0, -10.0}, {0.0, -1.0}));
    EXPECT_FALSE(result2.isIntersecting);

    auto result3 = sph.closestIntersection(Ray2D({3.0, 3.0}, {0.0, 1.0}));
    EXPECT_TRUE(result3.isIntersecting);
    EXPECT_DOUBLE_EQ(1.0, result3.distance);
    EXPECT_DOUBLE_EQ(3.0, result3.point.x);
    EXPECT_DOUBLE_EQ(4.0, result3.point.y);
}

TEST(Sphere2, BoundingBox) {
    Sphere2 sph({3.0, -1.0}, 5.0);
    BoundingBox2D bbox = sph.boundingBox();

    EXPECT_DOUBLE_EQ(-2.0, bbox.lowerCorner.x);
    EXPECT_DOUBLE_EQ(-6.0, bbox.lowerCorner.y);
    EXPECT_DOUBLE_EQ(8.0, bbox.upperCorner.x);
    EXPECT_DOUBLE_EQ(4.0, bbox.upperCorner.y);
}

TEST(Sphere2, ClosestNormal) {
    Sphere2 sph({3.0, -1.0}, 5.0);
    sph.isNormalFlipped = true;

    auto result1 = sph.closestNormal({10.0, -1.0});
    EXPECT_DOUBLE_EQ(-1.0, result1.x);
    EXPECT_DOUBLE_EQ(0.0, result1.y);

    auto result2 = sph.closestNormal({3.0, -10.0});
    EXPECT_DOUBLE_EQ(0.0, result2.x);
    EXPECT_DOUBLE_EQ(1.0, result2.y);

    auto result3 = sph.closestNormal({3.0, 3.0});
    EXPECT_DOUBLE_EQ(0.0, result3.x);
    EXPECT_DOUBLE_EQ(-1.0, result3.y);
}

TEST(Sphere2, Builder) {
    Sphere2 sph =
        Sphere2::builder().withCenter({3.0, -1.0}).withRadius(5.0).build();
    EXPECT_DOUBLE_EQ(3.0, sph.center.x);
    EXPECT_DOUBLE_EQ(-1.0, sph.center.y);
    EXPECT_DOUBLE_EQ(5.0, sph.radius);
}
