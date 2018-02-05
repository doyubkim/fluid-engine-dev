// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/bounding_box2.h>

using namespace jet;

TEST(BoundingBox2, Constructors) {
    {
        BoundingBox2D box;

        EXPECT_DOUBLE_EQ(kMaxD, box.lowerCorner.x);
        EXPECT_DOUBLE_EQ(kMaxD, box.lowerCorner.y);

        EXPECT_DOUBLE_EQ(-kMaxD, box.upperCorner.x);
        EXPECT_DOUBLE_EQ(-kMaxD, box.upperCorner.y);
    }

    {
        BoundingBox2D box(Vector2D(-2.0, 3.0), Vector2D(4.0, -2.0));

        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner.x);
        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner.y);

        EXPECT_DOUBLE_EQ(4.0, box.upperCorner.x);
        EXPECT_DOUBLE_EQ(3.0, box.upperCorner.y);
    }

    {
        BoundingBox2D box(Vector2D(-2.0, 3.0), Vector2D(4.0, -2.0));
        BoundingBox2D box2(box);

        EXPECT_DOUBLE_EQ(-2.0, box2.lowerCorner.x);
        EXPECT_DOUBLE_EQ(-2.0, box2.lowerCorner.y);

        EXPECT_DOUBLE_EQ(4.0, box2.upperCorner.x);
        EXPECT_DOUBLE_EQ(3.0, box2.upperCorner.y);
    }
}

TEST(BoundingBox2, BasicGetters) {
    BoundingBox2D box(Vector2D(-2.0, 3.0), Vector2D(4.0, -2.0));

    EXPECT_DOUBLE_EQ(6.0, box.width());
    EXPECT_DOUBLE_EQ(5.0, box.height());
    EXPECT_DOUBLE_EQ(6.0, box.length(0));
    EXPECT_DOUBLE_EQ(5.0, box.length(1));
}

TEST(BoundingBox2, Overlaps) {
    // x-axis is not overlapping
    {
        BoundingBox2D box1(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        BoundingBox2D box2(Vector2D(5.0, 1.0), Vector2D(8.0, 2.0));

        EXPECT_FALSE(box1.overlaps(box2));
    }

    // y-axis is not overlapping
    {
        BoundingBox2D box1(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        BoundingBox2D box2(Vector2D(3.0, 4.0), Vector2D(8.0, 6.0));

        EXPECT_FALSE(box1.overlaps(box2));
    }

    // overlapping
    {
        BoundingBox2D box1(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        BoundingBox2D box2(Vector2D(3.0, 1.0), Vector2D(8.0, 2.0));

        EXPECT_TRUE(box1.overlaps(box2));
    }
}

TEST(BoundingBox2, Contains) {
    // Not containing (x-axis is out)
    {
        BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        Vector2D point(-3.0, 0.0);

        EXPECT_FALSE(box.contains(point));
    }

    // Not containing (y-axis is out)
    {
        BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        Vector2D point(2.0, 3.5);

        EXPECT_FALSE(box.contains(point));
    }

    // Containing
    {
        BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        Vector2D point(2.0, 0.0);

        EXPECT_TRUE(box.contains(point));
    }
}

TEST(BoundingBox2, Intersects) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));

    Ray2D ray1(Vector2D(-3, 0), Vector2D(2, 1).normalized());
    EXPECT_TRUE(box.intersects(ray1));

    Ray2D ray2(Vector2D(3, -1), Vector2D(-1, 2).normalized());
    EXPECT_TRUE(box.intersects(ray2));

    Ray2D ray3(Vector2D(1, -5), Vector2D(2, 1).normalized());
    EXPECT_FALSE(box.intersects(ray3));
}

TEST(BoundingBox2, ClosestIntersection) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(1.0, 0.0));

    Ray2D ray1(Vector2D(-4, -3), Vector2D(1, 1).normalized());
    BoundingBoxRayIntersection2D intersection1 = box.closestIntersection(ray1);
    EXPECT_TRUE(intersection1.isIntersecting);
    EXPECT_DOUBLE_EQ(Vector2D(2, 2).length(), intersection1.tNear);
    EXPECT_DOUBLE_EQ(Vector2D(3, 3).length(), intersection1.tFar);

    Ray2D ray2(Vector2D(0, -1), Vector2D(-2, 1).normalized());
    BoundingBoxRayIntersection2D intersection2 = box.closestIntersection(ray2);
    EXPECT_TRUE(intersection2.isIntersecting);
    EXPECT_DOUBLE_EQ(Vector2D(2, 1).length(), intersection2.tNear);
}

TEST(BoundingBox2, MidPoint) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    Vector2D midPoint = box.midPoint();

    EXPECT_DOUBLE_EQ(1.0, midPoint.x);
    EXPECT_DOUBLE_EQ(0.5, midPoint.y);
}

TEST(BoundingBox2, DiagonalLength) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    double diagLen = box.diagonalLength();

    EXPECT_DOUBLE_EQ(std::sqrt(6.0 * 6.0 + 5.0 * 5.0), diagLen);
}

TEST(BoundingBox2, DiagonalLengthSquared) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    double diagLenSqr = box.diagonalLengthSquared();

    EXPECT_DOUBLE_EQ(6.0 * 6.0 + 5.0 * 5.0, diagLenSqr);
}

TEST(BoundingBox2, Reset) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    box.reset();

    static const double maxDouble = std::numeric_limits<double>::max();

    EXPECT_DOUBLE_EQ(maxDouble, box.lowerCorner.x);
    EXPECT_DOUBLE_EQ(maxDouble, box.lowerCorner.y);

    EXPECT_DOUBLE_EQ(-maxDouble, box.upperCorner.x);
    EXPECT_DOUBLE_EQ(-maxDouble, box.upperCorner.y);
}

TEST(BoundingBox2, Merge) {
    // Merge with point
    {
        BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        Vector2D point(5.0, 1.0);

        box.merge(point);

        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner.x);
        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner.y);

        EXPECT_DOUBLE_EQ(5.0, box.upperCorner.x);
        EXPECT_DOUBLE_EQ(3.0, box.upperCorner.y);
    }

    // Merge with other box
    {
        BoundingBox2D box1(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
        BoundingBox2D box2(Vector2D(3.0, 1.0), Vector2D(8.0, 2.0));

        box1.merge(box2);

        EXPECT_DOUBLE_EQ(-2.0, box1.lowerCorner.x);
        EXPECT_DOUBLE_EQ(-2.0, box1.lowerCorner.y);

        EXPECT_DOUBLE_EQ(8.0, box1.upperCorner.x);
        EXPECT_DOUBLE_EQ(3.0, box1.upperCorner.y);
    }
}

TEST(BoundingBox2, Expand) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    box.expand(3.0);

    EXPECT_DOUBLE_EQ(-5.0, box.lowerCorner.x);
    EXPECT_DOUBLE_EQ(-5.0, box.lowerCorner.y);

    EXPECT_DOUBLE_EQ(7.0, box.upperCorner.x);
    EXPECT_DOUBLE_EQ(6.0, box.upperCorner.y);
}

TEST(BoundingBox2, Corner) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    EXPECT_VECTOR2_EQ(Vector2D(-2.0, -2.0), box.corner(0));
    EXPECT_VECTOR2_EQ(Vector2D(4.0, -2.0), box.corner(1));
    EXPECT_VECTOR2_EQ(Vector2D(-2.0, 3.0), box.corner(2));
    EXPECT_VECTOR2_EQ(Vector2D(4.0, 3.0), box.corner(3));
}

TEST(BoundingBox2D, IsEmpty) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    EXPECT_FALSE(box.isEmpty());

    box.lowerCorner = Vector2D(5.0, 1.0);
    EXPECT_TRUE(box.isEmpty());

    box.lowerCorner = Vector2D(2.0, 4.0);
    EXPECT_TRUE(box.isEmpty());

    box.lowerCorner = Vector2D(4.0, 1.0);
    EXPECT_TRUE(box.isEmpty());
}
