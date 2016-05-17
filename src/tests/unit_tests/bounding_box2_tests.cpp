// Copyright (c) 2016 Doyub Kim

#include <jet/bounding_box2.h>
#include <gtest/gtest.h>
#include <limits>

using namespace jet;

TEST(BoundingBox2, Constructors) {
    {
        BoundingBox2D box;

        static const double maxDouble = std::numeric_limits<double>::max();

        EXPECT_DOUBLE_EQ(maxDouble, box.lowerCorner.x);
        EXPECT_DOUBLE_EQ(maxDouble, box.lowerCorner.y);

        EXPECT_DOUBLE_EQ(-maxDouble, box.upperCorner.x);
        EXPECT_DOUBLE_EQ(-maxDouble, box.upperCorner.y);
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

TEST(BoundingBox2, MidPoint) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    Vector2D midPoint = box.midPoint();

    EXPECT_DOUBLE_EQ(1.0, midPoint.x);
    EXPECT_DOUBLE_EQ(0.5, midPoint.y);
}

TEST(BoundingBox2, DiagonalLength) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    double diagLen = box.diagonalLength();

    EXPECT_DOUBLE_EQ(std::sqrt(6.0*6.0 + 5.0*5.0), diagLen);
}

TEST(BoundingBox2, DiagonalLengthSquared) {
    BoundingBox2D box(Vector2D(-2.0, -2.0), Vector2D(4.0, 3.0));
    double diagLenSqr = box.diagonalLengthSquared();

    EXPECT_DOUBLE_EQ(6.0*6.0 + 5.0*5.0, diagLenSqr);
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
