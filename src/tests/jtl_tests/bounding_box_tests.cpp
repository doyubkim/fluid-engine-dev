// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/bounding_box.h>
#include <gtest/gtest.h>
#include <limits>

using namespace jet;

TEST(BoundingBox, Constructors) {
    {
        BoundingBox<double, 2> box;

        EXPECT_DOUBLE_EQ(kMaxD, box.lowerCorner[0]);
        EXPECT_DOUBLE_EQ(kMaxD, box.lowerCorner[1]);

        EXPECT_DOUBLE_EQ(-kMaxD, box.upperCorner[0]);
        EXPECT_DOUBLE_EQ(-kMaxD, box.upperCorner[1]);
    }

    {
        BoundingBox<double, 2> box({-2.0, 3.0}, {4.0, -2.0});

        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner[0]);
        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner[1]);

        EXPECT_DOUBLE_EQ(4.0, box.upperCorner[0]);
        EXPECT_DOUBLE_EQ(3.0, box.upperCorner[1]);
    }

    {
        BoundingBox<double, 2> box({-2.0, 3.0}, {4.0, -2.0});
        BoundingBox<double, 2> box2(box);

        EXPECT_DOUBLE_EQ(-2.0, box2.lowerCorner[0]);
        EXPECT_DOUBLE_EQ(-2.0, box2.lowerCorner[1]);

        EXPECT_DOUBLE_EQ(4.0, box2.upperCorner[0]);
        EXPECT_DOUBLE_EQ(3.0, box2.upperCorner[1]);
    }
}

TEST(BoundingBox, Overlaps) {
    // x-axis is not overlapping
    {
        BoundingBox<double, 2> box1({-2.0, -2.0}, {4.0, 3.0});
        BoundingBox<double, 2> box2({5.0, 1.0}, {8.0, 2.0});

        EXPECT_FALSE(box1.overlaps(box2));
    }

    // y-axis is not overlapping
    {
        BoundingBox<double, 2> box1({-2.0, -2.0}, {4.0, 3.0});
        BoundingBox<double, 2> box2({3.0, 4.0}, {8.0, 6.0});

        EXPECT_FALSE(box1.overlaps(box2));
    }

    // overlapping
    {
        BoundingBox<double, 2> box1({-2.0, -2.0}, {4.0, 3.0});
        BoundingBox<double, 2> box2({3.0, 1.0}, {8.0, 2.0});

        EXPECT_TRUE(box1.overlaps(box2));
    }
}

TEST(BoundingBox, Contains) {
    // Not containing (x-axis is out)
    {
        BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
        Vector<double, 2> point({-3.0, 0.0});

        EXPECT_FALSE(box.contains(point));
    }

    // Not containing (y-axis is out)
    {
        BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
        Vector<double, 2> point({2.0, 3.5});

        EXPECT_FALSE(box.contains(point));
    }

    // Containing
    {
        BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
        Vector<double, 2> point({2.0, 0.0});

        EXPECT_TRUE(box.contains(point));
    }
}

TEST(BoundingBox, MidPoint) {
    BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
    Vector<double, 2> midPoint = box.midPoint();

    EXPECT_DOUBLE_EQ(1.0, midPoint[0]);
    EXPECT_DOUBLE_EQ(0.5, midPoint[1]);
}

TEST(BoundingBox, DiagonalLength) {
    BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
    double diagLen = box.diagonalLength();

    EXPECT_DOUBLE_EQ(std::sqrt(6.0*6.0 + 5.0*5.0), diagLen);
}

TEST(BoundingBox, DiagonalLengthSquared) {
    BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
    double diagLenSqr = box.diagonalLengthSquared();

    EXPECT_DOUBLE_EQ(6.0*6.0 + 5.0*5.0, diagLenSqr);
}

TEST(BoundingBox, Reset) {
    BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
    box.reset();

    static const double maxDouble = std::numeric_limits<double>::max();

    EXPECT_DOUBLE_EQ(maxDouble, box.lowerCorner[0]);
    EXPECT_DOUBLE_EQ(maxDouble, box.lowerCorner[1]);

    EXPECT_DOUBLE_EQ(-maxDouble, box.upperCorner[0]);
    EXPECT_DOUBLE_EQ(-maxDouble, box.upperCorner[1]);
}

TEST(BoundingBox, Merge) {
    // Merge with point
    {
        BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
        Vector<double, 2> point({5.0, 1.0});

        box.merge(point);

        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner[0]);
        EXPECT_DOUBLE_EQ(-2.0, box.lowerCorner[1]);

        EXPECT_DOUBLE_EQ(5.0, box.upperCorner[0]);
        EXPECT_DOUBLE_EQ(3.0, box.upperCorner[1]);
    }

    // Merge with other box
    {
        BoundingBox<double, 2> box1({-2.0, -2.0}, {4.0, 3.0});
        BoundingBox<double, 2> box2({3.0, 1.0}, {8.0, 2.0});

        box1.merge(box2);

        EXPECT_DOUBLE_EQ(-2.0, box1.lowerCorner[0]);
        EXPECT_DOUBLE_EQ(-2.0, box1.lowerCorner[1]);

        EXPECT_DOUBLE_EQ(8.0, box1.upperCorner[0]);
        EXPECT_DOUBLE_EQ(3.0, box1.upperCorner[1]);
    }
}

TEST(BoundingBox, Expand) {
    BoundingBox<double, 2> box({-2.0, -2.0}, {4.0, 3.0});
    box.expand(3.0);

    EXPECT_DOUBLE_EQ(-5.0, box.lowerCorner[0]);
    EXPECT_DOUBLE_EQ(-5.0, box.lowerCorner[1]);

    EXPECT_DOUBLE_EQ(7.0, box.upperCorner[0]);
    EXPECT_DOUBLE_EQ(6.0, box.upperCorner[1]);
}
