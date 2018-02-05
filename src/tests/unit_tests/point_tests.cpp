// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/point.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Point, Constructors) {
    Point<double, 5> pt1;

    EXPECT_DOUBLE_EQ(0.0, pt1[0]);
    EXPECT_DOUBLE_EQ(0.0, pt1[1]);
    EXPECT_DOUBLE_EQ(0.0, pt1[2]);
    EXPECT_DOUBLE_EQ(0.0, pt1[3]);
    EXPECT_DOUBLE_EQ(0.0, pt1[4]);

    Point<double, 5> pt2({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_DOUBLE_EQ(1.0, pt2[0]);
    EXPECT_DOUBLE_EQ(2.0, pt2[1]);
    EXPECT_DOUBLE_EQ(3.0, pt2[2]);
    EXPECT_DOUBLE_EQ(4.0, pt2[3]);
    EXPECT_DOUBLE_EQ(5.0, pt2[4]);

    Point<double, 5> pt3(pt2);

    EXPECT_DOUBLE_EQ(1.0, pt3[0]);
    EXPECT_DOUBLE_EQ(2.0, pt3[1]);
    EXPECT_DOUBLE_EQ(3.0, pt3[2]);
    EXPECT_DOUBLE_EQ(4.0, pt3[3]);
    EXPECT_DOUBLE_EQ(5.0, pt3[4]);
}

TEST(Point, SetMethods) {
    Point<double, 5> pt1;

    pt1.set({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_DOUBLE_EQ(1.0, pt1[0]);
    EXPECT_DOUBLE_EQ(2.0, pt1[1]);
    EXPECT_DOUBLE_EQ(3.0, pt1[2]);
    EXPECT_DOUBLE_EQ(4.0, pt1[3]);
    EXPECT_DOUBLE_EQ(5.0, pt1[4]);

    Point<double, 5> pt2;

    pt2.set(pt1);

    EXPECT_DOUBLE_EQ(1.0, pt2[0]);
    EXPECT_DOUBLE_EQ(2.0, pt2[1]);
    EXPECT_DOUBLE_EQ(3.0, pt2[2]);
    EXPECT_DOUBLE_EQ(4.0, pt2[3]);
    EXPECT_DOUBLE_EQ(5.0, pt2[4]);
}

TEST(Point, AssignmentOperators) {
    Point<double, 5> pt1;

    pt1 = {1.0, 2.0, 3.0, 4.0, 5.0};

    EXPECT_DOUBLE_EQ(1.0, pt1[0]);
    EXPECT_DOUBLE_EQ(2.0, pt1[1]);
    EXPECT_DOUBLE_EQ(3.0, pt1[2]);
    EXPECT_DOUBLE_EQ(4.0, pt1[3]);
    EXPECT_DOUBLE_EQ(5.0, pt1[4]);

    Point<double, 5> pt2;

    pt2 = pt1;

    EXPECT_DOUBLE_EQ(1.0, pt2[0]);
    EXPECT_DOUBLE_EQ(2.0, pt2[1]);
    EXPECT_DOUBLE_EQ(3.0, pt2[2]);
    EXPECT_DOUBLE_EQ(4.0, pt2[3]);
    EXPECT_DOUBLE_EQ(5.0, pt2[4]);
}

TEST(Point, BracketOperators) {
    Point<double, 5> pt1;

    pt1[0] = 1.0;
    pt1[1] = 2.0;
    pt1[2] = 3.0;
    pt1[3] = 4.0;
    pt1[4] = 5.0;

    EXPECT_DOUBLE_EQ(1.0, pt1[0]);
    EXPECT_DOUBLE_EQ(2.0, pt1[1]);
    EXPECT_DOUBLE_EQ(3.0, pt1[2]);
    EXPECT_DOUBLE_EQ(4.0, pt1[3]);
    EXPECT_DOUBLE_EQ(5.0, pt1[4]);
}
