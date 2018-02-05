// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/point3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Point2, Constructors) {
    Point2F pt;
    EXPECT_FLOAT_EQ(0.f, pt.x);
    EXPECT_FLOAT_EQ(0.f, pt.y);

    Point2F pt2(5.f, 3.f);
    EXPECT_FLOAT_EQ(5.f, pt2.x);
    EXPECT_FLOAT_EQ(3.f, pt2.y);

    Point2F pt5 = { 7.f, 6.f };
    EXPECT_FLOAT_EQ(7.f, pt5.x);
    EXPECT_FLOAT_EQ(6.f, pt5.y);

    Point2F pt6(pt5);
    EXPECT_FLOAT_EQ(7.f, pt6.x);
    EXPECT_FLOAT_EQ(6.f, pt6.y);
}

TEST(Point2, SetMethods) {
    Point2F pt;
    pt.set(4.f, 2.f);
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(2.f, pt.y);

    auto lst = {0.f, 5.f};
    pt.set(lst);
    EXPECT_FLOAT_EQ(0.f, pt.x);
    EXPECT_FLOAT_EQ(5.f, pt.y);

    pt.set(Point2F(9.f, 8.f));
    EXPECT_FLOAT_EQ(9.f, pt.x);
    EXPECT_FLOAT_EQ(8.f, pt.y);
}

TEST(Point2, BasicSetterMethods) {
    Point2F pt(3.f, 9.f);
    pt.setZero();
    EXPECT_FLOAT_EQ(0.f, pt.x);
    EXPECT_FLOAT_EQ(0.f, pt.y);
}

TEST(Point2, BinaryOperatorMethods) {
    Point2F pt(3.f, 9.f);
    pt = pt.add(4.f);
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(13.f, pt.y);

    pt = pt.add(Point2F(-2.f, 1.f));
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(14.f, pt.y);

    pt = pt.sub(8.f);
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt = pt.sub(Point2F(-5.f, 3.f));
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);

    pt = pt.mul(2.f);
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt = pt.mul(Point2F(3.f, -2.f));
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);

    pt = pt.div(4.f);
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);

    pt = pt.div(Point2F(3.f, -1.f));
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
}

TEST(Point2, BinaryInverseOperatorMethods) {
    Point2F pt(3.f, 9.f);
    pt = pt.rsub(8.f);
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(-1.f, pt.y);

    pt = pt.rsub(Point2F(-5.f, 3.f));
    EXPECT_FLOAT_EQ(-10.f, pt.x);
    EXPECT_FLOAT_EQ(4.f, pt.y);

    pt = Point2F(-4.f, -3.f);
    pt = pt.rdiv(12.f);
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, -4.f);

    pt = pt.rdiv(Point2F(3.f, -16.f));
    EXPECT_FLOAT_EQ(-1.f, pt.x);
    EXPECT_FLOAT_EQ(4.f, pt.y);
}

TEST(Point2, AugmentedOperatorMethods) {
    Point2F pt(3.f, 9.f);
    pt.iadd(4.f);
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, 13.f);

    pt.iadd(Point2F(-2.f, 1.f));
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, 14.f);

    pt.isub(8.f);
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt.isub(Point2F(-5.f, 3.f));
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);

    pt.imul(2.f);
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt.imul(Point2F(3.f, -2.f));
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);

    pt.idiv(4.f);
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);

    pt.idiv(Point2F(3.f, -1.f));
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
}

TEST(Point2, AtMethod) {
    Point2F pt(8.f, 9.f);
    EXPECT_FLOAT_EQ(pt.at(0), 8.f);
    EXPECT_FLOAT_EQ(pt.at(1), 9.f);

    pt.at(0) = 7.f;
    pt.at(1) = 6.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
}

TEST(Point2, BasicGetterMethods) {
    Point2F pt(3.f, 7.f), pt2(-3.f, -7.f);

    float sum = pt.sum();
    EXPECT_FLOAT_EQ(sum, 10.f);

    float min = pt.min();
    EXPECT_FLOAT_EQ(min, 3.f);

    float max = pt.max();
    EXPECT_FLOAT_EQ(max, 7.f);

    float absmin = pt2.absmin();
    EXPECT_FLOAT_EQ(absmin, -3.f);

    float absmax = pt2.absmax();
    EXPECT_FLOAT_EQ(absmax, -7.f);

    size_t daxis = pt.dominantAxis();
    EXPECT_EQ(daxis, (size_t)1);

    size_t saxis = pt.subminantAxis();
    EXPECT_EQ(saxis, (size_t)0);
}

TEST(Point2, BracketOperator) {
    Point2F pt(8.f, 9.f);
    EXPECT_FLOAT_EQ(pt[0], 8.f);
    EXPECT_FLOAT_EQ(pt[1], 9.f);

    pt[0] = 7.f;
    pt[1] = 6.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
}

TEST(Point2, AssignmentOperator) {
    Point2F pt(5.f, 1.f);
    Point2F pt2(3.f, 3.f);
    pt2 = pt;
    EXPECT_FLOAT_EQ(5.f, pt2.x);
    EXPECT_FLOAT_EQ(pt2.y, 1.f);
}

TEST(Point2, AugmentedOperators) {
    Point2F pt(3.f, 9.f);
    pt += 4.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, 13.f);

    pt += Point2F(-2.f, 1.f);
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, 14.f);

    pt -= 8.f;
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt -= Point2F(-5.f, 3.f);
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);

    pt *= 2.f;
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt *= Point2F(3.f, -2.f);
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);

    pt /= 4.f;
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);

    pt /= Point2F(3.f, -1.f);
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
}

TEST(Point2, EqualOperator) {
    Point2F pt, pt2(3.f, 7.f), pt3(3.f, 5.f), pt4(5.f, 1.f);
    pt = pt2;
    EXPECT_TRUE(pt == pt2);
    EXPECT_FALSE(pt == pt3);
    EXPECT_FALSE(pt != pt2);
    EXPECT_TRUE(pt != pt3);
    EXPECT_TRUE(pt != pt4);
}

TEST(Point2, MinMaxFunction) {
    Point2F pt(5.f, 1.f);
    Point2F pt2(3.f, 3.f);
    Point2F minPoint = min(pt, pt2);
    Point2F maxPoint = max(pt, pt2);
    EXPECT_EQ(Point2F(3.f, 1.f), minPoint);
    EXPECT_EQ(Point2F(5.f, 3.f), maxPoint);
}

TEST(Point2, ClampFunction) {
    Point2F pt(2.f, 4.f), low(3.f, -1.f), high(5.f, 2.f);
    Point2F clampedVec = clamp(pt, low, high);
    EXPECT_EQ(Point2F(3.f, 2.f), clampedVec);
}

TEST(Point2, CeilFloorFunction) {
    Point2F pt(2.2f, 4.7f);
    Point2F ceilVec = ceil(pt);
    EXPECT_EQ(Point2F(3.f, 5.f), ceilVec);

    Point2F floorVec = floor(pt);
    EXPECT_EQ(Point2F(2.f, 4.f), floorVec);
}

TEST(Point2, BinaryOperators) {
    Point2F pt(3.f, 9.f);
    pt = pt + 4.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, 13.f);

    pt = pt + Point2F(-2.f, 1.f);
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(pt.y, 14.f);

    pt = pt - 8.f;
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt = pt - Point2F(-5.f, 3.f);
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);

    pt = pt * 2.f;
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);

    pt = pt * Point2F(3.f, -2.f);
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);

    pt = pt / 4.f;
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);

    pt = pt / Point2F(3.f, -1.f);
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
}
