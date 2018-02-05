// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/point3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Point3, Constructors) {
    Point3F pt;
    EXPECT_FLOAT_EQ(0.f, pt.x);
    EXPECT_FLOAT_EQ(0.f, pt.y);
    EXPECT_FLOAT_EQ(0.f, pt.z);

    Point3F pt2(5.f, 3.f, 8.f);
    EXPECT_FLOAT_EQ(5.f, pt2.x);
    EXPECT_FLOAT_EQ(3.f, pt2.y);
    EXPECT_FLOAT_EQ(8.f, pt2.z);

    Point2F pt3(4.f, 7.f);
    Point3F pt4(pt3, 9.f);
    EXPECT_FLOAT_EQ(4.f, pt4.x);
    EXPECT_FLOAT_EQ(7.f, pt4.y);
    EXPECT_FLOAT_EQ(9.f, pt4.z);

    Point3F pt5 = { 7.f, 6.f, 1.f };
    EXPECT_FLOAT_EQ(7.f, pt5.x);
    EXPECT_FLOAT_EQ(6.f, pt5.y);
    EXPECT_FLOAT_EQ(1.f, pt5.z);

    Point3F pt6(pt5);
    EXPECT_FLOAT_EQ(7.f, pt6.x);
    EXPECT_FLOAT_EQ(6.f, pt6.y);
    EXPECT_FLOAT_EQ(1.f, pt6.z);
}

TEST(Point3, SetMethods) {
    Point3F pt;
    pt.set(4.f, 2.f, 8.f);
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(2.f, pt.y);
    EXPECT_FLOAT_EQ(8.f, pt.z);

    pt.set(Point2F(1.f, 3.f), 10.f);
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(10.f, pt.z);

    auto lst = {0.f, 5.f, 6.f};
    pt.set(lst);
    EXPECT_FLOAT_EQ(0.f, pt.x);
    EXPECT_FLOAT_EQ(5.f, pt.y);
    EXPECT_FLOAT_EQ(6.f, pt.z);

    pt.set(Point3F(9.f, 8.f, 2.f));
    EXPECT_FLOAT_EQ(9.f, pt.x);
    EXPECT_FLOAT_EQ(8.f, pt.y);
    EXPECT_FLOAT_EQ(2.f, pt.z);
}

TEST(Point3, BasicSetterMethods) {
    Point3F pt(3.f, 9.f, 4.f);
    pt.setZero();
    EXPECT_FLOAT_EQ(0.f, pt.x);
    EXPECT_FLOAT_EQ(0.f, pt.y);
    EXPECT_FLOAT_EQ(0.f, pt.z);
}

TEST(Point3, BinaryOperatorMethods) {
    Point3F pt(3.f, 9.f, 4.f);
    pt = pt.add(4.f);
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(13.f, pt.y);
    EXPECT_FLOAT_EQ(8.f, pt.z);

    pt = pt.add(Point3F(-2.f, 1.f, 5.f));
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(14.f, pt.y);
    EXPECT_FLOAT_EQ(13.f, pt.z);

    pt = pt.sub(8.f);
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(5.f, pt.z);

    pt = pt.sub(Point3F(-5.f, 3.f, 12.f));
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);

    pt = pt.mul(2.f);
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(-14.f,pt.z);

    pt = pt.mul(Point3F(3.f, -2.f, 0.5f));
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);

    pt = pt.div(4.f);
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);
    EXPECT_FLOAT_EQ(-1.75f, pt.z);

    pt = pt.div(Point3F(3.f, -1.f, 0.25f));
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);
}

TEST(Point3, BinaryInverseOperatorMethods) {
    Point3F pt(5.f, 14.f, 13.f);
    pt = pt.rsub(8.f);
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-6.f, pt.y);
    EXPECT_FLOAT_EQ(-5.f, pt.z);

    pt = pt.rsub(Point3F(-5.f, 3.f, -1.f));
    EXPECT_FLOAT_EQ(-8.f, pt.x);
    EXPECT_FLOAT_EQ(9.f, pt.y);
    EXPECT_FLOAT_EQ(4.f, pt.z);

    pt = Point3F(-12.f, -9.f, 8.f);
    pt = pt.rdiv(36.f);
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(-4.f, pt.y);
    EXPECT_FLOAT_EQ(4.5f, pt.z);

    pt = pt.rdiv(Point3F(3.f, -16.f, 18.f));
    EXPECT_FLOAT_EQ(-1.f, pt.x);
    EXPECT_FLOAT_EQ(4.f, pt.y);
    EXPECT_FLOAT_EQ(4.f, pt.z);
}

TEST(Point3, AugmentedOperatorMethods) {
    Point3F pt(3.f, 9.f, 4.f);
    pt.iadd(4.f);
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(13.f, pt.y);
    EXPECT_FLOAT_EQ(8.f, pt.z);

    pt.iadd(Point3F(-2.f, 1.f, 5.f));
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(14.f, pt.y);
    EXPECT_FLOAT_EQ(13.f, pt.z);

    pt.isub(8.f);
    EXPECT_FLOAT_EQ(-3.f,pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(5.f, pt.z);

    pt.isub(Point3F(-5.f, 3.f, 12.f));
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);

    pt.imul(2.f);
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(-14.f, pt.z);

    pt.imul(Point3F(3.f, -2.f, 0.5f));
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);

    pt.idiv(4.f);
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);
    EXPECT_FLOAT_EQ(-1.75f,pt.z);

    pt.idiv(Point3F(3.f, -1.f, 0.25f));
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);
}

TEST(Point3, AtMethods) {
    Point3F pt(8.f, 9.f, 1.f);
    EXPECT_FLOAT_EQ(8.f, pt.at(0));
    EXPECT_FLOAT_EQ(9.f, pt.at(1));
    EXPECT_FLOAT_EQ(1.f, pt.at(2));

    pt.at(0) = 7.f;
    pt.at(1) = 6.f;
    pt.at(2) = 5.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(5.f, pt.z);
}

TEST(Point3, BasicGetterMethods) {
    Point3F pt(3.f, 7.f, -1.f), pt2(-3.f, -7.f, 1.f);

    float sum = pt.sum();
    EXPECT_FLOAT_EQ(9.f, sum);

    float min = pt.min();
    EXPECT_FLOAT_EQ(-1.f, min);

    float max = pt.max();
    EXPECT_FLOAT_EQ(7.f, max);

    float absmin = pt2.absmin();
    EXPECT_FLOAT_EQ(1.f, absmin);

    float absmax = pt2.absmax();
    EXPECT_FLOAT_EQ(-7.f, absmax);

    size_t daxis = pt.dominantAxis();
    EXPECT_EQ((size_t)1, daxis);

    size_t saxis = pt.subminantAxis();
    EXPECT_EQ((size_t)2, saxis);
}

TEST(Point3, BracketOperators) {
    Point3F pt(8.f, 9.f, 1.f);
    EXPECT_FLOAT_EQ(8.f, pt[0]);
    EXPECT_FLOAT_EQ(9.f, pt[1]);
    EXPECT_FLOAT_EQ(1.f, pt[2]);

    pt[0] = 7.f;
    pt[1] = 6.f;
    pt[2] = 5.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(5.f, pt.z);
}

TEST(Point3, AssignmentOperators) {
    Point3F pt(5.f, 1.f, 0.f);
    Point3F pt2(3.f, 3.f, 3.f);
    pt2 = pt;
    EXPECT_FLOAT_EQ(5.f, pt2.x);
    EXPECT_FLOAT_EQ(1.f, pt2.y);
    EXPECT_FLOAT_EQ(0.f, pt2.z);
}

TEST(Point3, AugmentedOperators) {
    Point3F pt(3.f, 9.f, -2.f);
    pt += 4.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(13.f, pt.y);
    EXPECT_FLOAT_EQ(2.f, pt.z);

    pt += Point3F(-2.f, 1.f, 5.f);
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(14.f, pt.y);
    EXPECT_FLOAT_EQ(7.f, pt.z);

    pt -= 8.f;
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(-1.f, pt.z);

    pt -= Point3F(-5.f, 3.f, -6.f);
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(5.f, pt.z);

    pt *= 2.f;
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(10.f, pt.z);

    pt *= Point3F(3.f, -2.f, 0.4f);
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);
    EXPECT_FLOAT_EQ(4.f, pt.z);

    pt /= 4.f;
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);
    EXPECT_FLOAT_EQ(1.f, pt.z);

    pt /= Point3F(3.f, -1.f, 2.f);
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(0.5f, pt.z);
}

TEST(Point3, EqualOperatators) {
    Point3F pt, pt2(3.f, 7.f, 4.f), pt3(3.f, 5.f, 4.f), pt4(5.f, 1.f, 2.f);
    pt = pt2;
    EXPECT_TRUE(pt == pt2);
    EXPECT_FALSE(pt == pt3);
    EXPECT_FALSE(pt != pt2);
    EXPECT_TRUE(pt != pt3);
    EXPECT_TRUE(pt != pt4);
}

TEST(Point3, MinMaxFunctions) {
    Point3F pt(5.f, 1.f, 0.f);
    Point3F pt2(3.f, 3.f, 3.f);
    Point3F minPoint = min(pt, pt2);
    Point3F maxPoint = max(pt, pt2);
    EXPECT_TRUE(minPoint == Point3F(3.f, 1.f, 0.f));
    EXPECT_TRUE(maxPoint == Point3F(5.f, 3.f, 3.f));
}

TEST(Point3, ClampFunction) {
    Point3F pt(2.f, 4.f, 1.f), low(3.f, -1.f, 0.f), high(5.f, 2.f, 3.f);
    Point3F clampedVec = clamp(pt, low, high);
    EXPECT_TRUE(clampedVec == Point3F(3.f, 2.f, 1.f));
}

TEST(Point3, CeilFloorFunctions) {
    Point3F pt(2.2f, 4.7f, -0.2f);
    Point3F ceilVec = ceil(pt);
    EXPECT_TRUE(ceilVec == Point3F(3.f, 5.f, 0.f));

    Point3F floorVec = floor(pt);
    EXPECT_TRUE(floorVec == Point3F(2.f, 4.f, -1.f));
}

TEST(Point3, BinaryOperators) {
    Point3F pt(3.f, 9.f, 4.f);
    pt = pt + 4.f;
    EXPECT_FLOAT_EQ(7.f, pt.x);
    EXPECT_FLOAT_EQ(13.f,pt.y);
    EXPECT_FLOAT_EQ(8.f, pt.z);

    pt = pt + Point3F(-2.f, 1.f, 5.f);
    EXPECT_FLOAT_EQ(5.f, pt.x);
    EXPECT_FLOAT_EQ(14.f, pt.y);
    EXPECT_FLOAT_EQ(13.f, pt.z);

    pt = pt - 8.f;
    EXPECT_FLOAT_EQ(-3.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(5.f, pt.z);

    pt = pt - Point3F(-5.f, 3.f, 12.f);
    EXPECT_FLOAT_EQ(2.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);

    pt = pt * 2.f;
    EXPECT_FLOAT_EQ(4.f, pt.x);
    EXPECT_FLOAT_EQ(6.f, pt.y);
    EXPECT_FLOAT_EQ(-14.f, pt.z);

    pt = pt * Point3F(3.f, -2.f, 0.5f);
    EXPECT_FLOAT_EQ(12.f, pt.x);
    EXPECT_FLOAT_EQ(-12.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);

    pt = pt / 4.f;
    EXPECT_FLOAT_EQ(3.f, pt.x);
    EXPECT_FLOAT_EQ(-3.f, pt.y);
    EXPECT_FLOAT_EQ(-1.75f, pt.z);

    pt = pt / Point3F(3.f, -1.f, 0.25f);
    EXPECT_FLOAT_EQ(1.f, pt.x);
    EXPECT_FLOAT_EQ(3.f, pt.y);
    EXPECT_FLOAT_EQ(-7.f, pt.z);
}
