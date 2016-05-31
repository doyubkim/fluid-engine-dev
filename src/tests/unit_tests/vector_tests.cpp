// Copyright (c) 2016 Doyub Kim

#include <jet/vector.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Vector, Constructors) {
    Vector<double, 5> vec1;

    EXPECT_DOUBLE_EQ(0.0, vec1[0]);
    EXPECT_DOUBLE_EQ(0.0, vec1[1]);
    EXPECT_DOUBLE_EQ(0.0, vec1[2]);
    EXPECT_DOUBLE_EQ(0.0, vec1[3]);
    EXPECT_DOUBLE_EQ(0.0, vec1[4]);

    Vector<double, 5> vec2({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_DOUBLE_EQ(1.0, vec2[0]);
    EXPECT_DOUBLE_EQ(2.0, vec2[1]);
    EXPECT_DOUBLE_EQ(3.0, vec2[2]);
    EXPECT_DOUBLE_EQ(4.0, vec2[3]);
    EXPECT_DOUBLE_EQ(5.0, vec2[4]);

    Vector<double, 5> vec3(vec2);

    EXPECT_DOUBLE_EQ(1.0, vec3[0]);
    EXPECT_DOUBLE_EQ(2.0, vec3[1]);
    EXPECT_DOUBLE_EQ(3.0, vec3[2]);
    EXPECT_DOUBLE_EQ(4.0, vec3[3]);
    EXPECT_DOUBLE_EQ(5.0, vec3[4]);
}

TEST(Vector, SetMethods) {
    Vector<double, 5> vec1;

    vec1.set({1.0, 2.0, 3.0, 4.0, 5.0});

    EXPECT_DOUBLE_EQ(1.0, vec1[0]);
    EXPECT_DOUBLE_EQ(2.0, vec1[1]);
    EXPECT_DOUBLE_EQ(3.0, vec1[2]);
    EXPECT_DOUBLE_EQ(4.0, vec1[3]);
    EXPECT_DOUBLE_EQ(5.0, vec1[4]);

    Vector<double, 5> vec2;

    vec2.set(vec1);

    EXPECT_DOUBLE_EQ(1.0, vec2[0]);
    EXPECT_DOUBLE_EQ(2.0, vec2[1]);
    EXPECT_DOUBLE_EQ(3.0, vec2[2]);
    EXPECT_DOUBLE_EQ(4.0, vec2[3]);
    EXPECT_DOUBLE_EQ(5.0, vec2[4]);
}

TEST(Vector, AssignmentOperators) {
    Vector<double, 5> vec1;

    vec1 = {1.0, 2.0, 3.0, 4.0, 5.0};

    EXPECT_DOUBLE_EQ(1.0, vec1[0]);
    EXPECT_DOUBLE_EQ(2.0, vec1[1]);
    EXPECT_DOUBLE_EQ(3.0, vec1[2]);
    EXPECT_DOUBLE_EQ(4.0, vec1[3]);
    EXPECT_DOUBLE_EQ(5.0, vec1[4]);

    Vector<double, 5> vec2;

    vec2 = vec1;

    EXPECT_DOUBLE_EQ(1.0, vec2[0]);
    EXPECT_DOUBLE_EQ(2.0, vec2[1]);
    EXPECT_DOUBLE_EQ(3.0, vec2[2]);
    EXPECT_DOUBLE_EQ(4.0, vec2[3]);
    EXPECT_DOUBLE_EQ(5.0, vec2[4]);
}

TEST(Vector, BracketOperators) {
    Vector<double, 5> vec1;

    vec1[0] = 1.0;
    vec1[1] = 2.0;
    vec1[2] = 3.0;
    vec1[3] = 4.0;
    vec1[4] = 5.0;

    EXPECT_DOUBLE_EQ(1.0, vec1[0]);
    EXPECT_DOUBLE_EQ(2.0, vec1[1]);
    EXPECT_DOUBLE_EQ(3.0, vec1[2]);
    EXPECT_DOUBLE_EQ(4.0, vec1[3]);
    EXPECT_DOUBLE_EQ(5.0, vec1[4]);
}
