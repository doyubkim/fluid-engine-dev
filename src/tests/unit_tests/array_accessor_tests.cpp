// Copyright (c) 2016 Doyub Kim

#include <jet/array_accessor1.h>
#include <jet/array_accessor2.h>
#include <jet/array_accessor3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ConstArrayAccessor1, Constructors) {
    double data[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with ArrayAccessor1
    ArrayAccessor1<double> acc(5, data);
    ConstArrayAccessor1<double> cacc(acc);

    EXPECT_EQ(5u, cacc.size());
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstArrayAccessor2, Constructors) {
    double data[20];
    for (int i = 0; i < 20; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with ArrayAccessor2
    ArrayAccessor2<double> acc(Size2(5, 4), data);
    ConstArrayAccessor2<double> cacc(acc);

    EXPECT_EQ(5u, cacc.size().x);
    EXPECT_EQ(4u, cacc.size().y);
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstArrayAccessor3, Constructors) {
    double data[60];
    for (int i = 0; i < 60; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with ArrayAccessor3
    ArrayAccessor3<double> acc(Size3(5, 4, 3), data);
    ConstArrayAccessor3<double> cacc(acc);

    EXPECT_EQ(5u, cacc.size().x);
    EXPECT_EQ(4u, cacc.size().y);
    EXPECT_EQ(3u, cacc.size().z);
    EXPECT_EQ(data, cacc.data());
}

