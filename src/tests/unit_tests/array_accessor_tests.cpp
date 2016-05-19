// Copyright (c) 2016 Doyub Kim

#include <jet/array_accessor1.h>
#include <jet/array_accessor2.h>
#include <jet/array_accessor3.h>
#include <jet/array1.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ArrayAccessor1, Constructors) {
    double data[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = static_cast<double>(i);
    }

    ArrayAccessor1<double> acc(5, data);

    EXPECT_EQ(5u, acc.size());
    EXPECT_EQ(data, acc.data());
}

TEST(ArrayAccessor1, ForEach) {
    Array1<float> arr1 = {6.f,  4.f,  1.f,  -5.f};
    auto acc = arr1.accessor();

    size_t i = 0;
    acc.forEach([&](float val) {
        EXPECT_FLOAT_EQ(arr1[i], val);
        ++i;
    });
}

TEST(ArrayAccessor1, ForEachIndex) {
    Array1<float> arr1 = {6.f,  4.f,  1.f,  -5.f};
    auto acc = arr1.accessor();

    size_t cnt = 0;
    acc.forEachIndex([&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}

TEST(ArrayAccessor1, ParallelForEach) {
    Array1<float> arr1(200);
    auto acc = arr1.accessor();

    acc.forEachIndex([&](size_t i) {
        arr1[i] = static_cast<float>(200.f - i);
    });

    acc.parallelForEach([](float& val) {
        val *= 2.f;
    });

    acc.forEachIndex([&](size_t i) {
        float ans = 2.f * static_cast<float>(200.f - i);
        EXPECT_FLOAT_EQ(ans, arr1[i]);
    });
}

TEST(ArrayAccessor1, ParallelForEachIndex) {
    Array1<float> arr1(200);
    auto acc = arr1.accessor();

    acc.forEachIndex([&](size_t i) {
        arr1[i] = static_cast<float>(200.f - i);
    });

    acc.parallelForEachIndex([&](size_t i) {
        float ans = static_cast<float>(200.f - i);
        EXPECT_EQ(ans, arr1[i]);
    });
}

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

