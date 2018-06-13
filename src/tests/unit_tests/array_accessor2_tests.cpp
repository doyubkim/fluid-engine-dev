// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array.h>
#include <jet/array_view.h>
#include <jet/parallel.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(ArrayView2, Constructors) {
    double data[20];
    for (int i = 0; i < 20; ++i) {
        data[i] = static_cast<double>(i);
    }

    ArrayView2<double> acc(data, Size2(5, 4));

    EXPECT_EQ(5u, acc.size().x);
    EXPECT_EQ(4u, acc.size().y);
    EXPECT_EQ(data, acc.data());
}

TEST(ArrayView2, Iterators) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    float cnt = 1.f;
    for (float& elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }

    cnt = 1.f;
    for (const float& elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }
}

TEST(ArrayView2, ForEachIndex) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});

    forEachIndex(arr1.size(), [&](size_t i, size_t j) {
        size_t idx = i + (4 * j) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), arr1(i, j));
    });
}

TEST(ArrayView2, ParallelForEachIndex) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});

    parallelForEachIndex(arr1.size(), [&](size_t i, size_t j) {
        size_t idx = i + (4 * j) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), arr1(i, j));
    });
}



TEST(ConstArrayView2, Constructors) {
    double data[20];
    for (int i = 0; i < 20; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with ArrayView2
    ArrayView2<double> acc(data, Size2(5, 4));
    ConstArrayView2<double> cacc(acc);

    EXPECT_EQ(5u, cacc.size().x);
    EXPECT_EQ(4u, cacc.size().y);
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstArrayView2, Iterators) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    float cnt = 1.f;
    for (const float& elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }
}

TEST(ConstArrayView2, ForEach) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    size_t i = 0;
    std::for_each(acc.begin(), acc.end(), [&](float val) {
        EXPECT_FLOAT_EQ(acc[i], val);
        ++i;
    });
}

TEST(ConstArrayView2, ForEachIndex) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    forEachIndex(acc.size(), [&](size_t i, size_t j) {
        size_t idx = i + (4 * j) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), acc(i, j));
    });
}

TEST(ConstArrayView2, ParallelForEachIndex) {
    Array2<float> arr1(
        {{1.f,  2.f,  3.f,  4.f},
         {5.f,  6.f,  7.f,  8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    parallelForEachIndex(acc.size(), [&](size_t i, size_t j) {
        size_t idx = i + (4 * j) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), acc(i, j));
    });
}
