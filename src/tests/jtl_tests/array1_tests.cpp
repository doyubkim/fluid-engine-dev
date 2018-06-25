// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array.h>
#include <jet/parallel.h>
#include <jet/serialization.h>

#include <gtest/gtest.h>

#include <vector>

using namespace jet;

TEST(Array1, Constructors) {
    {
        Array1<float> arr;
        EXPECT_EQ(0u, arr.length());
    }
    {
        Array1<float> arr(9, 1.5f);
        EXPECT_EQ(9u, arr.length());
        for (size_t i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(1.5f, arr[i]);
        }
    }
    {
        Array1<float> arr({1.f, 2.f, 3.f, 4.f});
        EXPECT_EQ(4u, arr.length());
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
        }
    }
    {
        Array1<float> arr({1.f, 2.f, 3.f, 4.f});
        Array1<float> arr1(arr);
        EXPECT_EQ(4u, arr1.length());
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr1[i]);
        }
    }
    {
        Array1<float> arr({1.f, 2.f, 3.f, 4.f});
        ArrayView1<float> arrView(arr.data(), arr.size());
        EXPECT_EQ(4u, arrView.length());
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arrView[i]);
        }
    }
}

TEST(Array1, SetMethods) {
    Array1<float> arr1(12, -1.f);
    fill(arr1.view(), 3.5f);
    for (float a : arr1) {
        EXPECT_EQ(3.5f, a);
    }

    Array1<float> arr2;
    arr1.set(arr2);
    EXPECT_EQ(arr1.length(), arr2.length());
    for (size_t i = 0; i < arr2.length(); ++i) {
        EXPECT_EQ(arr1[i], arr2[i]);
    }

    arr2 = {2.f, 5.f, 9.f, -1.f};
    EXPECT_EQ(4u, arr2.length());
    EXPECT_EQ(2.f, arr2[0]);
    EXPECT_EQ(5.f, arr2[1]);
    EXPECT_EQ(9.f, arr2[2]);
    EXPECT_EQ(-1.f, arr2[3]);

    ArrayView1<float> arrView(arr2.data(), arr2.size());
    EXPECT_EQ(4u, arrView.length());
    EXPECT_EQ(2.f, arrView[0]);
    EXPECT_EQ(5.f, arrView[1]);
    EXPECT_EQ(9.f, arrView[2]);
    EXPECT_EQ(-1.f, arrView[3]);
}

TEST(Array1, Clear) {
    Array1<float> arr1 = {2.f, 5.f, 9.f, -1.f};
    arr1.clear();
    EXPECT_EQ(0u, arr1.length());
}

TEST(Array1, ResizeMethod) {
    {
        Array1<float> arr;
        arr.resize(9);
        EXPECT_EQ(9u, arr.length());
        for (size_t i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(12, 4.f);
        EXPECT_EQ(12u, arr.length());
        for (size_t i = 0; i < 8; ++i) {
            if (i < 9) {
                EXPECT_FLOAT_EQ(0.f, arr[i]);
            } else {
                EXPECT_FLOAT_EQ(4.f, arr[i]);
            }
        }
    }
}

TEST(Array1, Iterators) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};

    size_t i = 0;
    for (float& elem : arr1) {
        EXPECT_FLOAT_EQ(arr1[i], elem);
        ++i;
    }

    i = 0;
    for (const float& elem : arr1) {
        EXPECT_FLOAT_EQ(arr1[i], elem);
        ++i;
    }
}

TEST(Array1, ForEach) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    size_t i = 0;
    std::for_each(arr1.begin(), arr1.end(), [&](float val) {
        EXPECT_FLOAT_EQ(arr1[i], val);
        ++i;
    });
}

TEST(Array1, ForEachIndex) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    size_t cnt = 0;
    forEachIndex(arr1.length(), [&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}

TEST(Array1, ParallelForEachIndex) {
    Array1<float> arr1(200);
    forEachIndex(arr1.length(),
                 [&](size_t i) { arr1[i] = static_cast<float>(200.f - i); });

    parallelForEachIndex(arr1.length(), [&](size_t i) {
        float ans = static_cast<float>(200.f - i);
        EXPECT_EQ(ans, arr1[i]);
    });
}
