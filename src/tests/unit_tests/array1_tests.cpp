// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/serialization.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(Array1, Constructors) {
    {
        Array1<float> arr;
        EXPECT_EQ(0u, arr.size());
    }
    {
        Array1<float> arr(9, 1.5f);
        EXPECT_EQ(9u, arr.size());
        for (size_t i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(1.5f, arr[i]);
        }
    }
    {
        Array1<float> arr({1.f,  2.f,  3.f,  4.f});
        EXPECT_EQ(4u, arr.size());
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
        }
    }
    {
        Array1<float> arr({1.f,  2.f,  3.f,  4.f});
        Array1<float> arr1(arr);
        EXPECT_EQ(4u, arr1.size());
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr1[i]);
        }
    }
}

TEST(Array1, SetMethods) {
    Array1<float> arr1(12, -1.f);
    arr1.set(3.5f);
    for (float a : arr1) {
        EXPECT_EQ(3.5f, a);
    }

    Array1<float> arr2;
    arr1.set(arr2);
    EXPECT_EQ(arr1.size(), arr2.size());
    for (size_t i = 0; i < arr2.size(); ++i) {
        EXPECT_EQ(arr1[i], arr2[i]);
    }

    arr2 = { 2.f, 5.f, 9.f, -1.f };
    EXPECT_EQ(4u, arr2.size());
    EXPECT_EQ(2.f, arr2[0]);
    EXPECT_EQ(5.f, arr2[1]);
    EXPECT_EQ(9.f, arr2[2]);
    EXPECT_EQ(-1.f, arr2[3]);
}

TEST(Array1, Clear) {
    Array1<float> arr1 = { 2.f, 5.f, 9.f, -1.f };
    arr1.clear();
    EXPECT_EQ(0u, arr1.size());
}

TEST(Array1, ResizeMethod) {
    {
        Array1<float> arr;
        arr.resize(9);
        EXPECT_EQ(9u, arr.size());
        for (size_t i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(12, 4.f);
        EXPECT_EQ(12u, arr.size());
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
    Array1<float> arr1 = {6.f,  4.f,  1.f,  -5.f};

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
    Array1<float> arr1 = {6.f,  4.f,  1.f,  -5.f};
    size_t i = 0;
    arr1.forEach([&](float val) {
        EXPECT_FLOAT_EQ(arr1[i], val);
        ++i;
    });
}

TEST(Array1, ForEachIndex) {
    Array1<float> arr1 = {6.f,  4.f,  1.f,  -5.f};
    size_t cnt = 0;
    arr1.forEachIndex([&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}

TEST(Array1, ParallelForEach) {
    Array1<float> arr1(200);
    arr1.forEachIndex([&](size_t i) {
        arr1[i] = static_cast<float>(200.f - i);
    });

    arr1.parallelForEach([](float& val) {
        val *= 2.f;
    });

    arr1.forEachIndex([&](size_t i) {
        float ans = 2.f * static_cast<float>(200.f - i);
        EXPECT_FLOAT_EQ(ans, arr1[i]);
    });
}

TEST(Array1, ParallelForEachIndex) {
    Array1<float> arr1(200);
    arr1.forEachIndex([&](size_t i) {
        arr1[i] = static_cast<float>(200.f - i);
    });

    arr1.parallelForEachIndex([&](size_t i) {
        float ans = static_cast<float>(200.f - i);
        EXPECT_EQ(ans, arr1[i]);
    });
}

TEST(Array1, Serialization) {
    Array1<float> arr1 = {1.f,  2.f,  3.f,  4.f};

    // Serialize to in-memoery stream
    std::vector<uint8_t> buffer1;
    serialize(arr1.constAccessor(), &buffer1);

    // Deserialize to non-zero array
    Array1<float> arr2 = {5.f, 6.f, 7.f};
    deserialize(buffer1, &arr2);
    EXPECT_EQ(4u, arr2.size());
    EXPECT_EQ(1.f, arr2[0]);
    EXPECT_EQ(2.f, arr2[1]);
    EXPECT_EQ(3.f, arr2[2]);
    EXPECT_EQ(4.f, arr2[3]);

    // Serialize zero-sized array
    Array1<float> arr3;
    serialize(arr3.constAccessor(), &buffer1);

    // Deserialize to non-zero array
    deserialize(buffer1, &arr3);
    EXPECT_EQ(0u, arr3.size());
}
