// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array3.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace jet;

TEST(Array3, Constructors) {
    {
        Array3<float> arr;
        EXPECT_EQ(0u, arr.width());
        EXPECT_EQ(0u, arr.height());
        EXPECT_EQ(0u, arr.depth());
    }
    {
        Array3<float> arr(Size3(3, 7, 4));
        EXPECT_EQ(3u, arr.width());
        EXPECT_EQ(7u, arr.height());
        EXPECT_EQ(4u, arr.depth());
        for (size_t i = 0; i < 84; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }
    }
    {
        Array3<float> arr(Size3(1, 9, 5), 1.5f);
        EXPECT_EQ(1u, arr.width());
        EXPECT_EQ(9u, arr.height());
        EXPECT_EQ(5u, arr.depth());
        for (size_t i = 0; i < 45; ++i) {
            EXPECT_FLOAT_EQ(1.5f, arr[i]);
        }
    }
    {
        Array3<float> arr(5, 2, 8);
        EXPECT_EQ(5u, arr.width());
        EXPECT_EQ(2u, arr.height());
        EXPECT_EQ(8u, arr.depth());
        for (size_t i = 0; i < 80; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }
    }
    {
        Array3<float> arr(3, 4, 2, 7.f);
        EXPECT_EQ(3u, arr.width());
        EXPECT_EQ(4u, arr.height());
        EXPECT_EQ(2u, arr.depth());
        for (size_t i = 0; i < 24; ++i) {
            EXPECT_FLOAT_EQ(7.f, arr[i]);
        }
    }
    {
        Array3<float> arr(
            {{{ 1.f,  2.f,  3.f,  4.f},
              { 5.f,  6.f,  7.f,  8.f},
              { 9.f, 10.f, 11.f, 12.f}},
             {{13.f, 14.f, 15.f, 16.f},
              {17.f, 18.f, 19.f, 20.f},
              {21.f, 22.f, 23.f, 24.f}}});

        EXPECT_EQ(4u, arr.width());
        EXPECT_EQ(3u, arr.height());
        EXPECT_EQ(2u, arr.depth());
        for (size_t i = 0; i < 24; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
        }
    }

    {
        Array3<float> arr(
            {{{ 1.f,  2.f,  3.f,  4.f},
              { 5.f,  6.f,  7.f,  8.f},
              { 9.f, 10.f, 11.f, 12.f}},
             {{13.f, 14.f, 15.f, 16.f},
              {17.f, 18.f, 19.f, 20.f},
              {21.f, 22.f, 23.f, 24.f}}});
        Array3<float> arr2(arr);

        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        EXPECT_EQ(2u, arr2.depth());
        for (size_t i = 0; i < 24; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
}

TEST(Array3, Clear) {
    Array3<float> arr(
        {{{ 1.f,  2.f,  3.f,  4.f},
          { 5.f,  6.f,  7.f,  8.f},
          { 9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});

    arr.clear();
    EXPECT_EQ(0u, arr.width());
    EXPECT_EQ(0u, arr.height());
    EXPECT_EQ(0u, arr.depth());
}

TEST(Array3, ResizeMethod) {
    {
        Array3<float> arr;
        arr.resize(Size3(2, 9, 5));
        EXPECT_EQ(2u, arr.width());
        EXPECT_EQ(9u, arr.height());
        EXPECT_EQ(5u, arr.depth());
        for (size_t i = 0; i < 90; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(Size3(8, 13, 7), 4.f);
        EXPECT_EQ(8u, arr.width());
        EXPECT_EQ(13u, arr.height());
        EXPECT_EQ(7u, arr.depth());
        for (size_t k = 0; k < 7; ++k) {
            for (size_t j = 0; j < 13; ++j) {
                for (size_t i = 0; i < 8; ++i) {
                    if (i < 2 && j < 9 && k < 5) {
                        EXPECT_FLOAT_EQ(0.f, arr(i, j, k));
                    } else {
                        EXPECT_FLOAT_EQ(4.f, arr(i, j, k));
                    }
                }
            }
        }
    }
    {
        Array3<float> arr;
        arr.resize(7, 6, 3);
        EXPECT_EQ(7u, arr.width());
        EXPECT_EQ(6u, arr.height());
        EXPECT_EQ(3u, arr.depth());
        for (size_t i = 0; i < 126; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(1, 9, 4, 3.f);
        EXPECT_EQ(1u, arr.width());
        EXPECT_EQ(9u, arr.height());
        EXPECT_EQ(4u, arr.depth());
        for (size_t k = 0; k < 4; ++k) {
            for (size_t j = 0; j < 9; ++j) {
                for (size_t i = 0; i < 1; ++i) {
                    if (j < 6 && k < 3) {
                        EXPECT_FLOAT_EQ(0.f, arr(i, j, k));
                    } else {
                        EXPECT_FLOAT_EQ(3.f, arr(i, j, k));
                    }
                }
            }
        }
    }
}

TEST(Array3, Iterators) {
    Array3<float> arr1(
        {{{ 1.f,  2.f,  3.f,  4.f},
          { 5.f,  6.f,  7.f,  8.f},
          { 9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});

    float cnt = 1.f;
    for (float& elem : arr1) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }

    cnt = 1.f;
    for (const float& elem : arr1) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }
}

TEST(Array3, ForEach) {
    Array3<float> arr1(
        {{{ 1.f,  2.f,  3.f,  4.f},
          { 5.f,  6.f,  7.f,  8.f},
          { 9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});

    size_t i = 0;
    arr1.forEach([&](float val) {
        EXPECT_FLOAT_EQ(arr1[i], val);
        ++i;
    });
}

TEST(Array3, ForEachIndex) {
    Array3<float> arr1(
        {{{ 1.f,  2.f,  3.f,  4.f},
          { 5.f,  6.f,  7.f,  8.f},
          { 9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});

    arr1.forEachIndex([&](size_t i, size_t j, size_t k) {
        size_t idx = i + (4 * (j + 3 * k)) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), arr1(i, j, k));
    });
}

TEST(Array3, ParallelForEach) {
    Array3<float> arr1(
        {{{ 1.f,  2.f,  3.f,  4.f},
          { 5.f,  6.f,  7.f,  8.f},
          { 9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});

    arr1.parallelForEach([&](float& val) {
        val *= 2.f;
    });

    arr1.forEachIndex([&](size_t i, size_t j, size_t k) {
        size_t idx = i + (4 * (j + 3 * k)) + 1;
        float ans = 2.f * static_cast<float>(idx);
        EXPECT_FLOAT_EQ(ans, arr1(i, j, k));
    });
}

TEST(Array3, ParallelForEachIndex) {
    Array3<float> arr1(
        {{{ 1.f,  2.f,  3.f,  4.f},
          { 5.f,  6.f,  7.f,  8.f},
          { 9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});

    arr1.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        size_t idx = i + (4 * (j + 3 * k)) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), arr1(i, j, k));
    });
}
