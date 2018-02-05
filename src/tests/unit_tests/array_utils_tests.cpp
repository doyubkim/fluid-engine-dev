// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/array2.h>
#include <jet/array3.h>
#include <jet/array_utils.h>
#include <gtest/gtest.h>
#include <sstream>
#include <string>

using namespace jet;

TEST(ArrayUtils, SetRange1) {
    Array1<double> array0(5);

    setRange1(5, 3.4, &array0);

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(3.4, array0[i]);
    }

    setRange1(2, 4, 4.2, &array0);

    for (size_t i = 2; i < 4; ++i) {
        EXPECT_EQ(4.2, array0[i]);
    }
}

TEST(ArrayUtils, CopyRange1) {
    Array1<double> array0({1.0, 2.0, 3.0, 4.0, 5.0});
    Array1<double> array1(5);

    copyRange1(array0, 1, 3, &array1);

    for (size_t i = 1; i < 3; ++i) {
        EXPECT_EQ(array0[i], array1[i]);
    }

    copyRange1(array0, 5, &array1);

    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(array0[i], array1[i]);
    }
}

TEST(ArrayUtils, CopyRange2) {
    Array2<double> array0({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
    Array2<double> array1(2, 3);

    copyRange2(array0, 0, 1, 2, 3, &array1);

    for (size_t j = 2; j < 3; ++j) {
        for (size_t i = 0; i < 1; ++i) {
            EXPECT_EQ(array0(i, j), array1(i, j));
        }
    }

    copyRange2(array0, 2, 3, &array1);

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_EQ(array0(i, j), array1(i, j));
        }
    }
}

TEST(ArrayUtils, CopyRange3) {
    Array3<double> array0(
        {{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}},
         {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}});
    Array3<double> array1(2, 3, 2);

    copyRange3(array0, 0, 1, 2, 3, 1, 2, &array1);

    for (size_t k = 1; k < 2; ++k) {
        for (size_t j = 2; j < 3; ++j) {
            for (size_t i = 0; i < 1; ++i) {
                EXPECT_EQ(array0(i, j, k), array1(i, j, k));
            }
        }
    }

    copyRange3(array0, 2, 3, 2, &array1);

    for (size_t k = 0; k < 2; ++k) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 2; ++i) {
                EXPECT_EQ(array0(i, j, k), array1(i, j, k));
            }
        }
    }
}

TEST(ArrayUtils, ExtrapolateToRegion2) {
    Array2<double> data(10, 12, 0.0);
    Array2<char> valid(10, 12, 0);

    for (size_t j = 3; j < 10; ++j) {
        for (size_t i = 2; i < 6; ++i) {
            data(i, j) = static_cast<double>(i + j * 10.0);
            valid(i, j) = 1;
        }
    }

    extrapolateToRegion(
        data.constAccessor(), valid.constAccessor(), 6, data.accessor());

    Array2<double> dataAnswer(
        {{32.0, 32.0, 32.0, 33.0, 34.0, 35.0, 35.0, 35.0, 35.0,  0.0},
        {32.0, 32.0, 32.0, 33.0, 34.0, 35.0, 35.0, 35.0, 35.0, 35.0},
        {32.0, 32.0, 32.0, 33.0, 34.0, 35.0, 35.0, 35.0, 35.0, 35.0},
        {32.0, 32.0, 32.0, 33.0, 34.0, 35.0, 35.0, 35.0, 35.0, 35.0},
        {42.0, 42.0, 42.0, 43.0, 44.0, 45.0, 45.0, 45.0, 45.0, 45.0},
        {52.0, 52.0, 52.0, 53.0, 54.0, 55.0, 55.0, 55.0, 55.0, 55.0},
        {62.0, 62.0, 62.0, 63.0, 64.0, 65.0, 65.0, 65.0, 65.0, 65.0},
        {72.0, 72.0, 72.0, 73.0, 74.0, 75.0, 75.0, 75.0, 75.0, 75.0},
        {82.0, 82.0, 82.0, 83.0, 84.0, 85.0, 85.0, 85.0, 85.0, 85.0},
        {92.0, 92.0, 92.0, 93.0, 94.0, 95.0, 95.0, 95.0, 95.0, 95.0},
        {92.0, 92.0, 92.0, 93.0, 94.0, 95.0, 95.0, 95.0, 95.0, 95.0},
        {92.0, 92.0, 92.0, 93.0, 94.0, 95.0, 95.0, 95.0, 95.0, 95.0}
    });

    for (size_t j = 0; j < 12; ++j) {
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_DOUBLE_EQ(dataAnswer(i, j), data(i, j));
        }
    }
}

TEST(ArrayUtils, ExtrapolateToRegion3) {
    // TODO: Need better testing

    Array3<double> data(3, 4, 5, 0.0);
    Array3<char> valid(3, 4, 5, 0);

    for (size_t k = 1; k < 4; ++k) {
        for (size_t j = 2; j < 3; ++j) {
            for (size_t i = 1; i < 2; ++i) {
                data(i, j, k) = 42.0;
                valid(i, j, k) = 1;
            }
        }
    }

    extrapolateToRegion(
        data.constAccessor(), valid.constAccessor(), 5, data.accessor());

    for (size_t k = 0; k < 5; ++k) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                EXPECT_DOUBLE_EQ(42.0, data(i, j, k));
            }
        }
    }
}
