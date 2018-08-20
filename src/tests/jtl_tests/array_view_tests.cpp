// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_view.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(ConstArrayView, Constructors) {
    Array2<double> arr = {{1, 2}, {3, 4}, {5, 6}};
    ArrayView2<double> view = arr.view();

    // Copy from mutable Array
    ConstArrayView2<double> view2(arr);
    EXPECT_EQ(Vector2UZ(2, 3), view2.size());
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(arr(i, j), view2(i, j));
        }
    }

    // Copy from mutable ArrayView
    ConstArrayView2<double> view3(arr.view());
    EXPECT_EQ(Vector2UZ(2, 3), view3.size());
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(arr(i, j), view3(i, j));
        }
    }

    // Copy from immutable ArrayView
    ConstArrayView2<double> view4(view3);
    EXPECT_EQ(Vector2UZ(2, 3), view4.size());
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(arr(i, j), view4(i, j));
        }
    }
}
