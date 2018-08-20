// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/std_utils.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(StdUtils, TakeFirstM) {
    std::array<int, 5> a{{1, 2, 3, 4, 5}};
    std::array<int, 3> b = takeFirstM<int, 5, 3>(a);

    EXPECT_EQ(a[0], b[0]);
    EXPECT_EQ(a[1], b[1]);
    EXPECT_EQ(a[2], b[2]);
}

TEST(StdUtils, TakeLastM) {
    std::array<int, 5> a{{1, 2, 3, 4, 5}};
    std::array<int, 3> b = takeLastM<int, 5, 3>(a);

    EXPECT_EQ(a[2], b[0]);
    EXPECT_EQ(a[3], b[1]);
    EXPECT_EQ(a[4], b[2]);
}
