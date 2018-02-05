// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_mg_linear_system2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(FdmMgUtils2, ResizeArrayWithFinest) {
    std::vector<Array2<double>> levels;
    FdmMgUtils2::resizeArrayWithFinest({100, 200}, 4, &levels);

    EXPECT_EQ(3u, levels.size());
    EXPECT_EQ(Size2(100, 200), levels[0].size());
    EXPECT_EQ(Size2(50, 100), levels[1].size());
    EXPECT_EQ(Size2(25, 50), levels[2].size());

    FdmMgUtils2::resizeArrayWithFinest({32, 16}, 6, &levels);
    EXPECT_EQ(5u, levels.size());
    EXPECT_EQ(Size2(32, 16), levels[0].size());
    EXPECT_EQ(Size2(16, 8), levels[1].size());
    EXPECT_EQ(Size2(8, 4), levels[2].size());
    EXPECT_EQ(Size2(4, 2), levels[3].size());
    EXPECT_EQ(Size2(2, 1), levels[4].size());

    FdmMgUtils2::resizeArrayWithFinest({16, 16}, 6, &levels);
    EXPECT_EQ(5u, levels.size());
}
