// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/flip_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(FlipSolver3, UpdateEmpty) {
    FlipSolver3 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}

TEST(FlipSolver3, PicBlendingFactor) {
    FlipSolver3 solver;

    solver.setPicBlendingFactor(0.3);
    EXPECT_EQ(0.3, solver.picBlendingFactor());

    solver.setPicBlendingFactor(2.4);
    EXPECT_EQ(1.0, solver.picBlendingFactor());

    solver.setPicBlendingFactor(-0.9);
    EXPECT_EQ(0.0, solver.picBlendingFactor());
}
