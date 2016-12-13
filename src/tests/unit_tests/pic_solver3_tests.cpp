// Copyright (c) 2016 Doyub Kim

#include <jet/pic_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PicSolver3, UpdateEmpty) {
    PicSolver3 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}
