// Copyright (c) 2016 Doyub Kim

#include <jet/pic_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PicSolver2, UpdateEmpty) {
    PicSolver2 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}
