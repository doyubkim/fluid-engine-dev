// Copyright (c) 2016 Doyub Kim

#include <jet/pic_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PicSolver2, UpdateEmpty) {
    // Empty solver test
    PicSolver2 solver;
    Frame frame;
    solver.update(frame);
    solver.update(frame);
}
