// Copyright (c) 2016 Doyub Kim

#include <jet/pic_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PicSolver3, UpdateEmpty) {
    // Empty solver test
    PicSolver3 solver;
    Frame frame;
    solver.update(frame);
    solver.update(frame);
}
