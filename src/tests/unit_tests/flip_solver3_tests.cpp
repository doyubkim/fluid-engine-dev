// Copyright (c) 2016 Doyub Kim

#include <jet/flip_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(FlipSolver3, UpdateEmpty) {
    // Empty solver test
    FlipSolver3 solver;
    Frame frame;
    solver.update(frame);
    solver.update(frame);
}
