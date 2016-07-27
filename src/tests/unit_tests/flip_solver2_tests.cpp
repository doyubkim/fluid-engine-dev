// Copyright (c) 2016 Doyub Kim

#include <jet/flip_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(FlipSolver2, UpdateEmpty) {
    // Empty solver test
    FlipSolver2 solver;
    Frame frame;
    solver.update(frame);
    solver.update(frame);
}
