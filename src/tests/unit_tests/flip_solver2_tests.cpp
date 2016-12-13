// Copyright (c) 2016 Doyub Kim

#include <jet/flip_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(FlipSolver2, Empty) {
    FlipSolver2 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}
