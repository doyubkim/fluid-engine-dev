// Copyright (c) 2017 Doyub Kim

#include <jet/apic_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ApicSolver2, UpdateEmpty) {
    ApicSolver2 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}
