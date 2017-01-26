// Copyright (c) 2017 Doyub Kim

#include <jet/apic_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ApicSolver3, UpdateEmpty) {
    ApicSolver3 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}
