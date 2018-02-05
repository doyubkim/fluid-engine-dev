// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/pic_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PicSolver2, UpdateEmpty) {
    PicSolver2 solver;

    for (Frame frame; frame.index < 2; ++frame) {
        solver.update(frame);
    }
}
