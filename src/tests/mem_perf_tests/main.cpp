// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>
#include <gtest/gtest.h>
#include <fstream>

using namespace jet;

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::ofstream logFile("perf_tests.log");
    if (logFile) {
        Logging::setAllStream(&logFile);
    }

    int ret = RUN_ALL_TESTS();

    return ret;
}
