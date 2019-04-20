// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

using namespace jet;

int main(int argc, char** argv) {
    ::testing::InitGoogleMock(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    std::ofstream logFile("unit_tests.log");
    if (logFile) {
        Logging::setAllStream(&logFile);
    }

    int ret = RUN_ALL_TESTS();

    return ret;
}
