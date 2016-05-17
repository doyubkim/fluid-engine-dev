// Copyright (c) 2016 Doyub Kim

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
