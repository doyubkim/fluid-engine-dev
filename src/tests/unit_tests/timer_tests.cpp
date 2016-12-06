// Copyright (c) 2016 Doyub Kim

#include <jet/timer.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <thread>

using namespace jet;

TEST(Timer, Basics) {
    Timer timer, timer2;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_LT(0.01, timer.durationInSeconds());

    timer.reset();
    EXPECT_LE(timer.durationInSeconds(), timer2.durationInSeconds());
}
