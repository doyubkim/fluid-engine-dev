// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/timer.h>

using namespace jet;

Timer::Timer() {
    _startingPoint = _clock.now();
}

double Timer::durationInSeconds() const {
    auto end = std::chrono::steady_clock::now();
    auto count = std::chrono::duration_cast<std::chrono::microseconds>(
        end - _startingPoint).count();
    return count / 1000000.0;
}

void Timer::reset() {
    _startingPoint = _clock.now();
}
