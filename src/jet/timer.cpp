// Copyright (c) 2016 Doyub Kim

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
