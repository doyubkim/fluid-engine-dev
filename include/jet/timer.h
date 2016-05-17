// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_TIMER_H_
#define INCLUDE_JET_TIMER_H_

#include <chrono>

namespace jet {

class Timer {
 public:
    Timer();

    double durationInSeconds() const;

    void reset();

 private:
    std::chrono::steady_clock _clock;
    std::chrono::steady_clock::time_point _startingPoint;
};

}  // namespace jet

#endif  // INCLUDE_JET_TIMER_H_
