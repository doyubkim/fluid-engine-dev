// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/constants.h>
#include <jet/physics_animation.h>
#include <jet/timer.h>
#include <limits>

using namespace jet;

PhysicsAnimation::PhysicsAnimation() {
}

PhysicsAnimation::~PhysicsAnimation() {
}

bool PhysicsAnimation::isUsingFixedSubTimeSteps() const {
    return _isUsingFixedSubTimeSteps;
}

void PhysicsAnimation::setIsUsingFixedSubTimeSteps(bool isUsing) {
    _isUsingFixedSubTimeSteps = isUsing;
}

unsigned int PhysicsAnimation::numberOfFixedSubTimeSteps() const {
    return _numberOfFixedSubTimeSteps;
}

void PhysicsAnimation::setNumberOfFixedSubTimeSteps(
    unsigned int numberOfSteps) {
    _numberOfFixedSubTimeSteps = numberOfSteps;
}

unsigned int PhysicsAnimation::numberOfSubTimeSteps(
    double timeIntervalInSeconds) const {
    UNUSED_VARIABLE(timeIntervalInSeconds);

    // Returns number of fixed sub-timesteps by default
    return _numberOfFixedSubTimeSteps;
}

void PhysicsAnimation::onUpdate(const Frame& frame) {
    if (frame.index > _currentFrame.index) {
        unsigned int numberOfFrames = frame.index - _currentFrame.index;

        for (unsigned int i = 0; i < numberOfFrames; ++i) {
            advanceTimeStep(frame.timeIntervalInSeconds);
        }

        _currentFrame = frame;
    }
}

void PhysicsAnimation::advanceTimeStep(double timeIntervalInSeconds) {
    if (_isUsingFixedSubTimeSteps) {
        JET_INFO << "Using fixed sub-timesteps: " << _numberOfFixedSubTimeSteps;

        // Perform fixed time-stepping
        const double actualTimeInterval
            = timeIntervalInSeconds
            / static_cast<double>(_numberOfFixedSubTimeSteps);

        for (unsigned int i = 0; i < _numberOfFixedSubTimeSteps; ++i) {
            JET_INFO << "Begin onAdvanceTimeStep: " << actualTimeInterval
                     << " (1/" << 1.0 / actualTimeInterval
                     << ") seconds";

            Timer timer;
            onAdvanceTimeStep(actualTimeInterval);

            JET_INFO << "End onAdvanceTimeStep (took "
                     << timer.durationInSeconds()
                     << " seconds)";
        }
    } else {
        JET_INFO << "Using adaptive sub-timesteps";

        // Perform adaptive time-stepping
        double remainingTime = timeIntervalInSeconds;
        while (remainingTime > kEpsilonD) {
            unsigned int numSteps = numberOfSubTimeSteps(remainingTime);
            double actualTimeInterval
                = remainingTime / static_cast<double>(numSteps);

            JET_INFO << "Number of remaining sub-timesteps: " << numSteps;

            JET_INFO << "Begin onAdvanceTimeStep: " << actualTimeInterval
                     << " (1/" << 1.0 / actualTimeInterval
                     << ") seconds";

            Timer timer;
            onAdvanceTimeStep(actualTimeInterval);

            JET_INFO << "End onAdvanceTimeStep (took "
                     << timer.durationInSeconds()
                     << " seconds)";

            remainingTime -= actualTimeInterval;
        }
    }
}
