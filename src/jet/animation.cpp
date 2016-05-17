// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/animation.h>
#include <jet/timer.h>

using namespace jet;

Frame::Frame() {
}

Frame::Frame(double newTimeIntervalInSeconds)
    : timeIntervalInSeconds(newTimeIntervalInSeconds) {
}

double Frame::timeInSeconds() const {
    return index * timeIntervalInSeconds;
}

void Frame::advance() {
    ++index;
}

void Frame::advance(unsigned int delta) {
    index += delta;
}

Animation::Animation() {
}

Animation::~Animation() {
}

void Animation::update(const Frame& frame) {
    Timer timer;

    JET_INFO << "Begin updating frame: " << frame.index
             << " timeIntervalInSeconds: " << frame.timeIntervalInSeconds
             << " (1/" << 1.0 / frame.timeIntervalInSeconds
             << ") seconds";

    onUpdate(frame);

    JET_INFO << "End updating frame (took " << timer.durationInSeconds()
             << " seconds)";
}
