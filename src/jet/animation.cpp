// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/animation.h>
#include <jet/timer.h>

#include "./private_helpers.h"

using namespace jet;

Frame::Frame() {
}

Frame::Frame(unsigned int newIndex, double newTimeIntervalInSeconds)
    : index(newIndex)
    , timeIntervalInSeconds(newTimeIntervalInSeconds) {
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

Frame& Frame::operator++() {
    advance();
    return *this;
}

Frame Frame::operator++(int i) {
    UNUSED_VARIABLE(i);

    Frame result = *this;
    advance();
    return result;
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
