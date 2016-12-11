// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/grid_emitter2.h>

using namespace jet;

GridEmitter2::GridEmitter2() {
}

GridEmitter2::~GridEmitter2() {
}

void GridEmitter2::update(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(
            this, currentTimeInSeconds, timeIntervalInSeconds);
    }

    onUpdate(currentTimeInSeconds, timeIntervalInSeconds);
}

void GridEmitter2::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onBeginUpdateCallback = callback;
}

void GridEmitter2::callOnBeginUpdateCallback(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    _onBeginUpdateCallback(this, currentTimeInSeconds, timeIntervalInSeconds);
}
