// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/grid_emitter3.h>

using namespace jet;

GridEmitter3::GridEmitter3() {
}

GridEmitter3::~GridEmitter3() {
}

void GridEmitter3::update(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(
            this, currentTimeInSeconds, timeIntervalInSeconds);
    }

    onUpdate(currentTimeInSeconds, timeIntervalInSeconds);
}

void GridEmitter3::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onBeginUpdateCallback = callback;
}

void GridEmitter3::callOnBeginUpdateCallback(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    _onBeginUpdateCallback(this, currentTimeInSeconds, timeIntervalInSeconds);
}
