// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/grid_emitter3.h>

using namespace jet;

GridEmitter3::GridEmitter3() {}

GridEmitter3::~GridEmitter3() {}

void GridEmitter3::update(double currentTimeInSeconds,
                          double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(currentTimeInSeconds, timeIntervalInSeconds);
    }

    onUpdate(currentTimeInSeconds, timeIntervalInSeconds);
}

void GridEmitter3::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onBeginUpdateCallback = callback;
}
