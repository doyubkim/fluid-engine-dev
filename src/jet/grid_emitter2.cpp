// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/grid_emitter2.h>

using namespace jet;

GridEmitter2::GridEmitter2() {}

GridEmitter2::~GridEmitter2() {}

void GridEmitter2::update(double currentTimeInSeconds,
                          double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(currentTimeInSeconds, timeIntervalInSeconds);
    }

    onUpdate(currentTimeInSeconds, timeIntervalInSeconds);
}

void GridEmitter2::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onBeginUpdateCallback = callback;
}
