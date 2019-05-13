// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <vector>

#include <jet/grid_emitter_set3.h>

using namespace jet;

GridEmitterSet3::GridEmitterSet3() {}

GridEmitterSet3::GridEmitterSet3(const std::vector<GridEmitter3Ptr>& emitters) {
    for (const auto& e : emitters) {
        addEmitter(e);
    }
}

GridEmitterSet3::~GridEmitterSet3() {}

void GridEmitterSet3::addEmitter(const GridEmitter3Ptr& emitter) {
    _emitters.push_back(emitter);
}

void GridEmitterSet3::onUpdate(double currentTimeInSeconds,
                               double timeIntervalInSeconds) {
    if (!isEnabled()) {
        return;
    }

    for (auto& emitter : _emitters) {
        emitter->update(currentTimeInSeconds, timeIntervalInSeconds);
    }
}

GridEmitterSet3::Builder GridEmitterSet3::builder() { return Builder(); }

GridEmitterSet3::Builder& GridEmitterSet3::Builder::withEmitters(
    const std::vector<GridEmitter3Ptr>& emitters) {
    _emitters = emitters;
    return *this;
}

GridEmitterSet3 GridEmitterSet3::Builder::build() const {
    return GridEmitterSet3(_emitters);
}

GridEmitterSet3Ptr GridEmitterSet3::Builder::makeShared() const {
    return std::shared_ptr<GridEmitterSet3>(
        new GridEmitterSet3(_emitters),
        [](GridEmitterSet3* obj) { delete obj; });
}
