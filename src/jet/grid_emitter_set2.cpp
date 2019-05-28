// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <vector>

#include <jet/grid_emitter_set2.h>

using namespace jet;

GridEmitterSet2::GridEmitterSet2() {}

GridEmitterSet2::GridEmitterSet2(const std::vector<GridEmitter2Ptr>& emitters) {
    for (const auto& e : emitters) {
        addEmitter(e);
    }
}

GridEmitterSet2::~GridEmitterSet2() {}

void GridEmitterSet2::addEmitter(const GridEmitter2Ptr& emitter) {
    _emitters.push_back(emitter);
}

void GridEmitterSet2::onUpdate(double currentTimeInSeconds,
                               double timeIntervalInSeconds) {
    if (!isEnabled()) {
        return;
    }

    for (auto& emitter : _emitters) {
        emitter->update(currentTimeInSeconds, timeIntervalInSeconds);
    }
}

GridEmitterSet2::Builder GridEmitterSet2::builder() { return Builder(); }

GridEmitterSet2::Builder& GridEmitterSet2::Builder::withEmitters(
    const std::vector<GridEmitter2Ptr>& emitters) {
    _emitters = emitters;
    return *this;
}

GridEmitterSet2 GridEmitterSet2::Builder::build() const {
    return GridEmitterSet2(_emitters);
}

GridEmitterSet2Ptr GridEmitterSet2::Builder::makeShared() const {
    return std::shared_ptr<GridEmitterSet2>(
        new GridEmitterSet2(_emitters),
        [](GridEmitterSet2* obj) { delete obj; });
}
