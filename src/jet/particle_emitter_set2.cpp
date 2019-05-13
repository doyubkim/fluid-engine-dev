// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <vector>

#include <jet/particle_emitter_set2.h>

using namespace jet;

ParticleEmitterSet2::ParticleEmitterSet2() {}

ParticleEmitterSet2::ParticleEmitterSet2(
    const std::vector<ParticleEmitter2Ptr>& emitters)
    : _emitters(emitters) {}

ParticleEmitterSet2::~ParticleEmitterSet2() {}

void ParticleEmitterSet2::addEmitter(const ParticleEmitter2Ptr& emitter) {
    _emitters.push_back(emitter);
}

void ParticleEmitterSet2::onSetTarget(const ParticleSystemData2Ptr& particles) {
    for (auto& emitter : _emitters) {
        emitter->setTarget(particles);
    }
}

void ParticleEmitterSet2::onUpdate(double currentTimeInSeconds,
                                   double timeIntervalInSeconds) {
    if (!isEnabled()) {
        return;
    }

    for (auto& emitter : _emitters) {
        emitter->update(currentTimeInSeconds, timeIntervalInSeconds);
    }
}

ParticleEmitterSet2::Builder ParticleEmitterSet2::builder() {
    return Builder();
}

ParticleEmitterSet2::Builder& ParticleEmitterSet2::Builder::withEmitters(
    const std::vector<ParticleEmitter2Ptr>& emitters) {
    _emitters = emitters;
    return *this;
}

ParticleEmitterSet2 ParticleEmitterSet2::Builder::build() const {
    return ParticleEmitterSet2(_emitters);
}

ParticleEmitterSet2Ptr ParticleEmitterSet2::Builder::makeShared() const {
    return std::shared_ptr<ParticleEmitterSet2>(
        new ParticleEmitterSet2(_emitters),
        [](ParticleEmitterSet2* obj) { delete obj; });
}
