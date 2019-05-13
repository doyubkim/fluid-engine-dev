// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <vector>

#include <jet/particle_emitter_set3.h>

using namespace jet;

ParticleEmitterSet3::ParticleEmitterSet3() {}

ParticleEmitterSet3::ParticleEmitterSet3(
    const std::vector<ParticleEmitter3Ptr>& emitters)
    : _emitters(emitters) {}

ParticleEmitterSet3::~ParticleEmitterSet3() {}

void ParticleEmitterSet3::addEmitter(const ParticleEmitter3Ptr& emitter) {
    _emitters.push_back(emitter);
}

void ParticleEmitterSet3::onSetTarget(const ParticleSystemData3Ptr& particles) {
    for (auto& emitter : _emitters) {
        emitter->setTarget(particles);
    }
}

void ParticleEmitterSet3::onUpdate(double currentTimeInSeconds,
                                   double timeIntervalInSeconds) {
    if (!isEnabled()) {
        return;
    }

    for (auto& emitter : _emitters) {
        emitter->update(currentTimeInSeconds, timeIntervalInSeconds);
    }
}

ParticleEmitterSet3::Builder ParticleEmitterSet3::builder() {
    return Builder();
}

ParticleEmitterSet3::Builder& ParticleEmitterSet3::Builder::withEmitters(
    const std::vector<ParticleEmitter3Ptr>& emitters) {
    _emitters = emitters;
    return *this;
}

ParticleEmitterSet3 ParticleEmitterSet3::Builder::build() const {
    return ParticleEmitterSet3(_emitters);
}

ParticleEmitterSet3Ptr ParticleEmitterSet3::Builder::makeShared() const {
    return std::shared_ptr<ParticleEmitterSet3>(
        new ParticleEmitterSet3(_emitters),
        [](ParticleEmitterSet3* obj) { delete obj; });
}
