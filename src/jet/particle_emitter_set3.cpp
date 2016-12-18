// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/particle_emitter_set3.h>
#include <vector>

using namespace jet;

ParticleEmitterSet3::ParticleEmitterSet3() {
}

ParticleEmitterSet3::ParticleEmitterSet3(
    const std::vector<ParticleEmitter3Ptr>& emitters)
: _emitters(emitters) {
}

ParticleEmitterSet3::~ParticleEmitterSet3() {
}

void ParticleEmitterSet3::addEmitter(const ParticleEmitter3Ptr& emitter) {
    _emitters.push_back(emitter);
}

void ParticleEmitterSet3::onSetTarget(const ParticleSystemData3Ptr& particles) {
    for (auto& emitter : _emitters) {
        emitter->setTarget(particles);
    }
}

void ParticleEmitterSet3::onUpdate(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    for (auto& emitter : _emitters) {
        emitter->update(currentTimeInSeconds, timeIntervalInSeconds);
    }
}

ParticleEmitterSet3::Builder ParticleEmitterSet3::builder() {
    return Builder();
}


ParticleEmitterSet3::Builder&
ParticleEmitterSet3::Builder::withEmitters(
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
        [] (ParticleEmitterSet3* obj) {
            delete obj;
        });
}
