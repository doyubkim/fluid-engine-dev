// Copyright (c) 2016 Doyub Kim

#include <pch.h>

#include <jet/parallel.h>
#include <jet/particle_emitter3.h>

#include <limits>

namespace jet {

ParticleEmitter3::ParticleEmitter3() {
}

ParticleEmitter3::~ParticleEmitter3() {
}

const ParticleSystemData3Ptr& ParticleEmitter3::target() const {
    return _particles;
}

void ParticleEmitter3::setTarget(const ParticleSystemData3Ptr& particles) {
    _particles = particles;

    onSetTarget(particles);
}

void ParticleEmitter3::update(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(
            this, currentTimeInSeconds, timeIntervalInSeconds);
    }

    onUpdate(currentTimeInSeconds, timeIntervalInSeconds);
}

void ParticleEmitter3::onSetTarget(const ParticleSystemData3Ptr& particles) {
    UNUSED_VARIABLE(particles);
}

void ParticleEmitter3::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onBeginUpdateCallback = callback;
}

}  // namespace jet
