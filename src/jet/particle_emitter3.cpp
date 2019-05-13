// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/parallel.h>
#include <jet/particle_emitter3.h>

#include <limits>

namespace jet {

ParticleEmitter3::ParticleEmitter3() {}

ParticleEmitter3::~ParticleEmitter3() {}

const ParticleSystemData3Ptr& ParticleEmitter3::target() const {
    return _particles;
}

void ParticleEmitter3::setTarget(const ParticleSystemData3Ptr& particles) {
    _particles = particles;

    onSetTarget(particles);
}

bool ParticleEmitter3::isEnabled() const { return _isEnabled; }

void ParticleEmitter3::setIsEnabled(bool enabled) { _isEnabled = enabled; }

void ParticleEmitter3::update(double currentTimeInSeconds,
                              double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(this, currentTimeInSeconds,
                               timeIntervalInSeconds);
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
