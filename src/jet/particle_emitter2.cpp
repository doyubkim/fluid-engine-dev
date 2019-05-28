// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/parallel.h>
#include <jet/particle_emitter2.h>

#include <limits>

namespace jet {

ParticleEmitter2::ParticleEmitter2() {}

ParticleEmitter2::~ParticleEmitter2() {}

const ParticleSystemData2Ptr& ParticleEmitter2::target() const {
    return _particles;
}

void ParticleEmitter2::setTarget(const ParticleSystemData2Ptr& particles) {
    _particles = particles;

    onSetTarget(particles);
}

bool ParticleEmitter2::isEnabled() const { return _isEnabled; }

void ParticleEmitter2::setIsEnabled(bool enabled) { _isEnabled = enabled; }

void ParticleEmitter2::update(double currentTimeInSeconds,
                              double timeIntervalInSeconds) {
    if (_onBeginUpdateCallback) {
        _onBeginUpdateCallback(this, currentTimeInSeconds,
                               timeIntervalInSeconds);
    }

    onUpdate(currentTimeInSeconds, timeIntervalInSeconds);
}

void ParticleEmitter2::onSetTarget(const ParticleSystemData2Ptr& particles) {
    UNUSED_VARIABLE(particles);
}

void ParticleEmitter2::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onBeginUpdateCallback = callback;
}

}  // namespace jet
