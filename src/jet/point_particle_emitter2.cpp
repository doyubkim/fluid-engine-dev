// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/matrix2x2.h>
#include <jet/point_particle_emitter2.h>
#include <jet/samplers.h>

namespace jet {

PointParticleEmitter2::PointParticleEmitter2(
    const Vector2D& origin,
    const Vector2D& direction,
    double speed,
    double spreadAngleInDegrees,
    size_t maxNumOfNewParticlesPerSec,
    size_t maxNumOfParticles,
    uint32_t seed) :
    _rng(seed),
    _maxNumberOfNewParticlesPerSecond(maxNumOfNewParticlesPerSec),
    _maxNumberOfParticles(maxNumOfParticles),
    _origin(origin),
    _direction(direction),
    _speed(speed),
    _spreadAngleInRadians(degreesToRadians(spreadAngleInDegrees)) {
}

size_t PointParticleEmitter2::maxNumberOfNewParticlesPerSecond() const {
    return _maxNumberOfNewParticlesPerSecond;
}

void PointParticleEmitter2::setMaxNumberOfNewParticlesPerSecond(size_t rate) {
    _maxNumberOfNewParticlesPerSecond = rate;
}

size_t PointParticleEmitter2::maxNumberOfParticles() const {
    return _maxNumberOfParticles;
}

void PointParticleEmitter2::setMaxNumberOfParticles(
    size_t maxNumberOfParticles) {
    _maxNumberOfParticles = maxNumberOfParticles;
}

void PointParticleEmitter2::emit(
    const Frame& frame,
    const ParticleSystemData2Ptr& particles) {
    if (_numberOfEmittedParticles == 0) {
        _firstFrameTimeInSeconds = frame.timeInSeconds();
    }

    double elapsedTimeInSeconds = frame.timeInSeconds()
        - _firstFrameTimeInSeconds;

    size_t newMaxTotalNumberOfEmittedParticles = static_cast<size_t>(
        std::ceil((elapsedTimeInSeconds + frame.timeIntervalInSeconds)
            * _maxNumberOfNewParticlesPerSecond));
    newMaxTotalNumberOfEmittedParticles = std::min(
        newMaxTotalNumberOfEmittedParticles,
        _maxNumberOfParticles);
    size_t maxNumberOfNewParticles = newMaxTotalNumberOfEmittedParticles
        - _numberOfEmittedParticles;

    if (maxNumberOfNewParticles > 0) {
        Array1<Vector2D> candidatePositions;
        Array1<Vector2D> candidateVelocities;
        Array1<Vector2D> newPositions;
        Array1<Vector2D> newVelocities;

        emit(
            &candidatePositions,
            &candidateVelocities,
            maxNumberOfNewParticles);

        newPositions.append(candidatePositions);
        newVelocities.append(candidateVelocities);

        particles->addParticles(newPositions, newVelocities);

        _numberOfEmittedParticles += newPositions.size();
    }
}

void PointParticleEmitter2::emit(
    Array1<Vector2D>* newPositions,
    Array1<Vector2D>* newVelocities,
    size_t maxNewNumberOfParticles) {
    for (size_t i = 0; i < maxNewNumberOfParticles; ++i) {
        double newAngleInRadian = (random() - 0.5) * _spreadAngleInRadians;
        Matrix2x2D rotationMatrix =
            Matrix2x2D::makeRotationMatrix(newAngleInRadian);

        newPositions->append(_origin);
        newVelocities->append(_speed * (rotationMatrix * _direction));
    }
}

double PointParticleEmitter2::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
}

}  // namespace jet
