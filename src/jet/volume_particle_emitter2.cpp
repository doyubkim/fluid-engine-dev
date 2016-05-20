// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/matrix2x2.h>
#include <jet/point_hash_grid_searcher2.h>
#include <jet/samplers.h>
#include <jet/triangle_point_generator.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;

VolumeParticleEmitter2::VolumeParticleEmitter2(
    const ImplicitSurface2Ptr& implicitSurface,
    const BoundingBox2D& bounds,
    double spacing,
    const Vector2D& initialVel,
    uint32_t seed) :
    _rng(seed),
    _implicitSurface(implicitSurface),
    _bounds(bounds),
    _spacing(spacing),
    _initialVel(initialVel) {
    _pointsGen = std::make_shared<TrianglePointGenerator>();
}

void VolumeParticleEmitter2::emit(
    const Frame& frame,
    const ParticleSystemData2Ptr& particles) {
    UNUSED_VARIABLE(frame);

    if (_numberOfEmittedParticles > 0 && _isOneShot) {
        return;
    }

    Array1<Vector2D> candidatePositions;
    Array1<Vector2D> candidateVelocities;
    Array1<Vector2D> newPositions;
    Array1<Vector2D> newVelocities;

    emit(particles, &candidatePositions, &candidateVelocities);

    size_t expectedSize = _numberOfEmittedParticles + candidatePositions.size();

    if (expectedSize < _maxNumberOfParticles) {
        newPositions.append(candidatePositions);
        newVelocities.append(candidateVelocities);

        particles->addParticles(newPositions, newVelocities);

        _numberOfEmittedParticles += newPositions.size();
    }
}

void VolumeParticleEmitter2::emit(
    const ParticleSystemData2Ptr& particles,
    Array1<Vector2D>* newPositions,
    Array1<Vector2D>* newVelocities) {
    // Reserving more space for jittering
    const double j = jitter();
    const double maxJitterDist = 0.5 * j * _spacing;

    if (_allowOverlapping || _isOneShot) {
        _pointsGen->forEachPoint(
            _bounds,
            _spacing,
            [&] (const Vector2D& point) {
                double newAngleInRadian = (random() - 0.5) * kTwoPiD;
                Matrix2x2D rotationMatrix =
                    Matrix2x2D::makeRotationMatrix(newAngleInRadian);
                Vector2D randomDir = rotationMatrix * Vector2D();
                Vector2D offset = maxJitterDist * randomDir;
                Vector2D candidate = point + offset;
                if (_implicitSurface->signedDistance(candidate) <= 0.0) {
                    newPositions->append(candidate);
                }

                return true;
            });
    } else {
        particles->buildNeighborSearcher(_spacing);
        auto neighborSearcher = particles->neighborSearcher();

        _pointsGen->forEachPoint(
            _bounds,
            _spacing,
            [&] (const Vector2D& point) {
                double newAngleInRadian = (random() - 0.5) * kTwoPiD;
                Matrix2x2D rotationMatrix =
                    Matrix2x2D::makeRotationMatrix(newAngleInRadian);
                Vector2D randomDir = rotationMatrix * Vector2D();
                Vector2D offset = maxJitterDist * randomDir;
                Vector2D candidate = point + offset;
                if (_implicitSurface->signedDistance(candidate) <= 0.0
                    && !neighborSearcher->hasNearbyPoint(point, _spacing)) {
                    newPositions->append(candidate);
                }

                return true;
            });
    }

    newVelocities->resize(newPositions->size());
    newVelocities->set(_initialVel);
}

void VolumeParticleEmitter2::setPointGenerator(
    const PointGenerator2Ptr& newPointsGen) {
    _pointsGen = newPointsGen;
}

double VolumeParticleEmitter2::jitter() const {
    return _jitter;
}

void VolumeParticleEmitter2::setJitter(double newJitter) {
    _jitter = clamp(newJitter, 0.0, 1.0);
}

void VolumeParticleEmitter2::setIsOneShot(bool newValue) {
    _isOneShot = newValue;
}

void VolumeParticleEmitter2::setAllowOverlapping(bool newValue) {
    _allowOverlapping = newValue;
}

double VolumeParticleEmitter2::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
}
