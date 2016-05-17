// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/bcc_lattice_points_generator.h>
#include <jet/point_hash_grid_searcher3.h>
#include <jet/samplers.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

VolumeParticleEmitter3::VolumeParticleEmitter3(
    const ImplicitSurface3Ptr& implicitSurface,
    const BoundingBox3D& bounds,
    double spacing,
    const Vector3D& initialVel,
    uint32_t seed) :
    _rng(seed),
    _implicitSurface(implicitSurface),
    _bounds(bounds),
    _spacing(spacing),
    _initialVel(initialVel) {
    _pointsGen = std::make_shared<BccLatticePointsGenerator>();
}

void VolumeParticleEmitter3::emit(
    const Frame& frame,
    const ParticleSystemData3Ptr& particles) {
    UNUSED_VARIABLE(frame);

    if (_numberOfEmittedParticles > 0 && _isOneShot) {
        return;
    }

    Array1<Vector3D> candidatePositions;
    Array1<Vector3D> candidateVelocities;
    Array1<Vector3D> newPositions;
    Array1<Vector3D> newVelocities;

    emit(particles, &candidatePositions, &candidateVelocities);

    size_t expectedSize = _numberOfEmittedParticles + candidatePositions.size();

    if (expectedSize < _maxNumberOfParticles) {
        newPositions.append(candidatePositions);
        newVelocities.append(candidateVelocities);

        particles->addParticles(newPositions, newVelocities);

        _numberOfEmittedParticles += newPositions.size();
    }
}

void VolumeParticleEmitter3::emit(
    const ParticleSystemData3Ptr& particles,
    Array1<Vector3D>* newPositions,
    Array1<Vector3D>* newVelocities) {
    // Reserving more space for jittering
    const double j = jitter();
    const double maxJitterDist = 0.5 * j * _spacing;

    if (_allowOverlapping || _isOneShot) {
        _pointsGen->forEachPoint(
            _bounds,
            _spacing,
            [&] (const Vector3D& point) {
                Vector3D randomDir = uniformSampleSphere(
                    random(),
                    random());
                Vector3D offset = maxJitterDist * randomDir;
                Vector3D candidate = point + offset;
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
            [&] (const Vector3D& point) {
                Vector3D randomDir = uniformSampleSphere(
                    random(),
                    random());
                Vector3D offset = maxJitterDist * randomDir;
                Vector3D candidate = point + offset;
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

void VolumeParticleEmitter3::setPointGenerator(
    const PointsGenerator3Ptr& newPointsGen) {
    _pointsGen = newPointsGen;
}

double VolumeParticleEmitter3::jitter() const {
    return _jitter;
}

void VolumeParticleEmitter3::setJitter(double newJitter) {
    _jitter = clamp(newJitter, 0.0, 1.0);
}

void VolumeParticleEmitter3::setIsOneShot(bool newValue) {
    _isOneShot = newValue;
}

void VolumeParticleEmitter3::setAllowOverlapping(bool newValue) {
    _allowOverlapping = newValue;
}

double VolumeParticleEmitter3::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
}
