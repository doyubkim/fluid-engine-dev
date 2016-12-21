// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/matrix2x2.h>
#include <jet/point_hash_grid_searcher2.h>
#include <jet/samplers.h>
#include <jet/surface_to_implicit2.h>
#include <jet/triangle_point_generator.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;

static const size_t kDefaultHashGridResolution = 64;

VolumeParticleEmitter2::VolumeParticleEmitter2(
    const ImplicitSurface2Ptr& implicitSurface,
    const BoundingBox2D& bounds,
    double spacing,
    const Vector2D& initialVel,
    size_t maxNumberOfParticles,
    double jitter,
    bool isOneShot,
    bool allowOverlapping,
    uint32_t seed) :
    _rng(seed),
    _implicitSurface(implicitSurface),
    _bounds(bounds),
    _spacing(spacing),
    _initialVel(initialVel),
    _maxNumberOfParticles(maxNumberOfParticles),
    _jitter(jitter),
    _isOneShot(isOneShot),
    _allowOverlapping(allowOverlapping) {
    _pointsGen = std::make_shared<TrianglePointGenerator>();
}

void VolumeParticleEmitter2::onUpdate(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    UNUSED_VARIABLE(currentTimeInSeconds);
    UNUSED_VARIABLE(timeIntervalInSeconds);

    auto particles = target();

    if (particles == nullptr) {
        return;
    }

    if (_numberOfEmittedParticles > 0 && _isOneShot) {
        return;
    }

    Array1<Vector2D> newPositions;
    Array1<Vector2D> newVelocities;

    emit(particles, &newPositions, &newVelocities);

    particles->addParticles(newPositions, newVelocities);
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
                    if (_numberOfEmittedParticles < _maxNumberOfParticles) {
                        newPositions->append(candidate);
                        ++_numberOfEmittedParticles;
                    } else {
                        return false;
                    }
                }

                return true;
            });
    } else {
        // Use serial hash grid searcher for continuous update.
        PointHashGridSearcher2 neighborSearcher(
            Size2(
                kDefaultHashGridResolution,
                kDefaultHashGridResolution),
            2.0 * _spacing);
        if (!_allowOverlapping) {
            neighborSearcher.build(particles->positions());
        }

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
                if (_implicitSurface->signedDistance(candidate) <= 0.0 &&
                    (!_allowOverlapping &&
                     !neighborSearcher.hasNearbyPoint(candidate, _spacing))) {
                    if (_numberOfEmittedParticles < _maxNumberOfParticles) {
                        newPositions->append(candidate);
                        neighborSearcher.add(candidate);
                        ++_numberOfEmittedParticles;
                    } else {
                        return false;
                    }
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

bool VolumeParticleEmitter2::isOneShot() const {
    return _isOneShot;
}

void VolumeParticleEmitter2::setIsOneShot(bool newValue) {
    _isOneShot = newValue;
}

bool VolumeParticleEmitter2::allowOverlapping() const {
    return _allowOverlapping;
}

void VolumeParticleEmitter2::setAllowOverlapping(bool newValue) {
    _allowOverlapping = newValue;
}

size_t VolumeParticleEmitter2::maxNumberOfParticles() const {
    return _maxNumberOfParticles;
}

void VolumeParticleEmitter2::setMaxNumberOfParticles(
    size_t newMaxNumberOfParticles) {
    _maxNumberOfParticles = newMaxNumberOfParticles;
}

double VolumeParticleEmitter2::spacing() const {
    return _spacing;
}

void VolumeParticleEmitter2::setSpacing(double newSpacing) {
    _spacing = newSpacing;
}

Vector2D VolumeParticleEmitter2::initialVelocity() const {
    return _initialVel;
}

void VolumeParticleEmitter2::setInitialVelocity(const Vector2D& newInitialVel) {
    _initialVel = newInitialVel;
}

double VolumeParticleEmitter2::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
}

VolumeParticleEmitter2::Builder VolumeParticleEmitter2::builder() {
    return Builder();
}


VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withImplicitSurface(
    const ImplicitSurface2Ptr& implicitSurface) {
    _implicitSurface = implicitSurface;
    if (!_isBoundSet) {
        _bounds = _implicitSurface->boundingBox();
    }
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withSurface(
    const Surface2Ptr& surface) {
    _implicitSurface = std::make_shared<SurfaceToImplicit2>(surface);
    if (!_isBoundSet) {
        _bounds = surface->boundingBox();
    }
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withMaxRegion(const BoundingBox2D& bounds) {
    _bounds = bounds;
    _isBoundSet = true;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withSpacing(double spacing) {
    _spacing = spacing;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withInitialVelocity(
    const Vector2D& initialVel) {
    _initialVel = initialVel;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withMaxNumberOfParticles(
    size_t maxNumberOfParticles) {
    _maxNumberOfParticles = maxNumberOfParticles;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withJitter(double jitter) {
    _jitter = jitter;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withIsOneShot(bool isOneShot) {
    _isOneShot = isOneShot;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withAllowOverlapping(bool allowOverlapping) {
    _allowOverlapping = allowOverlapping;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withRandomSeed(uint32_t seed) {
    _seed = seed;
    return *this;
}

VolumeParticleEmitter2 VolumeParticleEmitter2::Builder::build() const {
    return VolumeParticleEmitter2(
        _implicitSurface,
        _bounds,
        _spacing,
        _initialVel,
        _maxNumberOfParticles,
        _jitter,
        _isOneShot,
        _allowOverlapping,
        _seed);
}

VolumeParticleEmitter2Ptr VolumeParticleEmitter2::Builder::makeShared() const {
    return std::shared_ptr<VolumeParticleEmitter2>(
        new VolumeParticleEmitter2(
            _implicitSurface,
            _bounds,
            _spacing,
            _initialVel,
            _maxNumberOfParticles,
            _jitter,
            _isOneShot,
            _allowOverlapping),
        [] (VolumeParticleEmitter2* obj) {
            delete obj;
        });
}
