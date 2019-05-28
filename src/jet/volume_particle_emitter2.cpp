// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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
    const ImplicitSurface2Ptr& implicitSurface, const BoundingBox2D& maxRegion,
    double spacing, const Vector2D& initialVel, const Vector2D& linearVel,
    double angularVel, size_t maxNumberOfParticles, double jitter,
    bool isOneShot, bool allowOverlapping, uint32_t seed)
    : _rng(seed),
      _implicitSurface(implicitSurface),
      _bounds(maxRegion),
      _spacing(spacing),
      _initialVel(initialVel),
      _linearVel(linearVel),
      _angularVel(angularVel),
      _maxNumberOfParticles(maxNumberOfParticles),
      _jitter(jitter),
      _isOneShot(isOneShot),
      _allowOverlapping(allowOverlapping) {
    _pointsGen = std::make_shared<TrianglePointGenerator>();
}

void VolumeParticleEmitter2::onUpdate(double currentTimeInSeconds,
                                      double timeIntervalInSeconds) {
    UNUSED_VARIABLE(currentTimeInSeconds);
    UNUSED_VARIABLE(timeIntervalInSeconds);

    auto particles = target();

    if (particles == nullptr) {
        return;
    }

    if (!isEnabled()) {
        return;
    }

    Array1<Vector2D> newPositions;
    Array1<Vector2D> newVelocities;

    emit(particles, &newPositions, &newVelocities);

    particles->addParticles(newPositions, newVelocities);

    if (_isOneShot) {
        setIsEnabled(false);
    }
}

void VolumeParticleEmitter2::emit(const ParticleSystemData2Ptr& particles,
                                  Array1<Vector2D>* newPositions,
                                  Array1<Vector2D>* newVelocities) {
    if (!_implicitSurface) {
        return;
    }

    _implicitSurface->updateQueryEngine();

    BoundingBox2D region = _bounds;
    if (_implicitSurface->isBounded()) {
        BoundingBox2D surfaceBBox = _implicitSurface->boundingBox();
        region.lowerCorner = max(region.lowerCorner, surfaceBBox.lowerCorner);
        region.upperCorner = min(region.upperCorner, surfaceBBox.upperCorner);
    }

    // Reserving more space for jittering
    const double j = jitter();
    const double maxJitterDist = 0.5 * j * _spacing;
    size_t numNewParticles = 0;

    if (_allowOverlapping || _isOneShot) {
        _pointsGen->forEachPoint(region, _spacing, [&](const Vector2D& point) {
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
                    ++numNewParticles;
                } else {
                    return false;
                }
            }

            return true;
        });
    } else {
        // Use serial hash grid searcher for continuous update.
        PointHashGridSearcher2 neighborSearcher(
            Size2(kDefaultHashGridResolution, kDefaultHashGridResolution),
            2.0 * _spacing);
        if (!_allowOverlapping) {
            neighborSearcher.build(particles->positions());
        }

        _pointsGen->forEachPoint(region, _spacing, [&](const Vector2D& point) {
            double newAngleInRadian = (random() - 0.5) * kTwoPiD;
            Matrix2x2D rotationMatrix =
                Matrix2x2D::makeRotationMatrix(newAngleInRadian);
            Vector2D randomDir = rotationMatrix * Vector2D();
            Vector2D offset = maxJitterDist * randomDir;
            Vector2D candidate = point + offset;
            if (_implicitSurface->isInside(candidate) &&
                (!_allowOverlapping &&
                 !neighborSearcher.hasNearbyPoint(candidate, _spacing))) {
                if (_numberOfEmittedParticles < _maxNumberOfParticles) {
                    newPositions->append(candidate);
                    neighborSearcher.add(candidate);
                    ++_numberOfEmittedParticles;
                    ++numNewParticles;
                } else {
                    return false;
                }
            }

            return true;
        });
    }

    JET_INFO << "Number of newly generated particles: " << numNewParticles;
    JET_INFO << "Number of total generated particles: "
             << _numberOfEmittedParticles;

    newVelocities->resize(newPositions->size());
    newVelocities->parallelForEachIndex([&](size_t i) {
        (*newVelocities)[i] = velocityAt((*newPositions)[i]);
    });
}

void VolumeParticleEmitter2::setPointGenerator(
    const PointGenerator2Ptr& newPointsGen) {
    _pointsGen = newPointsGen;
}

const ImplicitSurface2Ptr& VolumeParticleEmitter2::surface() const {
    return _implicitSurface;
}

void VolumeParticleEmitter2::setSurface(const ImplicitSurface2Ptr& newSurface) {
    _implicitSurface = newSurface;
}

const BoundingBox2D& VolumeParticleEmitter2::maxRegion() const {
    return _bounds;
}

void VolumeParticleEmitter2::setMaxRegion(const BoundingBox2D& newMaxRegion) {
    _bounds = newMaxRegion;
}

double VolumeParticleEmitter2::jitter() const { return _jitter; }

void VolumeParticleEmitter2::setJitter(double newJitter) {
    _jitter = clamp(newJitter, 0.0, 1.0);
}

bool VolumeParticleEmitter2::isOneShot() const { return _isOneShot; }

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

double VolumeParticleEmitter2::spacing() const { return _spacing; }

void VolumeParticleEmitter2::setSpacing(double newSpacing) {
    _spacing = newSpacing;
}

Vector2D VolumeParticleEmitter2::initialVelocity() const { return _initialVel; }

void VolumeParticleEmitter2::setInitialVelocity(const Vector2D& newInitialVel) {
    _initialVel = newInitialVel;
}

Vector2D VolumeParticleEmitter2::linearVelocity() const { return _linearVel; }

void VolumeParticleEmitter2::setLinearVelocity(const Vector2D& newLinearVel) {
    _linearVel = newLinearVel;
}

double VolumeParticleEmitter2::angularVelocity() const { return _angularVel; }

void VolumeParticleEmitter2::setAngularVelocity(double newAngularVel) {
    _angularVel = newAngularVel;
}

double VolumeParticleEmitter2::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
}

Vector2D VolumeParticleEmitter2::velocityAt(const Vector2D& point) const {
    Vector2D r = point - _implicitSurface->transform.translation();
    return _linearVel + _angularVel * Vector2D(-r.y, r.x) + _initialVel;
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

VolumeParticleEmitter2::Builder& VolumeParticleEmitter2::Builder::withSurface(
    const Surface2Ptr& surface) {
    _implicitSurface = std::make_shared<SurfaceToImplicit2>(surface);
    if (!_isBoundSet) {
        _bounds = surface->boundingBox();
    }
    return *this;
}

VolumeParticleEmitter2::Builder& VolumeParticleEmitter2::Builder::withMaxRegion(
    const BoundingBox2D& bounds) {
    _bounds = bounds;
    _isBoundSet = true;
    return *this;
}

VolumeParticleEmitter2::Builder& VolumeParticleEmitter2::Builder::withSpacing(
    double spacing) {
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
VolumeParticleEmitter2::Builder::withLinearVelocity(const Vector2D& linearVel) {
    _linearVel = linearVel;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withAngularVelocity(double angularVel) {
    _angularVel = angularVel;
    return *this;
}

VolumeParticleEmitter2::Builder&
VolumeParticleEmitter2::Builder::withMaxNumberOfParticles(
    size_t maxNumberOfParticles) {
    _maxNumberOfParticles = maxNumberOfParticles;
    return *this;
}

VolumeParticleEmitter2::Builder& VolumeParticleEmitter2::Builder::withJitter(
    double jitter) {
    _jitter = jitter;
    return *this;
}

VolumeParticleEmitter2::Builder& VolumeParticleEmitter2::Builder::withIsOneShot(
    bool isOneShot) {
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
    return VolumeParticleEmitter2(_implicitSurface, _bounds, _spacing,
                                  _initialVel, _linearVel, _angularVel,
                                  _maxNumberOfParticles, _jitter, _isOneShot,
                                  _allowOverlapping, _seed);
}

VolumeParticleEmitter2Ptr VolumeParticleEmitter2::Builder::makeShared() const {
    return std::shared_ptr<VolumeParticleEmitter2>(
        new VolumeParticleEmitter2(_implicitSurface, _bounds, _spacing,
                                   _initialVel, _linearVel, _angularVel,
                                   _maxNumberOfParticles, _jitter, _isOneShot,
                                   _allowOverlapping),
        [](VolumeParticleEmitter2* obj) { delete obj; });
}
