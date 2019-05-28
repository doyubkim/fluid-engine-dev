// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/bcc_lattice_point_generator.h>
#include <jet/point_hash_grid_searcher3.h>
#include <jet/samplers.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

static const size_t kDefaultHashGridResolution = 64;

VolumeParticleEmitter3::VolumeParticleEmitter3(
    const ImplicitSurface3Ptr& implicitSurface, const BoundingBox3D& maxRegion,
    double spacing, const Vector3D& initialVel, const Vector3D& linearVel,
    const Vector3D& angularVel, size_t maxNumberOfParticles, double jitter,
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
    _pointsGen = std::make_shared<BccLatticePointGenerator>();
}

void VolumeParticleEmitter3::onUpdate(double currentTimeInSeconds,
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

    Array1<Vector3D> newPositions;
    Array1<Vector3D> newVelocities;

    emit(particles, &newPositions, &newVelocities);

    particles->addParticles(newPositions, newVelocities);

    if (_isOneShot) {
        setIsEnabled(false);
    }
}

void VolumeParticleEmitter3::emit(const ParticleSystemData3Ptr& particles,
                                  Array1<Vector3D>* newPositions,
                                  Array1<Vector3D>* newVelocities) {
    if (!_implicitSurface) {
        return;
    }

    _implicitSurface->updateQueryEngine();

    BoundingBox3D region = _bounds;
    if (_implicitSurface->isBounded()) {
        BoundingBox3D surfaceBBox = _implicitSurface->boundingBox();
        region.lowerCorner = max(region.lowerCorner, surfaceBBox.lowerCorner);
        region.upperCorner = min(region.upperCorner, surfaceBBox.upperCorner);
    }

    // Reserving more space for jittering
    const double j = jitter();
    const double maxJitterDist = 0.5 * j * _spacing;
    size_t numNewParticles = 0;

    if (_allowOverlapping || _isOneShot) {
        _pointsGen->forEachPoint(region, _spacing, [&](const Vector3D& point) {
            Vector3D randomDir = uniformSampleSphere(random(), random());
            Vector3D offset = maxJitterDist * randomDir;
            Vector3D candidate = point + offset;
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
        PointHashGridSearcher3 neighborSearcher(
            Size3(kDefaultHashGridResolution, kDefaultHashGridResolution,
                  kDefaultHashGridResolution),
            2.0 * _spacing);
        if (!_allowOverlapping) {
            neighborSearcher.build(particles->positions());
        }

        _pointsGen->forEachPoint(region, _spacing, [&](const Vector3D& point) {
            Vector3D randomDir = uniformSampleSphere(random(), random());
            Vector3D offset = maxJitterDist * randomDir;
            Vector3D candidate = point + offset;
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

void VolumeParticleEmitter3::setPointGenerator(
    const PointGenerator3Ptr& newPointsGen) {
    _pointsGen = newPointsGen;
}

const ImplicitSurface3Ptr& VolumeParticleEmitter3::surface() const {
    return _implicitSurface;
}

void VolumeParticleEmitter3::setSurface(const ImplicitSurface3Ptr& newSurface) {
    _implicitSurface = newSurface;
}

const BoundingBox3D& VolumeParticleEmitter3::maxRegion() const {
    return _bounds;
}

void VolumeParticleEmitter3::setMaxRegion(const BoundingBox3D& newMaxRegion) {
    _bounds = newMaxRegion;
}

double VolumeParticleEmitter3::jitter() const { return _jitter; }

void VolumeParticleEmitter3::setJitter(double newJitter) {
    _jitter = clamp(newJitter, 0.0, 1.0);
}

bool VolumeParticleEmitter3::isOneShot() const { return _isOneShot; }

void VolumeParticleEmitter3::setIsOneShot(bool newValue) {
    _isOneShot = newValue;
}

bool VolumeParticleEmitter3::allowOverlapping() const {
    return _allowOverlapping;
}

void VolumeParticleEmitter3::setAllowOverlapping(bool newValue) {
    _allowOverlapping = newValue;
}

size_t VolumeParticleEmitter3::maxNumberOfParticles() const {
    return _maxNumberOfParticles;
}

void VolumeParticleEmitter3::setMaxNumberOfParticles(
    size_t newMaxNumberOfParticles) {
    _maxNumberOfParticles = newMaxNumberOfParticles;
}

double VolumeParticleEmitter3::spacing() const { return _spacing; }

void VolumeParticleEmitter3::setSpacing(double newSpacing) {
    _spacing = newSpacing;
}

Vector3D VolumeParticleEmitter3::initialVelocity() const { return _initialVel; }

void VolumeParticleEmitter3::setInitialVelocity(const Vector3D& newInitialVel) {
    _initialVel = newInitialVel;
}

Vector3D VolumeParticleEmitter3::linearVelocity() const { return _linearVel; }

void VolumeParticleEmitter3::setLinearVelocity(const Vector3D& newLinearVel) {
    _linearVel = newLinearVel;
}

Vector3D VolumeParticleEmitter3::angularVelocity() const { return _angularVel; }

void VolumeParticleEmitter3::setAngularVelocity(const Vector3D& newAngularVel) {
    _angularVel = newAngularVel;
}

double VolumeParticleEmitter3::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
}

Vector3D VolumeParticleEmitter3::velocityAt(const Vector3D& point) const {
    Vector3D r = point - _implicitSurface->transform.translation();
    return _linearVel + _angularVel.cross(r) + _initialVel;
}

VolumeParticleEmitter3::Builder VolumeParticleEmitter3::builder() {
    return Builder();
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withImplicitSurface(
    const ImplicitSurface3Ptr& implicitSurface) {
    _implicitSurface = implicitSurface;
    if (!_isBoundSet) {
        _bounds = _implicitSurface->boundingBox();
    }
    return *this;
}

VolumeParticleEmitter3::Builder& VolumeParticleEmitter3::Builder::withSurface(
    const Surface3Ptr& surface) {
    _implicitSurface = std::make_shared<SurfaceToImplicit3>(surface);
    if (!_isBoundSet) {
        _bounds = surface->boundingBox();
    }
    return *this;
}

VolumeParticleEmitter3::Builder& VolumeParticleEmitter3::Builder::withMaxRegion(
    const BoundingBox3D& bounds) {
    _bounds = bounds;
    _isBoundSet = true;
    return *this;
}

VolumeParticleEmitter3::Builder& VolumeParticleEmitter3::Builder::withSpacing(
    double spacing) {
    _spacing = spacing;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withInitialVelocity(
    const Vector3D& initialVel) {
    _initialVel = initialVel;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withLinearVelocity(const Vector3D& linearVel) {
    _linearVel = linearVel;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withAngularVelocity(
    const Vector3D& angularVel) {
    _angularVel = angularVel;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withMaxNumberOfParticles(
    size_t maxNumberOfParticles) {
    _maxNumberOfParticles = maxNumberOfParticles;
    return *this;
}

VolumeParticleEmitter3::Builder& VolumeParticleEmitter3::Builder::withJitter(
    double jitter) {
    _jitter = jitter;
    return *this;
}

VolumeParticleEmitter3::Builder& VolumeParticleEmitter3::Builder::withIsOneShot(
    bool isOneShot) {
    _isOneShot = isOneShot;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withAllowOverlapping(bool allowOverlapping) {
    _allowOverlapping = allowOverlapping;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withRandomSeed(uint32_t seed) {
    _seed = seed;
    return *this;
}

VolumeParticleEmitter3 VolumeParticleEmitter3::Builder::build() const {
    return VolumeParticleEmitter3(_implicitSurface, _bounds, _spacing,
                                  _initialVel, _linearVel, _angularVel,
                                  _maxNumberOfParticles, _jitter, _isOneShot,
                                  _allowOverlapping, _seed);
}

VolumeParticleEmitter3Ptr VolumeParticleEmitter3::Builder::makeShared() const {
    return std::shared_ptr<VolumeParticleEmitter3>(
        new VolumeParticleEmitter3(_implicitSurface, _bounds, _spacing,
                                   _initialVel, _linearVel, _angularVel,
                                   _maxNumberOfParticles, _jitter, _isOneShot,
                                   _allowOverlapping),
        [](VolumeParticleEmitter3* obj) { delete obj; });
}
