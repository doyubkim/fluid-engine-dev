// Copyright (c) 2017 Doyub Kim
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
    const ImplicitSurface3Ptr& implicitSurface,
    const BoundingBox3D& bounds,
    double spacing,
    const Vector3D& initialVel,
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
    _pointsGen = std::make_shared<BccLatticePointGenerator>();
}

void VolumeParticleEmitter3::onUpdate(
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

    Array1<Vector3D> newPositions;
    Array1<Vector3D> newVelocities;

    emit(particles, &newPositions, &newVelocities);

    particles->addParticles(newPositions, newVelocities);
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
//                    printf("%f/%f - %f, %f, %f\n", _implicitSurface->signedDistance(candidate), _spacing, candidate.x, candidate.y, candidate.z);
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
        PointHashGridSearcher3 neighborSearcher(
            Size3(
                kDefaultHashGridResolution,
                kDefaultHashGridResolution,
                kDefaultHashGridResolution),
            2.0 * _spacing);
        if (!_allowOverlapping) {
            neighborSearcher.build(particles->positions());
        }

        _pointsGen->forEachPoint(
            _bounds,
            _spacing,
            [&] (const Vector3D& point) {
                Vector3D randomDir = uniformSampleSphere(
                    random(),
                    random());
                Vector3D offset = maxJitterDist * randomDir;
                Vector3D candidate = point + offset;
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

void VolumeParticleEmitter3::setPointGenerator(
    const PointGenerator3Ptr& newPointsGen) {
    _pointsGen = newPointsGen;
}

double VolumeParticleEmitter3::jitter() const {
    return _jitter;
}

void VolumeParticleEmitter3::setJitter(double newJitter) {
    _jitter = clamp(newJitter, 0.0, 1.0);
}

bool VolumeParticleEmitter3::isOneShot() const {
    return _isOneShot;
}

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

double VolumeParticleEmitter3::spacing() const {
    return _spacing;
}

void VolumeParticleEmitter3::setSpacing(double newSpacing) {
    _spacing = newSpacing;
}

Vector3D VolumeParticleEmitter3::initialVelocity() const {
    return _initialVel;
}

void VolumeParticleEmitter3::setInitialVelocity(const Vector3D& newInitialVel) {
    _initialVel = newInitialVel;
}

double VolumeParticleEmitter3::random() {
    std::uniform_real_distribution<> d(0.0, 1.0);
    return d(_rng);
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

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withSurface(
    const Surface3Ptr& surface) {
    _implicitSurface = std::make_shared<SurfaceToImplicit3>(surface);
    if (!_isBoundSet) {
        _bounds = surface->boundingBox();
    }
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withMaxRegion(const BoundingBox3D& bounds) {
    _bounds = bounds;
    _isBoundSet = true;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withSpacing(double spacing) {
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
VolumeParticleEmitter3::Builder::withMaxNumberOfParticles(
    size_t maxNumberOfParticles) {
    _maxNumberOfParticles = maxNumberOfParticles;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withJitter(double jitter) {
    _jitter = jitter;
    return *this;
}

VolumeParticleEmitter3::Builder&
VolumeParticleEmitter3::Builder::withIsOneShot(bool isOneShot) {
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
    return VolumeParticleEmitter3(
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

VolumeParticleEmitter3Ptr VolumeParticleEmitter3::Builder::makeShared() const {
    return std::shared_ptr<VolumeParticleEmitter3>(
        new VolumeParticleEmitter3(
            _implicitSurface,
            _bounds,
            _spacing,
            _initialVel,
            _maxNumberOfParticles,
            _jitter,
            _isOneShot,
            _allowOverlapping),
        [] (VolumeParticleEmitter3* obj) {
            delete obj;
        });
}
