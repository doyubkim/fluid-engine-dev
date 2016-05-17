// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VOLUME_PARTICLE_EMITTER3_H_
#define INCLUDE_JET_VOLUME_PARTICLE_EMITTER3_H_

#include <jet/bounding_box3.h>
#include <jet/implicit_surface3.h>
#include <jet/particle_emitter3.h>
#include <jet/points_generator3.h>
#include <limits>
#include <memory>
#include <random>

namespace jet {

class VolumeParticleEmitter3 final : public ParticleEmitter3 {
 public:
    VolumeParticleEmitter3(
        const ImplicitSurface3Ptr& implicitSurface,
        const BoundingBox3D& bounds,
        double spacing,
        const Vector3D& initialVel = Vector3D(),
        uint32_t seed = 0);

    // ParticleEmitter3 implementation
    void emit(
        const Frame& frame,
        const ParticleSystemData3Ptr& particles) override;

    void setPointGenerator(const PointsGenerator3Ptr& newPointsGen);

    //! Returns jitter amount [0, 1].
    double jitter() const;

    //! Sets jitter amount [0, 1].
    void setJitter(double newJitter);

    void setIsOneShot(bool newValue);

    void setAllowOverlapping(bool newValue);

 protected:
    void emit(
        const ParticleSystemData3Ptr& particles,
        Array1<Vector3D>* newPositions,
        Array1<Vector3D>* newVelocities);

 private:
    std::mt19937 _rng;
    double _jitter = 0.0;
    bool _isOneShot = true;
    bool _allowOverlapping = false;

    ImplicitSurface3Ptr _implicitSurface;
    BoundingBox3D _bounds;
    double _spacing;
    Vector3D _initialVel;
    PointsGenerator3Ptr _pointsGen;

    size_t _maxNumberOfParticles = std::numeric_limits<size_t>::max();
    size_t _numberOfEmittedParticles = 0;

    double random();
};

typedef std::shared_ptr<VolumeParticleEmitter3> VolumeParticleEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_VOLUME_PARTICLE_EMITTER3_H_
