// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_
#define INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_

#include <jet/bounding_box2.h>
#include <jet/implicit_surface2.h>
#include <jet/particle_emitter2.h>
#include <jet/points_generator2.h>
#include <limits>
#include <memory>
#include <random>

namespace jet {

class VolumeParticleEmitter2 final : public ParticleEmitter2 {
 public:
    VolumeParticleEmitter2(
        const ImplicitSurface2Ptr& implicitSurface,
        const BoundingBox2D& bounds,
        double spacing,
        const Vector2D& initialVel = Vector2D(),
        uint32_t seed = 0);

    // ParticleEmitter2 implementation
    void emit(
        const Frame& frame,
        const ParticleSystemData2Ptr& particles) override;

    void setPointGenerator(const PointsGenerator2Ptr& newPointsGen);

    //! Returns jitter amount [0, 1].
    double jitter() const;

    //! Sets jitter amount [0, 1].
    void setJitter(double newJitter);

    void setIsOneShot(bool newValue);

    void setAllowOverlapping(bool newValue);

 protected:
    void emit(
        const ParticleSystemData2Ptr& particles,
        Array1<Vector2D>* newPositions,
        Array1<Vector2D>* newVelocities);

 private:
    std::mt19937 _rng;
    double _jitter = 0.0;
    bool _isOneShot = true;
    bool _allowOverlapping = false;

    ImplicitSurface2Ptr _implicitSurface;
    BoundingBox2D _bounds;
    double _spacing;
    Vector2D _initialVel;
    PointsGenerator2Ptr _pointsGen;

    size_t _maxNumberOfParticles = std::numeric_limits<size_t>::max();
    size_t _numberOfEmittedParticles = 0;

    double random();
};

typedef std::shared_ptr<VolumeParticleEmitter2> VolumeParticleEmitter2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_
