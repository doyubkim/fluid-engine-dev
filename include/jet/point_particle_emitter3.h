// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_
#define INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_

#include <jet/particle_emitter3.h>
#include <limits>
#include <random>

namespace jet {

class PointParticleEmitter3 final : public ParticleEmitter3 {
 public:
    PointParticleEmitter3(
        const Vector3D& origin,
        const Vector3D& direction,
        double speed,
        double spreadAngleInDegrees,
        uint32_t seed = 0);

    void emit(
        const Frame& frame,
        const ParticleSystemData3Ptr& particles) override;

    size_t maxNumberOfNewParticlesPerSecond() const;

    void setMaxNumberOfNewParticlesPerSecond(size_t rate);

    size_t maxNumberOfParticles() const;

    void setMaxNumberOfParticles(size_t maxNumberOfParticles);

 protected:
    // ParticleEmitter3 implementation
    void emit(
        Array1<Vector3D>* newPositions,
        Array1<Vector3D>* newVelocities,
        size_t maxNewNumberOfParticles);

 private:
    std::mt19937 _rng;

    double _firstFrameTimeInSeconds = 0.0;
    size_t _numberOfEmittedParticles = 0;

    size_t _maxNumberOfNewParticlesPerSecond = 1;
    size_t _maxNumberOfParticles = std::numeric_limits<size_t>::max();

    Vector3D _origin;
    Vector3D _direction;
    double _speed;
    double _spreadAngleInRadians;

    double random();
};

typedef std::shared_ptr<PointParticleEmitter3> PointParticleEmitter3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_
