// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_POINT_PARTICLE_EMITTER2_H_
#define INCLUDE_JET_POINT_PARTICLE_EMITTER2_H_

#include <jet/particle_emitter2.h>
#include <limits>
#include <random>

namespace jet {

class PointParticleEmitter2 final : public ParticleEmitter2 {
 public:
    PointParticleEmitter2(
        const Vector2D& origin,
        const Vector2D& direction,
        double speed,
        double spreadAngleInDegrees,
        uint32_t seed = 0);

    void emit(
        const Frame& frame,
        const ParticleSystemData2Ptr& particles) override;

    size_t maxNumberOfNewParticlesPerSecond() const;

    void setMaxNumberOfNewParticlesPerSecond(size_t rate);

    size_t maxNumberOfParticles() const;

    void setMaxNumberOfParticles(size_t maxNumberOfParticles);

 protected:
    // ParticleEmitter2 implementation
    void emit(
        Array1<Vector2D>* newPositions,
        Array1<Vector2D>* newVelocities,
        size_t maxNewNumberOfParticles);

 private:
    std::mt19937 _rng;

    double _firstFrameTimeInSeconds = 0.0;
    size_t _numberOfEmittedParticles = 0;

    size_t _maxNumberOfNewParticlesPerSecond = 1;
    size_t _maxNumberOfParticles = std::numeric_limits<size_t>::max();

    Vector2D _origin;
    Vector2D _direction;
    double _speed;
    double _spreadAngleInRadians;

    double random();
};

typedef std::shared_ptr<PointParticleEmitter2> PointParticleEmitter2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARTICLE_EMITTER2_H_
