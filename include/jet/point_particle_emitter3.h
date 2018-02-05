// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_
#define INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_

#include <jet/particle_emitter3.h>
#include <limits>
#include <random>

namespace jet {

//!
//! \brief 3-D point particle emitter.
//!
//! This class emits particles from a single point in given direction, speed,
//! and spreading angle.
//!
class PointParticleEmitter3 final : public ParticleEmitter3 {
 public:
    class Builder;

    //!
    //! Constructs an emitter that spawns particles from given origin,
    //! direction, speed, spread angle, max number of new particles per second,
    //! max total number of particles to be emitted, and random seed.
    //!
    //! \param[in]  origin                      The origin.
    //! \param[in]  direction                   The direction.
    //! \param[in]  speed                       The speed.
    //! \param[in]  spreadAngleInDegrees        The spread angle in degrees.
    //! \param[in]  maxNumOfNewParticlesPerSec  The max number of new particles
    //!                                         per second.
    //! \param[in]  maxNumOfParticles           The max number of particles to
    //!                                         be emitted.
    //! \param[in]  seed                        The random seed.
    //!
    PointParticleEmitter3(
        const Vector3D& origin,
        const Vector3D& direction,
        double speed,
        double spreadAngleInDegrees,
        size_t maxNumOfNewParticlesPerSec = 1,
        size_t maxNumOfParticles = std::numeric_limits<size_t>::max(),
        uint32_t seed = 0);

    //! Returns max number of new particles per second.
    size_t maxNumberOfNewParticlesPerSecond() const;

    //! Sets max number of new particles per second.
    void setMaxNumberOfNewParticlesPerSecond(size_t rate);

    //! Returns max number of particles to be emitted.
    size_t maxNumberOfParticles() const;

    //! Sets max number of particles to be emitted.
    void setMaxNumberOfParticles(size_t maxNumberOfParticles);

    //! Returns builder fox PointParticleEmitter3.
    static Builder builder();

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

    //!
    //! \brief      Emits particles to the particle system data.
    //!
    //! \param[in]  currentTimeInSeconds    Current simulation time.
    //! \param[in]  timeIntervalInSeconds   The time-step interval.
    //!
    void onUpdate(
        double currentTimeInSeconds,
        double timeIntervalInSeconds) override;

    void emit(
        Array1<Vector3D>* newPositions,
        Array1<Vector3D>* newVelocities,
        size_t maxNewNumberOfParticles);

    double random();
};

//! Shared pointer for the PointParticleEmitter3 type.
typedef std::shared_ptr<PointParticleEmitter3> PointParticleEmitter3Ptr;


//!
//! \brief Front-end to create PointParticleEmitter3 objects step by step.
//!
class PointParticleEmitter3::Builder final {
 public:
    //! Returns builder with origin.
    Builder& withOrigin(const Vector3D& origin);

    //! Returns builder with direction.
    Builder& withDirection(const Vector3D& direction);

    //! Returns builder with speed.
    Builder& withSpeed(double speed);

    //! Returns builder with spread angle in degrees.
    Builder& withSpreadAngleInDegrees(double spreadAngleInDegrees);

    Builder& withMaxNumberOfNewParticlesPerSecond(
        size_t maxNumOfNewParticlesPerSec);

    //! Returns builder with max number of particles.
    Builder& withMaxNumberOfParticles(size_t maxNumberOfParticles);

    //! Returns builder with random seed.
    Builder& withRandomSeed(uint32_t seed);

    //! Builds PointParticleEmitter3.
    PointParticleEmitter3 build() const;

    //! Builds shared pointer of PointParticleEmitter3 instance.
    PointParticleEmitter3Ptr makeShared() const;

 private:
    size_t _maxNumberOfNewParticlesPerSecond = 1;
    size_t _maxNumberOfParticles = kMaxSize;
    Vector3D _origin{0, 0, 0};
    Vector3D _direction{0, 1, 0};
    double _speed = 1.0;
    double _spreadAngleInDegrees = 90.0;
    uint32_t _seed = 0;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_PARTICLE_EMITTER3_H_
