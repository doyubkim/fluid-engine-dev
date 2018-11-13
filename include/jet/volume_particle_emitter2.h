// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_
#define INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_

#include <jet/constants.h>
#include <jet/bounding_box.h>
#include <jet/implicit_surface.h>
#include <jet/particle_emitter2.h>
#include <jet/point_generator2.h>
#include <limits>
#include <memory>
#include <random>

namespace jet {

//!
//! \brief 2-D volumetric particle emitter.
//!
//! This class emits particles from volumetric geometry.
//!
class VolumeParticleEmitter2 final : public ParticleEmitter2 {
 public:
    class Builder;

    //!
    //! Constructs an emitter that spawns particles from given implicit surface
    //! which defines the volumetric geometry. Provided bounding box limits
    //! the particle generation region.
    //!
    //! \param[in]  implicitSurface         The implicit surface.
    //! \param[in]  bounds                  The max region.
    //! \param[in]  spacing                 The spacing between particles.
    //! \param[in]  initialVel              The initial velocity.
    //! \param[in]  maxNumberOfParticles    The max number of particles to be
    //!                                     emitted.
    //! \param[in]  jitter                  The jitter amount between 0 and 1.
    //! \param[in]  isOneShot               Set true if particles are emitted
    //!                                     just once.
    //! \param[in]  allowOverlapping        True if particles can be overlapped.
    //! \param[in]  seed                    The random seed.
    //!
    VolumeParticleEmitter2(
        const ImplicitSurface2Ptr& implicitSurface,
        const BoundingBox2D& bounds,
        double spacing,
        const Vector2D& initialVel = Vector2D(),
        size_t maxNumberOfParticles = kMaxSize,
        double jitter = 0.0,
        bool isOneShot = true,
        bool allowOverlapping = false,
        uint32_t seed = 0);

    //!
    //! \brief      Sets the point generator.
    //!
    //! This function sets the point generator that defines the pattern of the
    //! point distribution within the volume.
    //!
    //! \param[in]  newPointsGen The new points generator.
    //!
    void setPointGenerator(const PointGenerator2Ptr& newPointsGen);

    //! Returns jitter amount.
    double jitter() const;

    //! Sets jitter amount between 0 and 1.
    void setJitter(double newJitter);

    //! Returns true if particles should be emitted just once.
    bool isOneShot() const;

    //!
    //! \brief      Sets the flag to true if particles are emitted just once.
    //!
    //! If true is set, the emitter will generate particles only once even after
    //! multiple emit calls. If false, it will keep generating particles from
    //! the volumetric geometry. Default value is true.
    //!
    //! \param[in]  newValue True if particles should be emitted just once.
    //!
    void setIsOneShot(bool newValue);

    //! Returns trhe if particles can be overlapped.
    bool allowOverlapping() const;

    //!
    //! \brief      Sets the flag to true if particles can overlap each other.
    //!
    //! If true is set, the emitter will generate particles even if the new
    //! particles can find existing nearby particles within the particle
    //! spacing.
    //!
    //! \param[in]  newValue True if particles can be overlapped.
    //!
    void setAllowOverlapping(bool newValue);

    //! Returns max number of particles to be emitted.
    size_t maxNumberOfParticles() const;

    //! Sets the max number of particles to be emitted.
    void setMaxNumberOfParticles(size_t newMaxNumberOfParticles);

    //! Returns the spacing between particles.
    double spacing() const;

    //! Sets the spacing between particles.
    void setSpacing(double newSpacing);

    //! Sets the initial velocity of the particles.
    Vector2D initialVelocity() const;

    //! Returns the initial velocity of the particles.
    void setInitialVelocity(const Vector2D& newInitialVel);

    //! Returns builder fox VolumeParticleEmitter2.
    static Builder builder();

 private:
    std::mt19937 _rng;

    ImplicitSurface2Ptr _implicitSurface;
    BoundingBox2D _bounds;
    double _spacing;
    Vector2D _initialVel;
    PointGenerator2Ptr _pointsGen;

    size_t _maxNumberOfParticles = kMaxSize;
    size_t _numberOfEmittedParticles = 0;

    double _jitter = 0.0;
    bool _isOneShot = true;
    bool _allowOverlapping = false;

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
        const ParticleSystemData2Ptr& particles,
        Array1<Vector2D>* newPositions,
        Array1<Vector2D>* newVelocities);

    double random();
};

//! Shared pointer for the VolumeParticleEmitter2 type.
typedef std::shared_ptr<VolumeParticleEmitter2> VolumeParticleEmitter2Ptr;


//!
//! \brief Front-end to create VolumeParticleEmitter2 objects step by step.
//!
class VolumeParticleEmitter2::Builder final {
 public:
    //! Returns builder with implicit surface defining volume shape.
    Builder& withImplicitSurface(const ImplicitSurface2Ptr& implicitSurface);

    //! Returns builder with surface defining volume shape.
    Builder& withSurface(const Surface2Ptr& surface);

    //! Returns builder with max region.
    Builder& withMaxRegion(const BoundingBox2D& bounds);

    //! Returns builder with spacing.
    Builder& withSpacing(double spacing);

    //! Returns builder with initial velocity.
    Builder& withInitialVelocity(const Vector2D& initialVel);

    //! Returns builder with max number of particles.
    Builder& withMaxNumberOfParticles(size_t maxNumberOfParticles);

    //! Returns builder with jitter amount.
    Builder& withJitter(double jitter);

    //! Returns builder with one-shot flag.
    Builder& withIsOneShot(bool isOneShot);

    //! Returns builder with overlapping flag.
    Builder& withAllowOverlapping(bool allowOverlapping);

    //! Returns builder with random seed.
    Builder& withRandomSeed(uint32_t seed);

    //! Builds VolumeParticleEmitter2.
    VolumeParticleEmitter2 build() const;

    //! Builds shared pointer of VolumeParticleEmitter2 instance.
    VolumeParticleEmitter2Ptr makeShared() const;

 private:
    ImplicitSurface2Ptr _implicitSurface;
    bool _isBoundSet = false;
    BoundingBox2D _bounds;
    double _spacing = 0.1;
    Vector2D _initialVel{0, 0};
    size_t _maxNumberOfParticles = kMaxSize;
    double _jitter = 0.0;
    bool _isOneShot = true;
    bool _allowOverlapping = false;
    uint32_t _seed = 0;
};

}  // namespace jet

#endif  // INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_
