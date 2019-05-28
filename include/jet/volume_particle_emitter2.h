// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_
#define INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_

#include <jet/constants.h>
#include <jet/bounding_box2.h>
#include <jet/implicit_surface2.h>
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

    //! Constructs an emitter that spawns particles from given implicit surface
    //! which defines the volumetric geometry. Provided bounding box limits
    //! the particle generation region.
    //!
    //! \param[in]  implicitSurface         The implicit surface.
    //! \param[in]  maxRegion               The max region.
    //! \param[in]  spacing                 The spacing between particles.
    //! \param[in]  initialVel              The initial velocity of new particles.
    //! \param[in]  linearVel               The linear velocity of the emitter.
    //! \param[in]  angularVel              The angular velocity of the emitter.
    //! \param[in]  maxNumberOfParticles    The max number of particles to be
    //!                                     emitted.
    //! \param[in]  jitter                  The jitter amount between 0 and 1.
    //! \param[in]  isOneShot               True if emitter gets disabled after one shot.
    //! \param[in]  allowOverlapping        True if particles can be overlapped.
    //! \param[in]  seed                    The random seed.
    //!
    VolumeParticleEmitter2(
        const ImplicitSurface2Ptr& implicitSurface,
        const BoundingBox2D& maxRegion,
        double spacing,
        const Vector2D& initialVel = Vector2D(),
        const Vector2D& linearVel = Vector2D(),
        double angularVel = 0.0,
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

    //! Returns source surface.
    const ImplicitSurface2Ptr& surface() const;

    //! Sets the source surface.
    void setSurface(const ImplicitSurface2Ptr& newSurface);

    //! Returns max particle gen region.
    const BoundingBox2D& maxRegion() const;

    //! Sets the max particle gen region.
    void setMaxRegion(const BoundingBox2D& newBox);

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

    //! Returns the linear velocity of the emitter.
    Vector2D linearVelocity() const;

    //! Sets the linear velocity of the emitter.
    void setLinearVelocity(const Vector2D& newLinearVel);

    //! Returns the angular velocity of the emitter.
    double angularVelocity() const;

    //! Sets the linear velocity of the emitter.
    void setAngularVelocity(double newAngularVel);

    //! Returns builder fox VolumeParticleEmitter2.
    static Builder builder();

 private:
    std::mt19937 _rng;

    ImplicitSurface2Ptr _implicitSurface;
    BoundingBox2D _bounds;
    double _spacing;
    Vector2D _initialVel;
    Vector2D _linearVel;
    double _angularVel = 0.0;
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

    Vector2D velocityAt(const Vector2D& point) const;
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

    //! Returns builder with linear velocity.
    Builder& withLinearVelocity(const Vector2D& linearVel);

    //! Returns builder with angular velocity.
    Builder& withAngularVelocity(double angularVel);

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
    Vector2D _initialVel;
    Vector2D _linearVel;
    double _angularVel = 0.0;
    size_t _maxNumberOfParticles = kMaxSize;
    double _jitter = 0.0;
    bool _isOneShot = true;
    bool _allowOverlapping = false;
    uint32_t _seed = 0;
};

}  // namespace jet

#endif  // INCLUDE_JET_VOLUME_PARTICLE_EMITTER2_H_
