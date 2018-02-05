// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_SOLVER2_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_SOLVER2_H_

#include <jet/collider2.h>
#include <jet/constants.h>
#include <jet/vector_field2.h>
#include <jet/particle_emitter2.h>
#include <jet/particle_system_data2.h>
#include <jet/physics_animation.h>

namespace jet {

//!
//! \brief      Basic 2-D particle system solver.
//!
//! This class implements basic particle system solver. It includes gravity,
//! air drag, and collision. But it does not compute particle-to-particle
//! interaction. Thus, this solver is suitable for performing simple spray-like
//! simulations with low computational cost. This class can be further extend
//! to add more sophisticated simulations, such as SPH, to handle
//! particle-to-particle intersection.
//!
//! \see        SphSolver2
//!
class ParticleSystemSolver2 : public PhysicsAnimation {
 public:
    class Builder;

    //! Constructs an empty solver.
    ParticleSystemSolver2();

    //! Constructs a solver with particle parameters.
    ParticleSystemSolver2(
        double radius,
        double mass);

    //! Destructor.
    virtual ~ParticleSystemSolver2();

    //! Returns the drag coefficient.
    double dragCoefficient() const;

    //!
    //! \brief      Sets the drag coefficient.
    //!
    //! The drag coefficient controls the amount of air-drag. The coefficient
    //! should be a positive number and 0 means no drag force.
    //!
    //! \param[in]  newDragCoefficient The new drag coefficient.
    //!
    void setDragCoefficient(double newDragCoefficient);

    //! Sets the restitution coefficient.
    double restitutionCoefficient() const;

    //!
    //! \brief      Sets the restitution coefficient.
    //!
    //! The restitution coefficient controls the bouncy-ness of a particle when
    //! it hits a collider surface. The range of the coefficient should be 0 to
    //! 1 -- 0 means no bounce back and 1 means perfect reflection.
    //!
    //! \param[in]  newRestitutionCoefficient The new restitution coefficient.
    //!
    void setRestitutionCoefficient(double newRestitutionCoefficient);

    //! Returns the gravity.
    const Vector2D& gravity() const;

    //! Sets the gravity.
    void setGravity(const Vector2D& newGravity);

    //!
    //! \brief      Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    //! \return     The particle system data.
    //!
    const ParticleSystemData2Ptr& particleSystemData() const;

    //! Returns the collider.
    const Collider2Ptr& collider() const;

    //! Sets the collider.
    void setCollider(const Collider2Ptr& newCollider);

    //! Returns the emitter.
    const ParticleEmitter2Ptr& emitter() const;

    //! Sets the emitter.
    void setEmitter(const ParticleEmitter2Ptr& newEmitter);

    //! Returns the wind field.
    const VectorField2Ptr& wind() const;

    //!
    //! \brief      Sets the wind.
    //!
    //! Wind can be applied to the particle system by setting a vector field to
    //! the solver.
    //!
    //! \param[in]  newWind The new wind.
    //!
    void setWind(const VectorField2Ptr& newWind);

    //! Returns builder fox ParticleSystemSolver2.
    static Builder builder();

 protected:
    //! Initializes the simulator.
    void onInitialize() override;

    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    //! Accumulates forces applied to the particles.
    virtual void accumulateForces(double timeStepInSeconds);

    //! Called when a time-step is about to begin.
    virtual void onBeginAdvanceTimeStep(double timeStepInSeconds);

    //! Called after a time-step is completed.
    virtual void onEndAdvanceTimeStep(double timeStepInSeconds);

    //! Resolves any collisions occured by the particles.
    void resolveCollision();

    //! Resolves any collisions occured by the particles where the particle
    //! state is given by the position and velocity arrays.
    void resolveCollision(
        ArrayAccessor1<Vector2D> newPositions,
        ArrayAccessor1<Vector2D> newVelocities);

    //! Assign a new particle system data.
    void setParticleSystemData(const ParticleSystemData2Ptr& newParticles);

 private:
    double _dragCoefficient = 1e-4;
    double _restitutionCoefficient = 0.0;
    Vector2D _gravity = Vector2D(0.0, kGravity);

    ParticleSystemData2Ptr _particleSystemData;
    ParticleSystemData2::VectorData _newPositions;
    ParticleSystemData2::VectorData _newVelocities;
    Collider2Ptr _collider;
    ParticleEmitter2Ptr _emitter;
    VectorField2Ptr _wind;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void accumulateExternalForces();

    void timeIntegration(double timeStepInSeconds);

    void updateCollider(double timeStepInSeconds);

    void updateEmitter(double timeStepInSeconds);
};

//! Shared pointer type for the ParticleSystemSolver2.
typedef std::shared_ptr<ParticleSystemSolver2> ParticleSystemSolver2Ptr;


//!
//! \brief Base class for particle-based solver builder.
//!
template <typename DerivedBuilder>
class ParticleSystemSolverBuilderBase2 {
 public:
    //! Returns builder with particle radius.
    DerivedBuilder& withRadius(double radius);

    //! Returns builder with mass per particle.
    DerivedBuilder& withMass(double mass);

 protected:
    double _radius = 1e-3;
    double _mass = 1e-3;
};

template <typename T>
T& ParticleSystemSolverBuilderBase2<T>::withRadius(double radius) {
    _radius = radius;
    return static_cast<T&>(*this);
}

template <typename T>
T& ParticleSystemSolverBuilderBase2<T>::withMass(double mass) {
    _mass = mass;
    return static_cast<T&>(*this);
}

//!
//! \brief Front-end to create ParticleSystemSolver2 objects step by step.
//!
class ParticleSystemSolver2::Builder final
    : public ParticleSystemSolverBuilderBase2<ParticleSystemSolver2::Builder> {
 public:
    //! Builds ParticleSystemSolver2.
    ParticleSystemSolver2 build() const;

    //! Builds shared pointer of ParticleSystemSolver2 instance.
    ParticleSystemSolver2Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_SOLVER2_H_
