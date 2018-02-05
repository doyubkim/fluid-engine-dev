// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_SOLVER3_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_SOLVER3_H_

#include <jet/collider3.h>
#include <jet/constants.h>
#include <jet/vector_field3.h>
#include <jet/particle_emitter3.h>
#include <jet/particle_system_data3.h>
#include <jet/physics_animation.h>

namespace jet {

//!
//! \brief      Basic 3-D particle system solver.
//!
//! This class implements basic particle system solver. It includes gravity,
//! air drag, and collision. But it does not compute particle-to-particle
//! interaction. Thus, this solver is suitable for performing simple spray-like
//! simulations with low computational cost. This class can be further extend
//! to add more sophisticated simulations, such as SPH, to handle
//! particle-to-particle intersection.
//!
//! \see        SphSolver3
//!
class ParticleSystemSolver3 : public PhysicsAnimation {
 public:
    class Builder;

    //! Constructs an empty solver.
    ParticleSystemSolver3();

    //! Constructs a solver with particle parameters.
    ParticleSystemSolver3(
        double radius,
        double mass);

    //! Destructor.
    virtual ~ParticleSystemSolver3();

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
    const Vector3D& gravity() const;

    //! Sets the gravity.
    void setGravity(const Vector3D& newGravity);

    //!
    //! \brief      Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    //! \return     The particle system data.
    //!
    const ParticleSystemData3Ptr& particleSystemData() const;

    //! Returns the collider.
    const Collider3Ptr& collider() const;

    //! Sets the collider.
    void setCollider(const Collider3Ptr& newCollider);

    //! Returns the emitter.
    const ParticleEmitter3Ptr& emitter() const;

    //! Sets the emitter.
    void setEmitter(const ParticleEmitter3Ptr& newEmitter);

    //! Returns the wind field.
    const VectorField3Ptr& wind() const;

    //!
    //! \brief      Sets the wind.
    //!
    //! Wind can be applied to the particle system by setting a vector field to
    //! the solver.
    //!
    //! \param[in]  newWind The new wind.
    //!
    void setWind(const VectorField3Ptr& newWind);

    //! Returns builder fox ParticleSystemSolver3.
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
        ArrayAccessor1<Vector3D> newPositions,
        ArrayAccessor1<Vector3D> newVelocities);

    //! Assign a new particle system data.
    void setParticleSystemData(const ParticleSystemData3Ptr& newParticles);

 private:
    double _dragCoefficient = 1e-4;
    double _restitutionCoefficient = 0.0;
    Vector3D _gravity = Vector3D(0.0, kGravity, 0.0);

    ParticleSystemData3Ptr _particleSystemData;
    ParticleSystemData3::VectorData _newPositions;
    ParticleSystemData3::VectorData _newVelocities;
    Collider3Ptr _collider;
    ParticleEmitter3Ptr _emitter;
    VectorField3Ptr _wind;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void accumulateExternalForces();

    void timeIntegration(double timeStepInSeconds);

    void updateCollider(double timeStepInSeconds);

    void updateEmitter(double timeStepInSeconds);
};

//! Shared pointer type for the ParticleSystemSolver3.
typedef std::shared_ptr<ParticleSystemSolver3> ParticleSystemSolver3Ptr;


//!
//! \brief Base class for particle-based solver builder.
//!
template <typename DerivedBuilder>
class ParticleSystemSolverBuilderBase3 {
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
T& ParticleSystemSolverBuilderBase3<T>::withRadius(double radius) {
    _radius = radius;
    return static_cast<T&>(*this);
}

template <typename T>
T& ParticleSystemSolverBuilderBase3<T>::withMass(double mass) {
    _mass = mass;
    return static_cast<T&>(*this);
}

//!
//! \brief Front-end to create ParticleSystemSolver3 objects step by step.
//!
class ParticleSystemSolver3::Builder final
    : public ParticleSystemSolverBuilderBase3<ParticleSystemSolver3::Builder> {
 public:
    //! Builds ParticleSystemSolver3.
    ParticleSystemSolver3 build() const;

    //! Builds shared pointer of ParticleSystemSolver3 instance.
    ParticleSystemSolver3Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_SOLVER3_H_
