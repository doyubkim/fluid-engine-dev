// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_PARTICLE_SYSTEM_SOLVER_BASE_H_
#define INCLUDE_JET_PARTICLE_SYSTEM_SOLVER_BASE_H_

#include <jet/collider.h>
#include <jet/constants.h>
#include <jet/particle_emitter.h>
#include <jet/particle_system_data.h>
#include <jet/physics_animation.h>
#include <jet/vector_field.h>

namespace jet {

//!
//! \brief Basic N-D particle system solver.
//!
//! This class implements basic particle system solver. It includes gravity, air
//! drag, and collision. But it does not compute particle-to-particle
//! interaction. Thus, this solver is suitable for performing simple spray-like
//! simulations with low computational cost. This class can be further extend to
//! add more sophisticated simulations, such as SPH, to handle
//! particle-to-particle intersection.
//!
//! \see ParticleSystemSolver
//! \see SphSolver
//!
template <size_t N>
class ParticleSystemSolver : public PhysicsAnimation {
 public:
    //! Constructs an empty solver.
    ParticleSystemSolver();

    //! Destructor.
    virtual ~ParticleSystemSolver();

    //! The amount of air-drag.
    double dragCoefficient() const;

    //!
    //! \brief Sets the drag coefficient.
    //!
    //! The coefficient should be a positive number and 0 means no drag force.
    //!
    //! \param newDragCoefficient The new drag coefficient.
    //!
    void setDragCoefficient(double newDragCoefficient);

    //!
    //! \brief The restitution coefficient.
    //!
    //! The restitution coefficient controls the bouncy-ness of a particle when
    //! it hits a collider surface. 0 means no bounce back and 1 means perfect
    //! reflection.
    //!
    double restitutionCoefficient() const;

    //!
    //! \brief Sets the restitution coefficient.
    //!
    //! The range of the coefficient should be 0 to 1 -- 0 means no bounce back
    //! and 1 means perfect reflection.
    //!
    //! \param newRestitutionCoefficient The new restitution coefficient.
    //!
    void setRestitutionCoefficient(double newRestitutionCoefficient);

    //! Returns the gravity.
    const Vector<double, N>& gravity() const;

    //! Sets the gravity.
    void setGravity(const Vector<double, N>& newGravity);

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    virtual const ParticleSystemDataPtr<N>& particleSystemData() const;

    //! Returns the collider.
    const ColliderPtr<N>& collider() const;

    //! Sets the collider.
    void setCollider(const ColliderPtr<N>& newCollider);

    //! Returns the emitter.
    const ParticleEmitterPtr<N>& emitter() const;

    //! Sets the emitter.
    void setEmitter(const ParticleEmitterPtr<N>& newEmitter);

    //! Returns the wind field.
    const VectorFieldPtr<N>& wind() const;

    //!
    //! \brief Sets the wind.
    //!
    //! Wind can be applied to the particle system by setting a vector field to
    //! the solver.
    //!
    //! \param[in] newWind The new wind.
    //!
    void setWind(const VectorFieldPtr<N>& newWind);

 protected:
    //! Initializes the simulator.
    void onInitialize() override;

    //! Called to advance a single time-step.
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
    void resolveCollision(ArrayView1<Vector<double, N>> newPositions,
                          ArrayView1<Vector<double, N>> newVelocities);

    //! Assign a new particle system data.
    void setParticleSystemData(const ParticleSystemDataPtr<N>& newParticles);

    //! Updates the collider state to given time in seconds.
    void updateCollider(double timeStepInSeconds);

    //! Updates the emitter state to given time in seconds.
    void updateEmitter(double timeStepInSeconds);

 private:
    double _dragCoefficient = 1e-4;
    double _restitutionCoefficient = 0.0;
    Vector<double, N> _gravity = kGravityD * Vector<double, N>::makeUnitY();

    ParticleSystemDataPtr<N> _particleSystemData;
    ColliderPtr<N> _collider;
    ParticleEmitterPtr<N> _emitter;
    VectorFieldPtr<N> _wind;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void timeIntegration(double timeStepInSeconds);
};

////! 2-D ParticleSystemSolver type.
//using ParticleSystemSolver2 = ParticleSystemSolver<2>;
//
////! 3-D ParticleSystemSolver type.
//using ParticleSystemSolver3 = ParticleSystemSolver<3>;

//! N-D shared pointer type for the ParticleSystemSolver.
template <size_t N>
using ParticleSystemSolverPtr =
    std::shared_ptr<ParticleSystemSolver<N>>;

////! Shared pointer type for the ParticleSystemSolver2.
//using ParticleSystemSolver2Ptr = ParticleSystemSolverPtr<2>;
//
////! Shared pointer type for the ParticleSystemSolver3.
//using ParticleSystemSolver3Ptr = ParticleSystemSolverPtr<3>;

//!
template <size_t N, typename DerivedBuilder>
class ParticleSystemSolverBuilder {
 public:
    DerivedBuilder& withDragCoefficient(double coeff);

    DerivedBuilder& withRestitutionCoefficient(double coeff);

    DerivedBuilder& withGravity(const Vector<double, N>& gravity);

 protected:
    double _dragCoefficient = 1e-4f;
    double _restitutionCoefficient = 0.0f;
    Vector<double, N> _gravity = kGravityD * Vector<double, N>::makeUnitY();
};

template <size_t N, typename DerivedBuilder>
DerivedBuilder&
ParticleSystemSolverBuilder<N, DerivedBuilder>::withDragCoefficient(
    double coeff) {
    _dragCoefficient = coeff;
    return static_cast<DerivedBuilder&>(*this);
}

template <size_t N, typename DerivedBuilder>
DerivedBuilder&
ParticleSystemSolverBuilder<N, DerivedBuilder>::withRestitutionCoefficient(
    double coeff) {
    _restitutionCoefficient = coeff;
    return static_cast<DerivedBuilder&>(*this);
}

template <size_t N, typename DerivedBuilder>
DerivedBuilder& ParticleSystemSolverBuilder<N, DerivedBuilder>::withGravity(
    const Vector<double, N>& gravity) {
    _gravity = gravity;
    return static_cast<DerivedBuilder&>(*this);
}

}  // namespace jet

#endif  // INCLUDE_JET_PARTICLE_SYSTEM_SOLVER_BASE_H_
