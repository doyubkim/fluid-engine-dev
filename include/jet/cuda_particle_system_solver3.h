// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef __CUDACC__

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER3_H_
#define INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER3_H_

#include <jet/constants.h>
#include <jet/cuda_particle_system_data3.h>
#include <jet/physics_animation.h>
#include <jet/vector3.h>

namespace jet {

namespace experimental {

//!
//! \brief CUDA-based basic 3-D particle system solver.
//!
//! This class implements basic particle system solver with CUDA. It includes
//! gravity, air drag, and collision. But it does not compute
//! particle-to-particle interaction. Thus, this solver is suitable for
//! performing simple spray-like simulations with low computational cost. This
//! class can be further extend to add more sophisticated simulations, such as
//! SPH, to handle particle-to-particle intersection.
//!
//! \see ParticleSystemSolver3
//! \see SphSolver3
//!
class CudaParticleSystemSolver3 : public PhysicsAnimation {
 public:
    class Builder;

    //! Constructs an empty solver.
    CudaParticleSystemSolver3();

    //! Constructs a solver with particle parameters.
    CudaParticleSystemSolver3(float radius, float mass);

    //! Destructor.
    virtual ~CudaParticleSystemSolver3();

    //! Returns the drag coefficient.
    float dragCoefficient() const;

    //!
    //! \brief      Sets the drag coefficient.
    //!
    //! The drag coefficient controls the amount of air-drag. The coefficient
    //! should be a positive number and 0 means no drag force.
    //!
    //! \param[in]  newDragCoefficient The new drag coefficient.
    //!
    void setDragCoefficient(float newDragCoefficient);

    //! Sets the restitution coefficient.
    float restitutionCoefficient() const;

    //!
    //! \brief      Sets the restitution coefficient.
    //!
    //! The restitution coefficient controls the bouncy-ness of a particle when
    //! it hits a collider surface. The range of the coefficient should be 0 to
    //! 1 -- 0 means no bounce back and 1 means perfect reflection.
    //!
    //! \param[in]  newRestitutionCoefficient The new restitution coefficient.
    //!
    void setRestitutionCoefficient(float newRestitutionCoefficient);

    //! Returns the gravity.
    const Vector3F& gravity() const;

    //! Sets the gravity.
    void setGravity(const Vector3F& newGravity);

    //!
    //! \brief      Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    //! \return     The particle system data.
    //!
    const CudaParticleSystemData3Ptr& particleSystemData() const;

    //! Returns builder fox CudaParticleSystemSolver3.
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
    void resolveCollision(ArrayView1<Vector4F> newPositions,
                          ArrayView1<Vector4F> newVelocities);

 private:
    float _radius = 1e-3f;
    float _mass = 1e-3f;
    float _dragCoefficient = 1e-4f;
    float _restitutionCoefficient = 0.0f;
    Vector3F _gravity{0.0, static_cast<float>(kGravity), 0.0};

    size_t _forcesIdx;
    size_t _newPositionsIdx;
    size_t _newVelocitiesIdx;

    CudaParticleSystemData3Ptr _particleSystemData;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void accumulateExternalForces();

    void timeIntegration(double timeStepInSeconds);

    void updateCollider(double timeStepInSeconds);

    void updateEmitter(double timeStepInSeconds);
};

//! Shared pointer type for the CudaParticleSystemSolver3.
typedef std::shared_ptr<CudaParticleSystemSolver3> CudaParticleSystemSolver3Ptr;

//!
//! \brief Base class for particle-based solver builder.
//!
template <typename DerivedBuilder>
class CudaParticleSystemSolverBuilderBase3 {
 public:
    //! Returns builder with particle radius.
    DerivedBuilder& withRadius(float radius);

    //! Returns builder with mass per particle.
    DerivedBuilder& withMass(float mass);

 protected:
    float _radius = 1e-3f;
    float _mass = 1e-3f;
};

template <typename T>
T& CudaParticleSystemSolverBuilderBase3<T>::withRadius(float radius) {
    _radius = radius;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaParticleSystemSolverBuilderBase3<T>::withMass(float mass) {
    _mass = mass;
    return static_cast<T&>(*this);
}

//!
//! \brief Front-end to create CudaParticleSystemSolver3 objects step by step.
//!
class CudaParticleSystemSolver3::Builder final
    : public CudaParticleSystemSolverBuilderBase3<
          CudaParticleSystemSolver3::Builder> {
 public:
    //! Builds CudaParticleSystemSolver3.
    CudaParticleSystemSolver3 build() const;

    //! Builds shared pointer of CudaParticleSystemSolver3 instance.
    CudaParticleSystemSolver3Ptr makeShared() const;
};

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER3_H_

#endif  // JET_USE_CUDA

#endif  // __CUDACC__
