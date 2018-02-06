// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

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

    //! Radius of a particle.
    float radius() const;

    //! Sets the radius of a particle.
    void setRadius(float newRadius);

    //! Mass of a particle.
    float mass() const;

    //! Sets the mass of a particle.
    void setMass(float newMass);

    //! The amount of air-drag.
    float dragCoefficient() const;

    //!
    //! \brief Sets the drag coefficient.
    //!
    //! The coefficient should be a positive number and 0 means no drag force.
    //!
    //! \param newDragCoefficient The new drag coefficient.
    //!
    void setDragCoefficient(float newDragCoefficient);

    //!
    //! \brief The restitution coefficient.
    //!
    //! The restitution coefficient controls the bouncy-ness of a particle when
    //! it hits a collider surface. 0 means no bounce back and 1 means perfect
    //! reflection.
    //!
    float restitutionCoefficient() const;

    //!
    //! \brief Sets the restitution coefficient.
    //!
    //! The range of the coefficient should be 0 to 1 -- 0 means no bounce back
    //! and 1 means perfect reflection.
    //!
    //! \param newRestitutionCoefficient The new restitution coefficient.
    //!
    void setRestitutionCoefficient(float newRestitutionCoefficient);

    //! Returns the gravity.
    const Vector3F& gravity() const;

    //! Sets the gravity.
    void setGravity(const Vector3F& newGravity);

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    const CudaParticleSystemData3Ptr& particleSystemData() const;

    //! Returns builder fox CudaParticleSystemSolver3.
    static Builder builder();

 protected:
    //! Initializes the simulator.
    void onInitialize() override;

    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    virtual void onBeginAdvanceTimeStep(double timeStepInSeconds);

    virtual void onEndAdvanceTimeStep(double timeStepInSeconds);

 private:
    float _radius = 1e-3f;
    float _mass = 1e-3f;
    float _dragCoefficient = 1e-4f;
    float _restitutionCoefficient = 0.0f;
    Vector3F _gravity{0.0f, static_cast<float>(kGravity), 0.0f};

    CudaParticleSystemData3Ptr _particleSystemData;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void updateCollider(double timeStepInSeconds);

    void updateEmitter(double timeStepInSeconds);
};

//! Shared pointer type for the CudaParticleSystemSolver3.
typedef std::shared_ptr<CudaParticleSystemSolver3> CudaParticleSystemSolver3Ptr;

//!
//! \brief Front-end to create CudaParticleSystemSolver3 objects step by step.
//!
class CudaParticleSystemSolver3::Builder final {
 public:
    //! Returns builder with particle radius.
    CudaParticleSystemSolver3::Builder& withRadius(float radius);

    //! Returns builder with mass per particle.
    CudaParticleSystemSolver3::Builder& withMass(float mass);

    //! Builds CudaParticleSystemSolver3.
    CudaParticleSystemSolver3 build() const;

    //! Builds shared pointer of CudaParticleSystemSolver3 instance.
    CudaParticleSystemSolver3Ptr makeShared() const;

 private:
    float _radius = 1e-3f;
    float _mass = 1e-3f;
};

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER3_H_

#endif  // JET_USE_CUDA
