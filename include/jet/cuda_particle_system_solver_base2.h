// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER_BASE2_H_
#define INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER_BASE2_H_

#include <jet/constants.h>
#include <jet/cuda_particle_system_data2.h>
#include <jet/physics_animation.h>
#include <jet/vector2.h>

namespace jet {

//!
//! \brief Abstract base class for CUDA-based 2-D particle solvers.
//!
//! This class defines commonly used particle-related interface for CUDA-based
//! particle solvers.
//!
class CudaParticleSystemSolverBase2 : public PhysicsAnimation {
 public:
    //! Constructs an empty solver.
    CudaParticleSystemSolverBase2();

    //! Destructor.
    virtual ~CudaParticleSystemSolverBase2();

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
    const Vector2F& gravity() const;

    //! Sets the gravity.
    void setGravity(const Vector2F& newGravity);

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    virtual CudaParticleSystemData2* particleSystemData();

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    virtual const CudaParticleSystemData2* particleSystemData() const;

 protected:
    //! Initializes the simulator.
    void onInitialize() override;

    void updateCollider(double timeStepInSeconds);

    void updateEmitter(double timeStepInSeconds);

 private:
    float _dragCoefficient = 1e-4f;
    float _restitutionCoefficient = 0.0f;
    Vector2F _gravity{0.0f, kGravityF};
    CudaParticleSystemData2Ptr _particleSystemData;
};

//! Shared pointer type for the CudaParticleSystemSolverBase2.
typedef std::shared_ptr<CudaParticleSystemSolverBase2>
    CudaParticleSystemSolverBase2Ptr;

//!
template <typename DerivedBuilder>
class CudaParticleSystemSolverBuilderBase2 {
 public:
    DerivedBuilder& withDragCoefficient(float coeff);

    DerivedBuilder& withRestitutionCoefficient(float coeff);

    DerivedBuilder& withGravity(const Vector2F& gravity);

 protected:
    float _dragCoefficient = 1e-4f;
    float _restitutionCoefficient = 0.0f;
    Vector2F _gravity{0.0f, kGravityF, 0.0f};
};

template <typename T>
T& CudaParticleSystemSolverBuilderBase2<T>::withDragCoefficient(float coeff) {
    _dragCoefficient = coeff;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaParticleSystemSolverBuilderBase2<T>::withRestitutionCoefficient(
    float coeff) {
    _restitutionCoefficient = coeff;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaParticleSystemSolverBuilderBase2<T>::withGravity(
    const Vector2F& gravity) {
    _gravity = gravity;
    return static_cast<T&>(*this);
}

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER_BASE2_H_

#endif  // JET_USE_CUDA
