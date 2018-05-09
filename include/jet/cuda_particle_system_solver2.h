// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER2_H_
#define INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER2_H_

#include <jet/cuda_particle_system_solver_base2.h>

namespace jet {

//!
//! \brief CUDA-based basic 2-D particle system solver.
//!
//! This class implements basic particle system solver with CUDA. It includes
//! gravity, air drag, and collision. But it does not compute
//! particle-to-particle interaction. Thus, this solver is suitable for
//! performing simple spray-like simulations with low computational cost. This
//! class can be further extend to add more sophisticated simulations, such as
//! SPH, to handle particle-to-particle intersection.
//!
//! \see ParticleSystemSolver2
//! \see SphSolver2
//!
class CudaParticleSystemSolver2 : public CudaParticleSystemSolverBase2 {
 public:
    class Builder;

    //! Constructs an empty solver.
    CudaParticleSystemSolver2();

    //! Constructs a solver with particle parameters.
    CudaParticleSystemSolver2(float radius, float mass);

    //! Destructor.
    virtual ~CudaParticleSystemSolver2();

    //! Radius of a particle.
    float radius() const;

    //! Sets the radius of a particle.
    void setRadius(float newRadius);

    //! Mass of a particle.
    float mass() const;

    //! Sets the mass of a particle.
    void setMass(float newMass);

    //! Returns builder fox CudaParticleSystemSolver2.
    static Builder builder();

 protected:
    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

 private:
    float _radius = 1e-3f;
    float _mass = 1e-3f;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);
};

//! Shared pointer type for the CudaParticleSystemSolver2.
typedef std::shared_ptr<CudaParticleSystemSolver2> CudaParticleSystemSolver2Ptr;

//!
//! \brief Front-end to create CudaParticleSystemSolver2 objects step by step.
//!
class CudaParticleSystemSolver2::Builder final
    : public CudaParticleSystemSolverBuilderBase2<
          CudaParticleSystemSolver2::Builder> {
 public:
    //! Returns builder with particle radius.
    CudaParticleSystemSolver2::Builder& withRadius(float radius);

    //! Returns builder with mass per particle.
    CudaParticleSystemSolver2::Builder& withMass(float mass);

    //! Builds CudaParticleSystemSolver2.
    CudaParticleSystemSolver2 build() const;

    //! Builds shared pointer of CudaParticleSystemSolver2 instance.
    CudaParticleSystemSolver2Ptr makeShared() const;

 private:
    float _radius = 1e-3f;
    float _mass = 1e-3f;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PARTICLE_SYSTEM_SOLVER2_H_

#endif  // JET_USE_CUDA
