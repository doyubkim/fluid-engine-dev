// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PCI_SPH_SOLVER3_H_
#define INCLUDE_JET_CUDA_PCI_SPH_SOLVER3_H_

#include <jet/cuda_sph_solver_base3.h>

namespace jet {

namespace experimental {

//!
//! \brief CUDA-based 3-D PCISPH solver.
//!
//! This class implements 3-D predictive-corrective SPH solver. The main
//! pressure solver is based on Solenthaler and Pajarola's 2009 SIGGRAPH paper.
//!
//! \see Solenthaler and Pajarola, Predictive-corrective incompressible SPH,
//!      ACM transactions on graphics (TOG). Vol. 28. No. 3. ACM, 2009.
//!
class CudaPciSphSolver3 : public CudaSphSolverBase3 {
 public:
    class Builder;

    //! Constructs a solver with empty particle set.
    CudaPciSphSolver3();

    //! Constructs a solver with target density, spacing, and relative kernel
    //! radius.
    CudaPciSphSolver3(float targetDensity, float targetSpacing,
                      float relativeKernelRadius);

    //! Destructor.
    virtual ~CudaPciSphSolver3();

    //! Returns max allowed density error ratio.
    float maxDensityErrorRatio() const;

    //!
    //! \brief Sets max allowed density error ratio.
    //!
    //! This function sets the max allowed density error ratio during the PCISPH
    //! iteration. Default is 0.01 (1%). The input value should be positive.
    //!
    void setMaxDensityErrorRatio(float ratio);

    //! Returns max number of iterations.
    unsigned int maxNumberOfIterations() const;

    //!
    //! \brief Sets max number of PCISPH iterations.
    //!
    //! This function sets the max number of PCISPH iterations. Default is 5.
    //!
    void setMaxNumberOfIterations(unsigned int n);

    //! Returns builder fox CudaParticleSystemSolver3.
    static Builder builder();

 protected:
    //! Returns the number of sub-time-steps.
    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override;

    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    CudaArrayView1<float4> tempPositions() const;

    CudaArrayView1<float4> tempVelocities() const;

    CudaArrayView1<float> tempDensities() const;

    CudaArrayView1<float4> pressureForces() const;

    CudaArrayView1<float> densityErrors() const;

 private:
    float _maxDensityErrorRatio = 0.01;
    unsigned int _maxNumberOfIterations = 5;

    size_t _tempPositionsIdx;
    size_t _tempVelocitiesIdx;
    size_t _tempDensitiesIdx;
    size_t _pressureForcesIdx;
    size_t _densityErrorsIdx;

    float computeDelta(float timeStepInSeconds);
    float computeBeta(float timeStepInSeconds);
};

//! Shared pointer type for the CudaPciSphSolver3.
typedef std::shared_ptr<CudaPciSphSolver3> CudaPciSphSolver3Ptr;

//!
//! \brief Front-end to create CudaPciSphSolver3 objects step by step.
//!
class CudaPciSphSolver3::Builder final
    : public CudaSphSolverBuilderBase3<CudaPciSphSolver3::Builder> {
 public:
    //! Builds CudaPciSphSolver3.
    CudaPciSphSolver3 build() const;

    //! Builds shared pointer of CudaPciSphSolver3 instance.
    CudaPciSphSolver3Ptr makeShared() const;
};

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PCI_SPH_SOLVER3_H_

#endif  // JET_USE_CUDA
