// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_PCI_SPH_SOLVER2_H_
#define INCLUDE_JET_CUDA_PCI_SPH_SOLVER2_H_

#include <jet/cuda_sph_solver_base2.h>

namespace jet {

//!
//! \brief CUDA-based 2-D PCISPH solver.
//!
//! This class implements 2-D predictive-corrective SPH solver. The main
//! pressure solver is based on Solenthaler and Pajarola's 2009 SIGGRAPH paper.
//!
//! \see Solenthaler and Pajarola, Predictive-corrective incompressible SPH,
//!      ACM transactions on graphics (TOG). Vol. 28. No. 3. ACM, 2009.
//!
class CudaPciSphSolver2 : public CudaSphSolverBase2 {
 public:
    class Builder;

    //! Constructs a solver with empty particle set.
    CudaPciSphSolver2();

    //! Constructs a solver with target density, spacing, and relative kernel
    //! radius.
    CudaPciSphSolver2(float targetDensity, float targetSpacing,
                      float relativeKernelRadius);

    //! Destructor.
    virtual ~CudaPciSphSolver2();

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

    //! Returns builder fox CudaParticleSystemSolver2.
    static Builder builder();

 protected:
    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    CudaArrayView1<float2> tempPositions();

    CudaArrayView1<float2> tempVelocities();

    CudaArrayView1<float> tempDensities();

    CudaArrayView1<float2> pressureForces();

    CudaArrayView1<float> densityErrors();

 private:
    float _maxDensityErrorRatio = 0.01f;
    unsigned int _maxNumberOfIterations = 5;

    size_t _tempPositionsIdx;
    size_t _tempVelocitiesIdx;
    size_t _tempDensitiesIdx;
    size_t _pressureForcesIdx;
    size_t _densityErrorsIdx;

    float computeDelta(float timeStepInSeconds);
    float computeBeta(float timeStepInSeconds);
};

//! Shared pointer type for the CudaPciSphSolver2.
typedef std::shared_ptr<CudaPciSphSolver2> CudaPciSphSolver2Ptr;

//!
//! \brief Front-end to create CudaPciSphSolver2 objects step by step.
//!
class CudaPciSphSolver2::Builder final
    : public CudaSphSolverBuilderBase2<CudaPciSphSolver2::Builder> {
 public:
    //! Builds CudaPciSphSolver2.
    CudaPciSphSolver2 build() const;

    //! Builds shared pointer of CudaPciSphSolver2 instance.
    CudaPciSphSolver2Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_PCI_SPH_SOLVER2_H_

#endif  // JET_USE_CUDA
