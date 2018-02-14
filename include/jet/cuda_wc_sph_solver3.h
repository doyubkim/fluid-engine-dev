// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_WC_SPH_SOLVER3_H_
#define INCLUDE_JET_CUDA_WC_SPH_SOLVER3_H_

#include <jet/cuda_sph_solver_base3.h>

namespace jet {

namespace experimental {

//!
//! \brief CUDA-based 3-D WCSPH solver.
//!
//! This class implements 3-D WCSPH solver using CUDA. The main pressure solver
//! is based on equation-of-state (EOS).
//!
//! \see CudaParticleSystemSolver3
//! \see SphSolver3
//!
//! \see M{\"u}ller et al., Particle-based fluid simulation for interactive
//!      applications, SCA 2003.
//! \see M. Becker and M. Teschner, Weakly compressible SPH for free surface
//!      flows, SCA 2007.
//! \see Adams and Wicke, Meshless approximation methods and applications in
//!      physics based modeling and animation, Eurographics tutorials 2009.
//!
class CudaWcSphSolver3 : public CudaSphSolverBase3 {
 public:
    class Builder;

    //! Constructs a solver with empty particle set.
    CudaWcSphSolver3();

    //! Constructs a solver with target density, spacing, and relative kernel
    //! radius.
    CudaWcSphSolver3(float targetDensity, float targetSpacing,
                     float relativeKernelRadius);

    //! Destructor.
    virtual ~CudaWcSphSolver3();

    //! Exponent component of equation-of-state (or Tait's equation).
    float eosExponent() const;

    //!
    //! \brief Sets the exponent part of the equation-of-state.
    //!
    //! This function sets the exponent part of the equation-of-state.
    //! The value must be greater than 1.0, and smaller inputs will be clamped.
    //! Default is 7.
    //!
    void setEosExponent(float newEosExponent);

    //! Returns builder fox CudaParticleSystemSolver3.
    static Builder builder();

 protected:
    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

 private:
    // WCSPH solver properties
    float _eosExponent = 7.0f;
};

//! Shared pointer type for the CudaWcSphSolver3.
typedef std::shared_ptr<CudaWcSphSolver3> CudaWcSphSolver3Ptr;

//!
//! \brief Front-end to create CudaWcSphSolver3 objects step by step.
//!
class CudaWcSphSolver3::Builder final
    : public CudaSphSolverBuilderBase3<CudaWcSphSolver3::Builder> {
 public:
    //! Builds CudaWcSphSolver3.
    CudaWcSphSolver3 build() const;

    //! Builds shared pointer of CudaWcSphSolver3 instance.
    CudaWcSphSolver3Ptr makeShared() const;
};

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_WC_SPH_SOLVER3_H_

#endif  // JET_USE_CUDA
