// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_SPH_SOLVER_BASE3_H_
#define INCLUDE_JET_CUDA_SPH_SOLVER_BASE3_H_

#include <jet/bounding_box3.h>
#include <jet/cuda_particle_system_solver_base3.h>
#include <jet/cuda_sph_system_data3.h>

namespace jet {

namespace experimental {

//!
class CudaSphSolverBase3 : public CudaParticleSystemSolverBase3 {
 public:
    //! Constructs a solver with empty particle set.
    CudaSphSolverBase3();

    //! Destructor.
    virtual ~CudaSphSolverBase3();

    //!
    //! \brief Negative pressure scaling factor.
    //!
    //! Zero means clamping. One means do nothing.
    //!
    float negativePressureScale() const;

    //!
    //! \brief Sets the negative pressure scale.
    //!
    //! This function sets the negative pressure scale. By setting the number
    //! between 0 and 1, the solver will scale the effect of negative pressure
    //! which can prevent the clumping of the particles near the surface. Input
    //! value outside 0 and 1 will be clamped within the range. Default is 0.
    //!
    void setNegativePressureScale(float newNegativePressureScale);

    //! Returns the viscosity coefficient.
    float viscosityCoefficient() const;

    //! Sets the viscosity coefficient.
    void setViscosityCoefficient(float newViscosityCoefficient);

    //!
    //! \brief Pseudo-viscosity coefficient velocity filtering.
    //!
    //! This is a minimum "safety-net" for SPH solver which is quite
    //! sensitive to the parameters.
    //!
    float pseudoViscosityCoefficient() const;

    //! Sets the pseudo viscosity coefficient.
    void setPseudoViscosityCoefficient(float newPseudoViscosityCoefficient);

    //!
    //! \brief Speed of sound in medium to determin the stiffness of the system.
    //!
    //! Ideally, it should be the actual speed of sound in the fluid, but in
    //! practice, use lower value to trace-off performance and compressibility.
    //!
    float speedOfSound() const;

    //! Sets the speed of sound.
    void setSpeedOfSound(float newSpeedOfSound);

    //!
    //! \brief Multiplier that scales the max allowed time-step.
    //!
    //! This function returns the multiplier that scales the max allowed
    //! time-step. When the scale is 1.0, the time-step is bounded by the speed
    //! of sound and max acceleration.
    //!
    float timeStepLimitScale() const;

    //! Sets the multiplier that scales the max allowed time-step.
    void setTimeStepLimitScale(float newScale);

    //!
    //! \brief Returns the container of the simulation.
    //!
    //! This function returns the container which bounds the particles. The
    //! default size of the container is [-1000, -1000] x [1000, 1000].
    //!
    const BoundingBox3F& container() const;

    //! Sets the container of the simulation.
    void setContainer(const BoundingBox3F& cont);

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    CudaParticleSystemData3* particleSystemData() override;

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    const CudaParticleSystemData3* particleSystemData() const override;

    //!
    //! \brief Returns the SPH system data.
    //!
    //! This function returns the SPH system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    CudaSphSystemData3* sphSystemData();

    //!
    //! \brief Returns the SPH system data.
    //!
    //! This function returns the SPH system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    const CudaSphSystemData3* sphSystemData() const;

 protected:
    //! Returns the number of sub-time-steps.
    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override;

    CudaArrayView1<float4> forces() const;

    CudaArrayView1<float4> smoothedVelocities() const;

 private:
    // Basic SPH solver properties
    size_t _forcesIdx;
    size_t _smoothedVelIdx;
    float _negativePressureScale = 0.0f;
    float _viscosityCoefficient = 0.01f;
    float _pseudoViscosityCoefficient = 10.0f;
    float _speedOfSound = 100.0f;
    float _timeStepLimitScale = 1.0f;
    BoundingBox3F _container;

    // Data model
    CudaSphSystemData3Ptr _sphSystemData;
};

//! Shared pointer type for the CudaSphSolverBase3.
typedef std::shared_ptr<CudaSphSolverBase3> CudaSphSolverBase3Ptr;

//!
template <typename DerivedBuilder>
class CudaSphSolverBuilderBase3
    : public CudaParticleSystemSolverBuilderBase3<DerivedBuilder> {
 public:
    //! Returns builder with target density.
    DerivedBuilder& withTargetDensity(float targetDensity);

    //! Returns builder with target spacing.
    DerivedBuilder& withTargetSpacing(float targetSpacing);

    //! Returns builder with relative kernel radius.
    DerivedBuilder& withRelativeKernelRadius(float relativeKernelRadius);

    DerivedBuilder& withNegativePressureScale(float negativePressureScale);

    DerivedBuilder& withViscosityCoefficient(float viscosityCoefficient);

    DerivedBuilder& withPseudoViscosityCoefficient(
        float pseudoViscosityCoefficient);

 protected:
    float _targetDensity = kWaterDensityF;
    float _targetSpacing = 0.1f;
    float _relativeKernelRadius = 1.8f;
    float _negativePressureScale = 0.0f;
    float _viscosityCoefficient = 0.01f;
    float _pseudoViscosityCoefficient = 10.0f;
};

template <typename T>
T& CudaSphSolverBuilderBase3<T>::withTargetDensity(float targetDensity) {
    _targetDensity = targetDensity;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaSphSolverBuilderBase3<T>::withTargetSpacing(float targetSpacing) {
    _targetSpacing = targetSpacing;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaSphSolverBuilderBase3<T>::withRelativeKernelRadius(
    float relativeKernelRadius) {
    _relativeKernelRadius = relativeKernelRadius;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaSphSolverBuilderBase3<T>::withNegativePressureScale(
    float negativePressureScale) {
    _negativePressureScale = negativePressureScale;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaSphSolverBuilderBase3<T>::withViscosityCoefficient(
    float viscosityCoefficient) {
    _viscosityCoefficient = viscosityCoefficient;
    return static_cast<T&>(*this);
}

template <typename T>
T& CudaSphSolverBuilderBase3<T>::withPseudoViscosityCoefficient(
    float pseudoViscosityCoefficient) {
    _pseudoViscosityCoefficient = pseudoViscosityCoefficient;
    return static_cast<T&>(*this);
}

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_SPH_SOLVER_BASE3_H_

#endif  // JET_USE_CUDA
