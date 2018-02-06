// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_SPH_SOLVER3_H_
#define INCLUDE_JET_CUDA_SPH_SOLVER3_H_

#include <jet/constants.h>
#include <jet/cuda_sph_system_data3.h>
#include <jet/physics_animation.h>
#include <jet/vector3.h>

namespace jet {

namespace experimental {

//!
//! \brief CUDA-based 3-D SPH solver.
//!
//! This class implements 3-D SPH solver using CUDA. The main pressure solver is
//! based on equation-of-state (EOS).
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
class CudaSphSolver3 : public PhysicsAnimation {
 public:
    class Builder;

    //! Constructs a solver with empty particle set.
    CudaSphSolver3();

    //! Constructs a solver with target density, spacing, and relative kernel
    //! radius.
    CudaSphSolver3(float targetDensity, float targetSpacing,
                   float relativeKernelRadius);

    //! Destructor.
    virtual ~CudaSphSolver3();

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
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    CudaSphSystemData3* particleSystemData();

    //!
    //! \brief Returns the particle system data.
    //!
    //! This function returns the particle system data. The data is created when
    //! this solver is constructed and also owned by the solver.
    //!
    const CudaSphSystemData3* particleSystemData() const;

    //! Returns builder fox CudaParticleSystemSolver3.
    static Builder builder();

 protected:
    //! Returns the number of sub-time-steps.
    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override;

    //! Initializes the simulator.
    void onInitialize() override;

    //! Called to advane a single time-step.
    void onAdvanceTimeStep(double timeStepInSeconds) override;

    virtual void onBeginAdvanceTimeStep(double timeStepInSeconds);

    virtual void onEndAdvanceTimeStep(double timeStepInSeconds);

 private:
    // Basic particle solver properties
    float _dragCoefficient = 1e-4f;
    float _restitutionCoefficient = 0.0f;
    Vector3F _gravity{0.0f, static_cast<float>(kGravity), 0.0f};

    // Basic SPH solver properties
    float _targetDensity = static_cast<float>(kWaterDensity);
    float _targetSpacing = 0.1f;
    float _relativeKernelRadius = 1.8f;
    size_t _forcesIdx;
    size_t _smoothedVelIdx;

    // WCSPH solver properties
    float _eosExponent = 7.0f;
    float _negativePressureScale = 0.0f;
    float _viscosityCoefficient = 0.01f;
    float _pseudoViscosityCoefficient = 10.0f;
    float _speedOfSound = 100.0f;
    float _timeStepLimitScale = 1.0f;

    // Data model
    CudaSphSystemData3Ptr _sphSystemData;

    void beginAdvanceTimeStep(double timeStepInSeconds);

    void endAdvanceTimeStep(double timeStepInSeconds);

    void updateCollider(double timeStepInSeconds);

    void updateEmitter(double timeStepInSeconds);
};

//! Shared pointer type for the CudaSphSolver3.
typedef std::shared_ptr<CudaSphSolver3> CudaSphSolver3Ptr;

//!
//! \brief Front-end to create CudaSphSolver3 objects step by step.
//!
class CudaSphSolver3::Builder final {
 public:
    //! Returns builder with target density.
    CudaSphSolver3::Builder& withTargetDensity(float targetDensity);

    //! Returns builder with target spacing.
    CudaSphSolver3::Builder& withTargetSpacing(float targetSpacing);

    //! Returns builder with relative kernel radius.
    CudaSphSolver3::Builder& withRelativeKernelRadius(
        float relativeKernelRadius);

    //! Builds CudaSphSolver3.
    CudaSphSolver3 build() const;

    //! Builds shared pointer of CudaSphSolver3 instance.
    CudaSphSolver3Ptr makeShared() const;

 private:
    float _targetDensity = static_cast<float>(kWaterDensity);
    float _targetSpacing = 0.1f;
    float _relativeKernelRadius = 1.8f;
};

}  // namespace experimental

}  // namespace jet

#endif  // INCLUDE_JET_CUDA_SPH_SOLVER3_H_

#endif  // JET_USE_CUDA
