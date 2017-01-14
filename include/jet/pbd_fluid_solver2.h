// Copyright (c) 2017 Doyub Kim

#ifndef INCLUDE_JET_PBD_FLUID_SOLVER2_H_
#define INCLUDE_JET_PBD_FLUID_SOLVER2_H_

#include <jet/constants.h>
#include <jet/particle_system_solver2.h>
#include <jet/sph_system_data2.h>

namespace jet {

class PbdFluidSolver2 final : public ParticleSystemSolver2 {
 public:
    class Builder;

    //! Constructs a solver with empty particle set.
    PbdFluidSolver2();

    //! Constructs a solver with target density, spacing, and relative kernel
    //! radius.
    PbdFluidSolver2(
        double targetDensity,
        double targetSpacing,
        double relativeKernelRadius);

    virtual ~PbdFluidSolver2();

    //! Returns the pseudo viscosity coefficient.
    double pseudoViscosityCoefficient() const;

    //!
    //! \brief Sets the pseudo viscosity coefficient.
    //!
    //! This function sets the pseudo viscosity coefficient which applies
    //! additional pseudo-physical damping to the system. Default is 0.1.
    //!
    void setPseudoViscosityCoefficient(double newPseudoViscosityCoefficient);

    //! Returns max number of iterations.
    unsigned int maxNumberOfIterations() const;

    //!
    //! \brief Sets max number of PBD iterations.
    //!
    //! This function sets the max number of PBD iterations. Default is 5.
    //!
    void setMaxNumberOfIterations(unsigned int n);

    //! Returns the SPH system data.
    SphSystemData2Ptr sphSystemData() const;

    //! Returns builder fox PbdFluidSolver2.
    static Builder builder();

 private:
    double _pseudoViscosityCoefficient = 0.1;
    unsigned int _maxNumberOfIterations = 10;

    ParticleSystemData2::VectorData _originalPositions;

    void onAdvanceTimeStep(double timeStepInSeconds) override;

    void onEndAdvanceTimeStep(double timeStepInSeconds) override;

    void predictPosition(double timeStepInSeconds);

    void updatePosition(double timeStepInSeconds);

    void computePseudoViscosity(double timeStepInSeconds);
};

//! Shared pointer type for the PbdFluidSolver2.
typedef std::shared_ptr<PbdFluidSolver2> PbdFluidSolver2Ptr;

class PbdFluidSolver2::Builder final {
 public:
    //! Returns builder with target density.
    Builder& withTargetDensity(double targetDensity);

    //! Returns builder with target spacing.
    Builder& withTargetSpacing(double targetSpacing);

    //! Returns builder with relative kernel radius.
    Builder& withRelativeKernelRadius(double relativeKernelRadius);

    //! Builds PbdFluidSolver2.
    PbdFluidSolver2 build() const;

    //! Builds shared pointer of PbdFluidSolver2 instance.
    PbdFluidSolver2Ptr makeShared() const;

 private:
    double _targetDensity = kWaterDensity;
    double _targetSpacing = 0.1;
    double _relativeKernelRadius = 1.8;
};

}  // namespace jet

#endif  // INCLUDE_JET_PBD_FLUID_SOLVER2_H_
