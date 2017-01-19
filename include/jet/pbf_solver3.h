// Copyright (c) 2017 Doyub Kim

#ifndef INCLUDE_JET_PBF_SOLVER3_H_
#define INCLUDE_JET_PBF_SOLVER3_H_

#include <jet/constants.h>
#include <jet/particle_system_solver3.h>
#include <jet/sph_system_data3.h>

namespace jet {

//!
//! \brief 3-D PBD-based fluid solver.
//!
//! This class implements 3-D position based dynamics (PBD) fluid solver.
//!
//! \see Müller, Matthias, et al. "Position based dynamics."
//!      Journal of Visual Communication and Image Representation 18.2 (2007):
//!      109-118.
//! \see Macklin, Miles, and Matthias Müller. "Position based fluids."
//!      ACM Transactions on Graphics (TOG) 32.4 (2013): 104.
//!
class PbfSolver3 final : public ParticleSystemSolver3 {
 public:
    class Builder;

    //! Constructs a solver with empty particle set.
    PbfSolver3();

    //! Constructs a solver with target density, spacing, and relative kernel
    //! radius.
    PbfSolver3(
        double targetDensity,
        double targetSpacing,
        double relativeKernelRadius);

    virtual ~PbfSolver3();

    //! Returns the pseudo viscosity coefficient.
    double pseudoViscosityCoefficient() const;

    //!
    //! \brief Sets the pseudo viscosity coefficient.
    //!
    //! This function sets the pseudo viscosity coefficient which applies
    //! additional pseudo-physical damping to the system. Default is 0.1.
    //! Should be in range between 0 and 1.
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

    //!
    //! \brief Sets the relaxation parameter.
    //!
    //! This function sets the relaxation parameter which is used when computing
    //! the step size (lambda) of the optimization process. Default is 0.1.
    //!
    //! \param[in]  eps   The relaxation parameter.
    //!
    void setLambdaRelaxation(double eps);

    //! Returns the relaxation parameter
    double lambdaRelaxation() const;

    //!
    //! \brief Sets the anti-clustering denominator factor.
    //!
    //! This function sets the denominator factor of the anti-clustering force.
    //! Default is 0.2. Parameters between 0.1 and 0.3 are recommended by the
    //! original paper.
    //!
    //! \param[in]  factor The anti-clustering denominator factor.
    //!
    void setAntiClusteringDenominatorFactor(double factor);

    //! Returns the anti-clustering denominator factor.
    double antiClusteringDenominatorFactor() const;

    //!
    //! \brief Sets the strength of the anti-clustering.
    //!
    //! This function sets the strength of the anti-clustering force.
    //! Default is 1e-5. Should be a small number.
    //!
    //! \param[in]  strength The anti-clustering strength.
    //!
    void setAntiClusteringStrength(double strength);

    //! Returns the strength of the anti-clustering.
    double antiClusteringStrength() const;

    //!
    //! \brief Sets the exponent of the anti-clustering.
    //!
    //! This function sets the exponent of the anti-clustering force.
    //! Default is 4 which is the recommended value from the paper.
    //!
    //! \param[in]  exponent The anti-clustering exponent.
    //!
    void setAntiClusteringExponent(double exponent);

    //! Returns the exponent of the anti-clustering.
    double antiClusteringExponent() const;

    //! Returns the SPH system data.
    SphSystemData3Ptr sphSystemData() const;

    //! Returns builder fox PbfSolver3.
    static Builder builder();

 private:
    double _pseudoViscosityCoefficient = 0.1;
    unsigned int _maxNumberOfIterations = 10;
    double _lambdaRelaxation = 0.1;
    double _antiClusteringDenom = 0.2;
    double _antiClusteringStrength = 1e-5;
    double _antiClusteringExp = 4.0;

    ParticleSystemData3::VectorData _originalPositions;

    void onAdvanceTimeStep(double timeStepInSeconds) override;

    void predictPosition(double timeStepInSeconds);

    void updatePosition(double timeStepInSeconds);

    void computePseudoViscosity(double timeStepInSeconds);
};

//! Shared pointer type for the PbfSolver3.
typedef std::shared_ptr<PbfSolver3> PbfSolver3Ptr;

//!
//! \brief Front-end to create PbfSolver3 objects step by step.
//!
class PbfSolver3::Builder final {
 public:
    //! Returns builder with target density.
    Builder& withTargetDensity(double targetDensity);

    //! Returns builder with target spacing.
    Builder& withTargetSpacing(double targetSpacing);

    //! Returns builder with relative kernel radius.
    Builder& withRelativeKernelRadius(double relativeKernelRadius);

    //! Builds PbfSolver3.
    PbfSolver3 build() const;

    //! Builds shared pointer of PbfSolver3 instance.
    PbfSolver3Ptr makeShared() const;

 private:
    double _targetDensity = kWaterDensity;
    double _targetSpacing = 0.1;
    double _relativeKernelRadius = 1.8;
};

}  // namespace jet

#endif  // INCLUDE_JET_PBF_SOLVER3_H_
