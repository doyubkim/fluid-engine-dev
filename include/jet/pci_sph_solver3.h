// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PCI_SPH_SOLVER3_H_
#define INCLUDE_JET_PCI_SPH_SOLVER3_H_

#include <jet/sph_solver3.h>

namespace jet {

//!
//! \brief 3-D PCISPH solver.
//!
//! This class implements 3-D predictive-corrective SPH solver. The main
//! pressure solver is based on Solenthaler and Pajarola's 2009 SIGGRAPH paper.
//!
//! \see Solenthaler and Pajarola, Predictive-corrective incompressible SPH,
//!      ACM transactions on graphics (TOG). Vol. 28. No. 3. ACM, 2009.
//!
class PciSphSolver3 : public SphSolver3 {
 public:
    //! Constructs a solver with empty particle set.
    PciSphSolver3();

    virtual ~PciSphSolver3();

    //! Returns max allowed density error ratio.
    double maxDensityErrorRatio() const;

    //!
    //! \brief Sets max allowed density error ratio.
    //!
    //! This function sets the max allowed density error ratio during the PCISPH
    //! iteration. Default is 0.01 (1%). The input value should be positive.
    //!
    void setMaxDensityErrorRatio(double ratio);

    //! Returns max number of iterations.
    unsigned int maxNumberOfIterations() const;

    //!
    //! \brief Sets max number of PCISPH iterations.
    //!
    //! This function sets the max number of PCISPH iterations. Default is 5.
    //!
    void setMaxNumberOfIterations(unsigned int n);

 protected:
    //! Accumulates the pressure force to the forces array in the particle
    //! system.
    void accumulatePressureForce(double timeIntervalInSeconds) override;

    //! Performs pre-processing step before the simulation.
    void onBeginAdvanceTimeStep(double timeStepInSeconds) override;

 private:
    double _maxDensityErrorRatio = 0.01;
    unsigned int _maxNumberOfIterations = 5;

    ParticleSystemData3::VectorData _tempPositions;
    ParticleSystemData3::VectorData _tempVelocities;
    ParticleSystemData3::VectorData _pressureForces;
    ParticleSystemData3::ScalarData _densityErrors;

    double computeDelta(double timeStepInSeconds);
    double computeBeta(double timeStepInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_PCI_SPH_SOLVER3_H_
