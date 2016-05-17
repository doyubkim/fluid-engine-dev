// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PCI_SPH_SOLVER3_H_
#define INCLUDE_JET_PCI_SPH_SOLVER3_H_

#include <jet/sph_solver3.h>

namespace jet {

class PciSphSolver3 : public SphSolver3 {
 public:
    PciSphSolver3();

    virtual ~PciSphSolver3();

    double maxDensityErrorRatio() const;

    void setMaxDensityErrorRatio(double ratio);

    unsigned int maxNumberOfIterations() const;

    void setMaxNumberOfIterations(unsigned int n);

 protected:
    void accumulatePressureForce(double timeIntervalInSeconds) override;

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
