// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_PCI_SPH_SOLVER2_H_
#define INCLUDE_JET_PCI_SPH_SOLVER2_H_

#include <jet/sph_solver2.h>

namespace jet {

class PciSphSolver2 : public SphSolver2 {
 public:
    PciSphSolver2();

    virtual ~PciSphSolver2();

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

    ParticleSystemData2::VectorData _tempPositions;
    ParticleSystemData2::VectorData _tempVelocities;
    ParticleSystemData2::VectorData _pressureForces;
    ParticleSystemData2::ScalarData _densityErrors;

    double computeDelta(double timeStepInSeconds);
    double computeBeta(double timeStepInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_PCI_SPH_SOLVER2_H_
