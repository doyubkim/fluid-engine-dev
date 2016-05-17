// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_SMOKE_SOLVER3_H_
#define INCLUDE_JET_GRID_SMOKE_SOLVER3_H_

#include <jet/grid_fluid_solver3.h>

namespace jet {

class GridSmokeSolver3 : public GridFluidSolver3 {
 public:
    GridSmokeSolver3();

    virtual ~GridSmokeSolver3();

    double smokeDiffusionCoefficient() const;

    void setSmokeDiffusionCoefficient(double newValue);

    double temperatureDiffusionCoefficient() const;

    void setTemperatureDiffusionCoefficient(double newValue);

    double buoyancySmokeDensityFactor() const;

    void setBuoyancySmokeDensityFactor(double newValue);

    double buoyancyTemperatureFactor() const;

    void setBuoyancyTemperatureFactor(double newValue);

    ScalarGrid3Ptr smokeDensity() const;

    ScalarGrid3Ptr temperature() const;

 protected:
    void onEndAdvanceTimeStep(double timeIntervalInSeconds) override;

    void computeExternalForces(double timeIntervalInSeconds) override;

 private:
    size_t _smokeDensityDataId;
    size_t _temperatureDataId;
    double _smokeDiffusionCoefficient = 0.0;
    double _temperatureDiffusionCoefficient = 0.0;
    double _buoyancySmokeDensityFactor = -0.000625;
    double _buoyancyTemperatureFactor = 5.0;

    void computeDiffusion(double timeIntervalInSeconds);

    void computeBuoyancyForce(double timeIntervalInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SMOKE_SOLVER3_H_
