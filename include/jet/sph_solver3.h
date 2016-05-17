// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SPH_SOLVER3_H_
#define INCLUDE_JET_SPH_SOLVER3_H_

#include <jet/constants.h>
#include <jet/particle_system_solver3.h>
#include <jet/sph_system_data3.h>

namespace jet {

class SphSolver3 : public ParticleSystemSolver3 {
 public:
    SphSolver3();

    virtual ~SphSolver3();

    double eosScale() const;

    void setEosScale(double newEosScale);

    double eosExponent() const;

    void setEosExponent(double newEosExponent);

    double negativePressureScale() const;

    void setNegativePressureScale(double newNegativePressureScale);

    double viscosityCoefficient() const;

    void setViscosityCoefficient(double newViscosityCoefficient);

    double pseudoViscosityCoefficient() const;

    void setPseudoViscosityCoefficient(double newPseudoViscosityCoefficient);

    double speedOfSound() const;

    void setSpeedOfSound(double newSpeedOfSound);

    double timeStepLimitScale() const;

    void setTimeStepLimitScale(double newScale);

    SphSystemData3Ptr sphSystemData() const;

 protected:
    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override;

    void accumulateForces(double timeStepInSeconds) override;

    void onBeginAdvanceTimeStep(double timeStepInSeconds) override;

    void onEndAdvanceTimeStep(double timeStepInSeconds) override;

    virtual void accumulateNonPressureForces(double timeStepInSeconds);

    virtual void accumulatePressureForce(double timeStepInSeconds);

    void computePressure();

    void accumulatePressureForce(
        const ConstArrayAccessor1<Vector3D>& positions,
        const ConstArrayAccessor1<double>& densities,
        const ConstArrayAccessor1<double>& pressures,
        ArrayAccessor1<Vector3D> pressureForces);

    void accumulateViscosityForce();

    void computePseudoViscosity(double timeStepInSeconds);

 private:
    //! Scale factor of equation-of-state (kappa).
    double _eosScale = 500.0;

    //! Exponent component of equation-of-state (gamma).
    double _eosExponent = 1.0;

    //! Negative pressure scaling factor.
    //! Zero means clamping. One means do nothing.
    double _negativePressureScale = 0.0;

    //! Viscosity coefficient.
    double _viscosityCoefficient = 0.001;

    //! Pseudo-viscosity coefficient velocity filtering.
    //! This is a minimum "safety-net" for SPH solver which is quite
    //! sensitive to the parameters.
    double _pseudoViscosityCoefficient = 500.0;

    //! Speed of sound in medium. Default value is the speed of sound in
    //! water at 20 degrees celcius.
    double _speedOfSound = kSpeedOfSoundInWater;

    //! Scales the max allowed time-step.
    double _timeStepLimitScale = 1.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_SPH_SOLVER3_H_
