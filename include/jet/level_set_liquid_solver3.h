// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_LEVEL_SET_LIQUID_SOLVER3_H_
#define INCLUDE_JET_LEVEL_SET_LIQUID_SOLVER3_H_

#include <jet/grid_fluid_solver3.h>
#include <jet/level_set_solver3.h>

namespace jet {

class LevelSetLiquidSolver3 : public GridFluidSolver3 {
 public:
    LevelSetLiquidSolver3();

    virtual ~LevelSetLiquidSolver3();

    ScalarGrid3Ptr signedDistanceField() const;

    LevelSetSolver3Ptr levelSetSolver() const;

    void setLevelSetSolver(const LevelSetSolver3Ptr& newSolver);

    void setMinReinitializeDistance(double distance);

    //!
    //! \brief Enables (or disables) global compensation feature flag.
    //!
    //! When \p isEnabled is true, the global compensation feature is enabled.
    //! The global compensation measures the volume at the beginning and the end
    //! of the time-step and adds the volume change back to the level-set field
    //! by globally shifting the front.
    //!
    //! \see Song, Oh-Young, Hyuncheol Shin, and Hyeong-Seok Ko.
    //! "Stable but nondissipative water." ACM Transactions on Graphics (TOG)
    //! 24, no. 1 (2005): 81-97.
    //!
    void setIsGlobalCompensationEnabled(bool isEnabled);

    //!
    //! \brief Returns liquid volume measured by smeared Heaviside function.
    //!
    //! This function measures the liquid volume using smeared Heaviside
    //! function. Thus, the estimated volume is an approximated quantity.
    //!
    double computeVolume() const;

 protected:
    //! Called at the beginning of the time-step.
    void onBeginAdvanceTimeStep(double timeIntervalInSeconds) override;

    //! Called at the end of the time-step.
    void onEndAdvanceTimeStep(double timeIntervalInSeconds) override;

    //! Customizes advection step.
    void computeAdvection(double timeIntervalInSeconds) override;

    //!
    //! \brief Returns fluid region as a signed-distance field.
    //!
    //! This function returns fluid region as a signed-distance field. For this
    //! particular class, it returns the same field as the function
    //! LevelSetLiquidSolver2::signedDistanceField().
    //!
    ScalarField3Ptr fluidSdf() const override;

 private:
    size_t _signedDistanceFieldId;
    LevelSetSolver3Ptr _levelSetSolver;
    double _minReinitializeDistance = 10.0;
    bool _isGlobalCompensationEnabled = false;
    double _lastKnownVolume = 0.0;

    void reinitialize(double currentCfl);

    void extrapolateVelocityToAir(double currentCfl);

    void addVolume(double volDiff);
};

}  // namespace jet

#endif  // INCLUDE_JET_LEVEL_SET_LIQUID_SOLVER3_H_
