// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_LEVEL_SET_LIQUID_SOLVER3_H_
#define INCLUDE_JET_LEVEL_SET_LIQUID_SOLVER3_H_

#include <jet/grid_fluid_solver3.h>
#include <jet/level_set_solver3.h>

namespace jet {

//!
//! \brief      Level set based 3-D liquid solver.
//!
//! This class implements level set-based 3-D liquid solver. It defines the
//! surface of the liquid using signed-distance field and use stable fluids
//! framework to compute the forces.
//!
//! \see Enright, Douglas, Stephen Marschner, and Ronald Fedkiw.
//!     "Animation and rendering of complex water surfaces." ACM Transactions on
//!     Graphics (TOG). Vol. 21. No. 3. ACM, 2002.
//!
class LevelSetLiquidSolver3 : public GridFluidSolver3 {
 public:
    class Builder;

    //! Default constructor.
    LevelSetLiquidSolver3();

    //! Constructs solver with initial grid size.
    LevelSetLiquidSolver3(
        const Size3& resolution,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin);

    //! Destructor.
    virtual ~LevelSetLiquidSolver3();

    //! Returns signed-distance field.
    ScalarGrid3Ptr signedDistanceField() const;

    //! Returns the level set solver.
    LevelSetSolver3Ptr levelSetSolver() const;

    //! Sets the level set solver.
    void setLevelSetSolver(const LevelSetSolver3Ptr& newSolver);

    //! Sets minimum reinitialization distance.
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

    //! Returns builder fox LevelSetLiquidSolver3.
    static Builder builder();

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

//! Shared pointer type for the LevelSetLiquidSolver3.
typedef std::shared_ptr<LevelSetLiquidSolver3> LevelSetLiquidSolver3Ptr;


//!
//! \brief Front-end to create LevelSetLiquidSolver3 objects step by step.
//!
class LevelSetLiquidSolver3::Builder final
    : public GridFluidSolverBuilderBase3<LevelSetLiquidSolver3::Builder> {
 public:
    //! Builds LevelSetLiquidSolver3.
    LevelSetLiquidSolver3 build() const;

    //! Builds shared pointer of LevelSetLiquidSolver3 instance.
    LevelSetLiquidSolver3Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_LEVEL_SET_LIQUID_SOLVER3_H_
