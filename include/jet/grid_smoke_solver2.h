// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_SMOKE_SOLVER2_H_
#define INCLUDE_JET_GRID_SMOKE_SOLVER2_H_

#include <jet/grid_fluid_solver2.h>

namespace jet {

//!
//! \brief      2-D grid-based smoke solver.
//!
//! This class extends GridFluidSolver2 to implement smoke simulation solver.
//! It adds smoke density and temperature fields to define the smoke and uses
//! buoyancy force to simulate hot rising smoke.
//!
//! \see Fedkiw, Ronald, Jos Stam, and Henrik Wann Jensen.
//!     "Visual simulation of smoke." Proceedings of the 28th annual conference
//!     on Computer graphics and interactive techniques. ACM, 2001.
//!
class GridSmokeSolver2 : public GridFluidSolver2 {
 public:
    class Builder;

    //! Default constructor.
    GridSmokeSolver2();

    //! Constructs solver with initial grid size.
    GridSmokeSolver2(
        const Size2& resolution,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin);

    //! Destructor.
    virtual ~GridSmokeSolver2();

    //! Returns smoke diffusion coefficient.
    double smokeDiffusionCoefficient() const;

    //! Sets smoke diffusion coefficient.
    void setSmokeDiffusionCoefficient(double newValue);

    //! Returns temperature diffusion coefficient.
    double temperatureDiffusionCoefficient() const;

    //! Sets temperature diffusion coefficient.
    void setTemperatureDiffusionCoefficient(double newValue);

    //!
    //! \brief      Returns the buoyancy factor which will be multiplied to the
    //!     smoke density.
    //!
    //! This class computes buoyancy by looking up the value of smoke density
    //! and temperature, compare them to the average values, and apply
    //! multiplier factor to the diff between the value and the average. That
    //! multiplier is defined for each smoke density and temperature separately.
    //! For example, negative smoke density buoyancy factor means a heavier
    //! smoke should sink.
    //!
    //! \return     The buoyance factor for the smoke density.
    //!
    double buoyancySmokeDensityFactor() const;

    //!
    //! \brief          Sets the buoyancy factor which will be multiplied to the
    //!     smoke density.
    //!
    //! This class computes buoyancy by looking up the value of smoke density
    //! and temperature, compare them to the average values, and apply
    //! multiplier factor to the diff between the value and the average. That
    //! multiplier is defined for each smoke density and temperature separately.
    //! For example, negative smoke density buoyancy factor means a heavier
    //! smoke should sink.
    //!
    //! \param newValue The new buoyancy factor for smoke density.
    //!
    void setBuoyancySmokeDensityFactor(double newValue);

    //!
    //! \brief      Returns the buoyancy factor which will be multiplied to the
    //!     temperature.
    //!
    //! This class computes buoyancy by looking up the value of smoke density
    //! and temperature, compare them to the average values, and apply
    //! multiplier factor to the diff between the value and the average. That
    //! multiplier is defined for each smoke density and temperature separately.
    //! For example, negative smoke density buoyancy factor means a heavier
    //! smoke should sink.
    //!
    //! \return     The buoyance factor for the temperature.
    //!
    double buoyancyTemperatureFactor() const;

    //!
    //! \brief          Sets the buoyancy factor which will be multiplied to the
    //!     temperature.
    //!
    //! This class computes buoyancy by looking up the value of smoke density
    //! and temperature, compare them to the average values, and apply
    //! multiplier factor to the diff between the value and the average. That
    //! multiplier is defined for each smoke density and temperature separately.
    //! For example, negative smoke density buoyancy factor means a heavier
    //! smoke should sink.
    //!
    //! \param newValue The new buoyancy factor for temperature.
    //!
    void setBuoyancyTemperatureFactor(double newValue);

    //!
    //! \brief      Returns smoke decay factor.
    //!
    //! In addition to the diffusion, the smoke also can fade-out over time by
    //! setting the decay factor between 0 and 1.
    //!
    //! \return     The decay factor for smoke density.
    //!
    double smokeDecayFactor() const;

    //!
    //! \brief      Sets the smoke decay factor.
    //!
    //! In addition to the diffusion, the smoke also can fade-out over time by
    //! setting the decay factor between 0 and 1.
    //!
    //! \param[in]  newValue The new decay factor.
    //!
    void setSmokeDecayFactor(double newValue);

    //!
    //! \brief      Returns temperature decay factor.
    //!
    //! In addition to the diffusion, the smoke also can fade-out over time by
    //! setting the decay factor between 0 and 1.
    //!
    //! \return     The decay factor for smoke temperature.
    //!
    double smokeTemperatureDecayFactor() const;

    //!
    //! \brief      Sets the temperature decay factor.
    //!
    //! In addition to the diffusion, the temperature also can fade-out over
    //! time by setting the decay factor between 0 and 1.
    //!
    //! \param[in]  newValue The new decay factor.
    //!
    void setTemperatureDecayFactor(double newValue);

    //! Returns smoke density field.
    ScalarGrid2Ptr smokeDensity() const;

    //! Returns temperature field.
    ScalarGrid2Ptr temperature() const;

    //! Returns builder fox GridSmokeSolver2.
    static Builder builder();

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
    double _smokeDecayFactor = 0.001;
    double _temperatureDecayFactor = 0.001;

    void computeDiffusion(double timeIntervalInSeconds);

    void computeBuoyancyForce(double timeIntervalInSeconds);
};

//! Shared pointer type for the GridSmokeSolver2.
typedef std::shared_ptr<GridSmokeSolver2> GridSmokeSolver2Ptr;


//!
//! \brief Front-end to create GridSmokeSolver2 objects step by step.
//!
class GridSmokeSolver2::Builder final
    : public GridFluidSolverBuilderBase2<GridSmokeSolver2::Builder> {
 public:
    //! Builds GridSmokeSolver2.
    GridSmokeSolver2 build() const;

    //! Builds shared pointer of GridSmokeSolver2 instance.
    GridSmokeSolver2Ptr makeShared() const;
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SMOKE_SOLVER2_H_
