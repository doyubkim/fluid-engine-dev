// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_FLUID_SOLVER2_H_
#define INCLUDE_JET_GRID_FLUID_SOLVER2_H_

#include <jet/advection_solver2.h>
#include <jet/cell_centered_scalar_grid2.h>
#include <jet/collider2.h>
#include <jet/face_centered_grid2.h>
#include <jet/grid_boundary_condition_solver2.h>
#include <jet/grid_diffusion_solver2.h>
#include <jet/grid_pressure_solver2.h>
#include <jet/grid_system_data2.h>
#include <jet/physics_animation.h>

namespace jet {

//!
//! \brief Abstract base class for grid-based 2-D fluid solver.
//!
//! This is an abstract base class for grid-based 2-D fluid solver based on
//! Jos Stam's famous 1999 paper - "Stable Fluids". This solver takes fractional
//! step method as its foundation which is consisted of independant advection,
//! diffusion, external forces, and pressure projection steps. Each step is
//! configurable so that a custom step can be implemented. For example, if a
//! user wants to change the advection solver to her/his own implementation,
//! simply call GridFluidSolver2::setAdvectionSolver(newSolver).
//!
class GridFluidSolver2 : public PhysicsAnimation {
 public:
    //! Default constructor.
    GridFluidSolver2();

    //! Default destructor.
    virtual ~GridFluidSolver2();

    //! Returns the gravity vector of the system.
    const Vector2D& gravity() const;

    //! Sets the gravity of the system.
    void setGravity(const Vector2D& newGravity);

    //! Returns the viscosity coefficient.
    double viscosityCoefficient() const;

    //!
    //! \brief Sets the viscosity coefficient.
    //!
    //! This function sets the viscosity coefficient. Non-positive input will be
    //! clamped to zero.
    //!
    //! \param[in] newValue The new viscosity coefficient value
    //!
    void setViscosityCoefficient(double newValue);

    //!
    //! \brief Returns the CFL number from the current velocity field for given
    //!     time interval.
    //!
    //! \param[in] timeIntervalInSeconds The time interval in seconds
    //!
    double cfl(double timeIntervalInSeconds) const;

    //! Returns the max allowed CFL number.
    double maxCfl() const;

    //! Sets the max allowed CFL number.
    void setMaxCfl(double newCfl);

    //! Returns the advection solver instance.
    const AdvectionSolver2Ptr& advectionSolver() const;

    //! Sets the advection solver.
    void setAdvectionSolver(const AdvectionSolver2Ptr& newSolver);

    //! Returns the diffusion solver instance.
    const GridDiffusionSolver2Ptr& diffusionSolver() const;

    //! Sets the diffusion solver.
    void setDiffusionSolver(const GridDiffusionSolver2Ptr& newSolver);

    //! Returns the pressure solver instance.
    const GridPressureSolver2Ptr& pressureSolver() const;

    //! Sets the pressure solver.
    void setPressureSolver(const GridPressureSolver2Ptr& newSolver);

    //! Returns the closed domain boundary flag.
    int closedDomainBoundaryFlag() const;

    //! Sets the closed domain boundary flag.
    void setClosedDomainBoundaryFlag(int flag);

    //!
    //! \brief Returns the grid system data.
    //!
    //! This function returns the grid system data. The grid system data stores
    //! the core fluid flow fields such as velocity. By default, the data
    //! instance has velocity field only.
    //!
    //! \see GridSystemData2
    //!
    const GridSystemData2Ptr& gridSystemData() const;

    //!
    //! \brief Returns the velocity field.
    //!
    //! This function returns the velocity field from the grid system data.
    //! It is just a shortcut to the most commonly accessed data chunk.
    //!
    const FaceCenteredGrid2Ptr& velocity() const;

    //! Returns the collider.
    const Collider2Ptr& collider() const;

    //! Sets the collider.
    void setCollider(const Collider2Ptr& newCollider);

 protected:
    //! Called when advancing a single time-step.
    void onAdvanceTimeStep(double timeIntervalInSeconds) override;

    //!
    //! \brief Returns the required sub-time-steps for given time interval.
    //!
    //! This function returns the required sub-time-steps for given time
    //! interval based on the max allowed CFL number. If the time interval is
    //! too large so that it makes the CFL number greater than the max value,
    //! This function will return a numebr that is greater than 1.
    //!
    //! \see GridFluidSolver2::maxCfl
    //!
    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override;

    //! Called at the beginning of a time-step.
    virtual void onBeginAdvanceTimeStep(double timeIntervalInSeconds);

    //! Called at the end of a time-step.
    virtual void onEndAdvanceTimeStep(double timeIntervalInSeconds);

    //!
    //! \brief Computes the external force terms.
    //!
    //! This function computes the external force applied for given time
    //! interval. By default, it only computes the gravity.
    //!
    //! \see GridFluidSolver2::computeGravity
    //!
    virtual void computeExternalForces(double timeIntervalInSeconds);

    //! Computes the viscosity term using the diffusion solver.
    virtual void computeViscosity(double timeIntervalInSeconds);

    //! Computes the pressure term using the pressure solver.
    virtual void computePressure(double timeIntervalInSeconds);

    //! Computes the advection term using the advection solver.
    virtual void computeAdvection(double timeIntervalInSeconds);

    //!
    //! \breif Returns the signed-distance representation of the fluid.
    //!
    //! This function returns the signed-distance representation of the fluid.
    //! Positive sign area is considered to be atmosphere and won't be included
    //! for computing the dynamics. By default, this will return constant scalar
    //! field of -kMaxD, meaning that the entire volume is occupied with fluid.
    //!
    virtual ScalarField2Ptr fluidSdf() const;

    //! Computes the gravity term.
    void computeGravity(double timeIntervalInSeconds);

    //!
    //! \brief Applies the boundary condition to the velocity field.
    //!
    //! This function applies the boundary condition to the velocity field by
    //! constraining the flow based on the boundary condition solver.
    //!
    void applyBoundaryCondition();

    //! Extrapolates given field into the collider-occupied region.
    void extrapolateIntoCollider(ScalarGrid2* grid);

    //! Extrapolates given field into the collider-occupied region.
    void extrapolateIntoCollider(CollocatedVectorGrid2* grid);

    //! Extrapolates given field into the collider-occupied region.
    void extrapolateIntoCollider(FaceCenteredGrid2* grid);

    //! Returns the signed-distance field representation of the collider.
    const CellCenteredScalarGrid2& colliderSdf() const;

 private:
    Vector2D _gravity = Vector2D(0.0, -9.8);
    double _viscosityCoefficient = 0.0;
    double _maxCfl = 5.0;
    int _closedDomainBoundaryFlag = kDirectionAll;

    GridSystemData2Ptr _grids;
    Collider2Ptr _collider;
    CellCenteredScalarGrid2 _colliderSdf;

    AdvectionSolver2Ptr _advectionSolver;
    GridDiffusionSolver2Ptr _diffusionSolver;
    GridPressureSolver2Ptr _pressureSolver;
    GridBoundaryConditionSolver2Ptr _boundaryConditionSolver;

    void beginAdvanceTimeStep(double timeIntervalInSeconds);

    void endAdvanceTimeStep(double timeIntervalInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FLUID_SOLVER2_H_
