// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GRID_FLUID_SOLVER3_H_
#define INCLUDE_JET_GRID_FLUID_SOLVER3_H_

#include <jet/advection_solver3.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/collider3.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/grid_diffusion_solver3.h>
#include <jet/grid_emitter3.h>
#include <jet/grid_pressure_solver3.h>
#include <jet/grid_system_data3.h>
#include <jet/physics_animation.h>

namespace jet {

//!
//! \brief Abstract base class for grid-based 3-D fluid solver.
//!
//! This is an abstract base class for grid-based 3-D fluid solver based on
//! Jos Stam's famous 1999 paper - "Stable Fluids". This solver takes fractional
//! step method as its foundation which is consisted of independent advection,
//! diffusion, external forces, and pressure projection steps. Each step is
//! configurable so that a custom step can be implemented. For example, if a
//! user wants to change the advection solver to her/his own implementation,
//! simply call GridFluidSolver3::setAdvectionSolver(newSolver).
//!
class GridFluidSolver3 : public PhysicsAnimation {
 public:
    class Builder;

    //! Default constructor.
    GridFluidSolver3();

    //! Constructs solver with initial grid size.
    GridFluidSolver3(const Size3& resolution, const Vector3D& gridSpacing,
                     const Vector3D& gridOrigin);

    //! Default destructor.
    virtual ~GridFluidSolver3();

    //! Returns the gravity vector of the system.
    const Vector3D& gravity() const;

    //! Sets the gravity of the system.
    void setGravity(const Vector3D& newGravity);

    //! Returns the viscosity coefficient.
    double viscosityCoefficient() const;

    //!
    //! \brief Sets the viscosity coefficient.
    //!
    //! This function sets the viscosity coefficient. Non-positive input will be
    //! clamped to zero.
    //!
    //! \param[in] newValue The new viscosity coefficient value.
    //!
    void setViscosityCoefficient(double newValue);

    //!
    //! \brief Returns the CFL number from the current velocity field for given
    //!     time interval.
    //!
    //! \param[in] timeIntervalInSeconds The time interval in seconds.
    //!
    double cfl(double timeIntervalInSeconds) const;

    //! Returns the max allowed CFL number.
    double maxCfl() const;

    //! Sets the max allowed CFL number.
    void setMaxCfl(double newCfl);

    //! Returns true if the solver is using compressed linear system.
    bool useCompressedLinearSystem() const;

    //! Sets whether the solver should use compressed linear system.
    void setUseCompressedLinearSystem(bool onoff);

    //! Returns the advection solver instance.
    const AdvectionSolver3Ptr& advectionSolver() const;

    //! Sets the advection solver.
    void setAdvectionSolver(const AdvectionSolver3Ptr& newSolver);

    //! Returns the diffusion solver instance.
    const GridDiffusionSolver3Ptr& diffusionSolver() const;

    //! Sets the diffusion solver.
    void setDiffusionSolver(const GridDiffusionSolver3Ptr& newSolver);

    //! Returns the pressure solver instance.
    const GridPressureSolver3Ptr& pressureSolver() const;

    //! Sets the pressure solver.
    void setPressureSolver(const GridPressureSolver3Ptr& newSolver);

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
    //! \see GridSystemData3
    //!
    const GridSystemData3Ptr& gridSystemData() const;

    //!
    //! \brief Resizes grid system data.
    //!
    //! This function resizes grid system data. You can also resize the grid by
    //! calling resize function directly from
    //! GridFluidSolver3::gridSystemData(), but this function provides a
    //! shortcut for the same operation.
    //!
    //! \param[in] newSize        The new size.
    //! \param[in] newGridSpacing The new grid spacing.
    //! \param[in] newGridOrigin  The new grid origin.
    //!
    void resizeGrid(const Size3& newSize, const Vector3D& newGridSpacing,
                    const Vector3D& newGridOrigin);

    //!
    //! \brief Returns the resolution of the grid system data.
    //!
    //! This function returns the resolution of the grid system data. This is
    //! equivalent to calling gridSystemData()->resolution(), but provides a
    //! shortcut.
    //!
    Size3 resolution() const;

    //!
    //! \brief Returns the grid spacing of the grid system data.
    //!
    //! This function returns the resolution of the grid system data. This is
    //! equivalent to calling gridSystemData()->gridSpacing(), but provides a
    //! shortcut.
    //!
    Vector3D gridSpacing() const;

    //!
    //! \brief Returns the origin of the grid system data.
    //!
    //! This function returns the resolution of the grid system data. This is
    //! equivalent to calling gridSystemData()->origin(), but provides a
    //! shortcut.
    //!
    Vector3D gridOrigin() const;

    //!
    //! \brief Returns the velocity field.
    //!
    //! This function returns the velocity field from the grid system data.
    //! It is just a shortcut to the most commonly accessed data chunk.
    //!
    const FaceCenteredGrid3Ptr& velocity() const;

    //! Returns the collider.
    const Collider3Ptr& collider() const;

    //! Sets the collider.
    void setCollider(const Collider3Ptr& newCollider);

    //! Returns the emitter.
    const GridEmitter3Ptr& emitter() const;

    //! Sets the emitter.
    void setEmitter(const GridEmitter3Ptr& newEmitter);

    //! Returns builder fox GridFluidSolver3.
    static Builder builder();

 protected:
    //! Called when it needs to setup initial condition.
    void onInitialize() override;

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
    //! \see GridFluidSolver3::maxCfl
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
    //! \see GridFluidSolver3::computeGravity
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
    virtual ScalarField3Ptr fluidSdf() const;

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
    void extrapolateIntoCollider(ScalarGrid3* grid);

    //! Extrapolates given field into the collider-occupied region.
    void extrapolateIntoCollider(CollocatedVectorGrid3* grid);

    //! Extrapolates given field into the collider-occupied region.
    void extrapolateIntoCollider(FaceCenteredGrid3* grid);

    //! Returns the signed-distance field representation of the collider.
    ScalarField3Ptr colliderSdf() const;

    //! Returns the velocity field of the collider.
    VectorField3Ptr colliderVelocityField() const;

 private:
    Vector3D _gravity = Vector3D(0.0, -9.8, 0.0);
    double _viscosityCoefficient = 0.0;
    double _maxCfl = 5.0;
    bool _useCompressedLinearSys = false;
    int _closedDomainBoundaryFlag = kDirectionAll;

    GridSystemData3Ptr _grids;
    Collider3Ptr _collider;
    GridEmitter3Ptr _emitter;

    AdvectionSolver3Ptr _advectionSolver;
    GridDiffusionSolver3Ptr _diffusionSolver;
    GridPressureSolver3Ptr _pressureSolver;
    GridBoundaryConditionSolver3Ptr _boundaryConditionSolver;

    void beginAdvanceTimeStep(double timeIntervalInSeconds);

    void endAdvanceTimeStep(double timeIntervalInSeconds);

    void updateCollider(double timeIntervalInSeconds);

    void updateEmitter(double timeIntervalInSeconds);
};

//! Shared pointer type for the GridFluidSolver3.
typedef std::shared_ptr<GridFluidSolver3> GridFluidSolver3Ptr;

//!
//! \brief Base class for grid-based fluid solver builder.
//!
template <typename DerivedBuilder>
class GridFluidSolverBuilderBase3 {
 public:
    //! Returns builder with grid resolution.
    DerivedBuilder& withResolution(const Size3& resolution);

    //! Returns builder with grid spacing.
    DerivedBuilder& withGridSpacing(const Vector3D& gridSpacing);

    //! Returns builder with grid spacing.
    DerivedBuilder& withGridSpacing(double gridSpacing);

    //!
    //! \brief Returns builder with domain size in x-direction.
    //!
    //! To build a solver, one can use either grid spacing directly or domain
    //! size in x-direction to set the final grid spacing.
    //!
    DerivedBuilder& withDomainSizeX(double domainSizeX);

    //! Returns builder with grid origin
    DerivedBuilder& withOrigin(const Vector3D& gridOrigin);

 protected:
    Size3 _resolution{1, 1, 1};
    Vector3D _gridSpacing{1, 1, 1};
    Vector3D _gridOrigin{0, 0, 0};
    double _domainSizeX = 1.0;
    bool _useDomainSize = false;

    Vector3D getGridSpacing() const;
};

template <typename T>
T& GridFluidSolverBuilderBase3<T>::withResolution(const Size3& resolution) {
    _resolution = resolution;
    return static_cast<T&>(*this);
}

template <typename T>
T& GridFluidSolverBuilderBase3<T>::withGridSpacing(
    const Vector3D& gridSpacing) {
    _gridSpacing = gridSpacing;
    _useDomainSize = false;
    return static_cast<T&>(*this);
}

template <typename T>
T& GridFluidSolverBuilderBase3<T>::withGridSpacing(double gridSpacing) {
    _gridSpacing.x = gridSpacing;
    _gridSpacing.y = gridSpacing;
    _gridSpacing.z = gridSpacing;
    _useDomainSize = false;
    return static_cast<T&>(*this);
}

template <typename T>
T& GridFluidSolverBuilderBase3<T>::withDomainSizeX(double domainSizeX) {
    _domainSizeX = domainSizeX;
    _useDomainSize = true;
    return static_cast<T&>(*this);
}

template <typename T>
T& GridFluidSolverBuilderBase3<T>::withOrigin(const Vector3D& gridOrigin) {
    _gridOrigin = gridOrigin;
    return static_cast<T&>(*this);
}

template <typename T>
Vector3D GridFluidSolverBuilderBase3<T>::getGridSpacing() const {
    Vector3D gridSpacing = _gridSpacing;
    if (_useDomainSize) {
        gridSpacing.set(_domainSizeX / static_cast<double>(_resolution.x));
    }
    return gridSpacing;
}

//!
//! \brief Front-end to create GridFluidSolver3 objects step by step.
//!
class GridFluidSolver3::Builder final
    : public GridFluidSolverBuilderBase3<GridFluidSolver3::Builder> {
 public:
    //! Builds GridFluidSolver3.
    GridFluidSolver3 build() const;

    //! Builds shared pointer of GridFluidSolver3 instance.
    GridFluidSolver3Ptr makeShared() const {
        return std::make_shared<GridFluidSolver3>(_resolution, getGridSpacing(),
                                                  _gridOrigin);
    }
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FLUID_SOLVER3_H_
