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

class GridFluidSolver2 : public PhysicsAnimation {
 public:
    GridFluidSolver2();

    virtual ~GridFluidSolver2();

    const Vector2D& gravity() const;

    void setGravity(const Vector2D& newGravity);

    double viscosityCoefficient() const;

    void setViscosityCoefficient(double newValue);

    double cfl(double timeIntervalInSeconds) const;

    double maxCfl() const;

    void setMaxCfl(double newCfl);

    const AdvectionSolver2Ptr& advectionSolver() const;

    void setAdvectionSolver(const AdvectionSolver2Ptr& newSolver);

    const GridDiffusionSolver2Ptr& diffusionSolver() const;

    void setDiffusionSolver(const GridDiffusionSolver2Ptr& newSolver);

    const GridPressureSolver2Ptr& pressureSolver() const;

    void setPressureSolver(const GridPressureSolver2Ptr& newSolver);

    const GridBoundaryConditionSolver2Ptr& boundaryConditionSolver() const;

    const GridSystemData2Ptr& gridSystemData() const;

    const FaceCenteredGrid2Ptr& velocity() const;

    const Collider2Ptr& collider() const;

    void setCollider(const Collider2Ptr& newCollider);

 protected:
    void onAdvanceTimeStep(double timeIntervalInSeconds) override;

    unsigned int numberOfSubTimeSteps(
        double timeIntervalInSeconds) const override;

    virtual void onBeginAdvanceTimeStep(double timeIntervalInSeconds);

    virtual void onEndAdvanceTimeStep(double timeIntervalInSeconds);

    virtual void computeExternalForces(double timeIntervalInSeconds);

    virtual void computeViscosity(double timeIntervalInSeconds);

    virtual void computePressure(double timeIntervalInSeconds);

    virtual void computeAdvection(double timeIntervalInSeconds);

    virtual ScalarField2Ptr fluidSdf() const;

    void computeGravity(double timeIntervalInSeconds);

    void applyBoundaryCondition();

    void extrapolateIntoCollider(ScalarGrid2* grid);

    void extrapolateIntoCollider(CollocatedVectorGrid2* grid);

    void extrapolateIntoCollider(FaceCenteredGrid2* grid);

    const CellCenteredScalarGrid2& colliderSdf() const;

 private:
    Vector2D _gravity = Vector2D(0.0, -9.8);
    double _viscosityCoefficient = 0.0;
    double _maxCfl = 2.0;

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
