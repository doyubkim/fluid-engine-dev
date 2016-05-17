// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_FLUID_SOLVER3_H_
#define INCLUDE_JET_GRID_FLUID_SOLVER3_H_

#include <jet/advection_solver3.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/collider3.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/grid_diffusion_solver3.h>
#include <jet/grid_pressure_solver3.h>
#include <jet/grid_system_data3.h>
#include <jet/physics_animation.h>

namespace jet {

class GridFluidSolver3 : public PhysicsAnimation {
 public:
    GridFluidSolver3();

    virtual ~GridFluidSolver3();

    const Vector3D& gravity() const;

    void setGravity(const Vector3D& newGravity);

    double viscosityCoefficient() const;

    void setViscosityCoefficient(double newValue);

    double cfl(double timeIntervalInSeconds) const;

    double maxCfl() const;

    void setMaxCfl(double newCfl);

    const AdvectionSolver3Ptr& advectionSolver() const;

    void setAdvectionSolver(const AdvectionSolver3Ptr& newSolver);

    const GridDiffusionSolver3Ptr& diffusionSolver() const;

    void setDiffusionSolver(const GridDiffusionSolver3Ptr& newSolver);

    const GridPressureSolver3Ptr& pressureSolver() const;

    void setPressureSolver(const GridPressureSolver3Ptr& newSolver);

    const GridBoundaryConditionSolver3Ptr& boundaryConditionSolver() const;

    const GridSystemData3Ptr& gridSystemData() const;

    const FaceCenteredGrid3Ptr& velocity() const;

    const Collider3Ptr& collider() const;

    void setCollider(const Collider3Ptr& newCollider);

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

    virtual ScalarField3Ptr fluidSdf() const;

    void computeGravity(double timeIntervalInSeconds);

    void applyBoundaryCondition();

    void extrapolateIntoCollider(ScalarGrid3* grid);

    void extrapolateIntoCollider(CollocatedVectorGrid3* grid);

    void extrapolateIntoCollider(FaceCenteredGrid3* grid);

    const CellCenteredScalarGrid3& colliderSdf() const;

 private:
    Vector3D _gravity = Vector3D(0.0, -9.8, 0.0);
    double _viscosityCoefficient = 0.0;
    double _maxCfl = 5.0;

    GridSystemData3Ptr _grids;
    Collider3Ptr _collider;
    CellCenteredScalarGrid3 _colliderSdf;

    AdvectionSolver3Ptr _advectionSolver;
    GridDiffusionSolver3Ptr _diffusionSolver;
    GridPressureSolver3Ptr _pressureSolver;
    GridBoundaryConditionSolver3Ptr _boundaryConditionSolver;

    void beginAdvanceTimeStep(double timeIntervalInSeconds);

    void endAdvanceTimeStep(double timeIntervalInSeconds);
};

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FLUID_SOLVER3_H_
