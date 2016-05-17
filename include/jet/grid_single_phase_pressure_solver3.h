// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER3_H_
#define INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER3_H_

#include <jet/fdm_linear_system_solver3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/grid_pressure_solver3.h>
#include <memory>

namespace jet {

class GridSinglePhasePressureSolver3 : public GridPressureSolver3 {
 public:
    GridSinglePhasePressureSolver3();

    virtual ~GridSinglePhasePressureSolver3();

    void solve(
        const FaceCenteredGrid3& input,
        double timeIntervalInSeconds,
        FaceCenteredGrid3* output,
        const ScalarField3& boundarySdf
            = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf
            = ConstantScalarField3(-kMaxD)) override;

    GridBoundaryConditionSolver3Ptr
        suggestedBoundaryConditionSolver() const override;

    void setLinearSystemSolver(const FdmLinearSystemSolver3Ptr& solver);

    const FdmVector3& pressure() const;

 protected:
    FdmLinearSystem3 _system;
    FdmLinearSystemSolver3Ptr _systemSolver;
    Array3<char> _markers;

    void buildMarkers(
        const Size3& size,
        const std::function<Vector3D(size_t, size_t, size_t)>& pos,
        const ScalarField3& boundarySdf,
        const ScalarField3& fluidSdf);

    virtual void buildSystem(const FaceCenteredGrid3& input);

    virtual void applyPressureGradient(
        const FaceCenteredGrid3& input,
        FaceCenteredGrid3* output);
};

typedef std::shared_ptr<GridSinglePhasePressureSolver3>
    GridSinglePhasePressureSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER3_H_
