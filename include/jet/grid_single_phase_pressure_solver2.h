// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER2_H_
#define INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER2_H_

#include <jet/fdm_linear_system_solver2.h>
#include <jet/grid_boundary_condition_solver2.h>
#include <jet/grid_pressure_solver2.h>
#include <memory>

namespace jet {

class GridSinglePhasePressureSolver2 : public GridPressureSolver2 {
 public:
    GridSinglePhasePressureSolver2();

    virtual ~GridSinglePhasePressureSolver2();

    void solve(
        const FaceCenteredGrid2& input,
        double timeIntervalInSeconds,
        FaceCenteredGrid2* output,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) override;

    GridBoundaryConditionSolver2Ptr
        suggestedBoundaryConditionSolver() const override;

    void setLinearSystemSolver(const FdmLinearSystemSolver2Ptr& solver);

    const FdmVector2& pressure() const;

 protected:
    FdmLinearSystem2 _system;
    FdmLinearSystemSolver2Ptr _systemSolver;
    Array2<char> _markers;

    void buildMarkers(
        const Size2& size,
        const std::function<Vector2D(size_t, size_t)>& pos,
        const ScalarField2& boundarySdf,
        const ScalarField2& fluidSdf);

    virtual void buildSystem(const FaceCenteredGrid2& input);

    virtual void applyPressureGradient(
        const FaceCenteredGrid2& input,
        FaceCenteredGrid2* output);
};

typedef std::shared_ptr<GridSinglePhasePressureSolver2>
    GridSinglePhasePressureSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_SINGLE_PHASE_PRESSURE_SOLVER2_H_
