// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_FRACTIONAL_SINGLE_PHASE_PRESSURE_SOLVER3_H_
#define INCLUDE_JET_GRID_FRACTIONAL_SINGLE_PHASE_PRESSURE_SOLVER3_H_

#include <jet/cell_centered_scalar_grid3.h>
#include <jet/fdm_linear_system_solver3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/grid_pressure_solver3.h>
#include <jet/vertex_centered_scalar_grid3.h>
#include <memory>

namespace jet {

class GridFractionalSinglePhasePressureSolver3 : public GridPressureSolver3 {
 public:
    GridFractionalSinglePhasePressureSolver3();

    virtual ~GridFractionalSinglePhasePressureSolver3();

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
    Array3<double> _uWeights;
    Array3<double> _vWeights;
    Array3<double> _wWeights;
    CellCenteredScalarGrid3 _fluidSdf;

    void buildWeights(
        const FaceCenteredGrid3& input,
        const ScalarField3& boundarySdf,
        const ScalarField3& fluidSdf);

    virtual void buildSystem(const FaceCenteredGrid3& input);

    virtual void applyPressureGradient(
        const FaceCenteredGrid3& input,
        FaceCenteredGrid3* output);
};

typedef std::shared_ptr<GridFractionalSinglePhasePressureSolver3>
    GridFractionalSinglePhasePressureSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FRACTIONAL_SINGLE_PHASE_PRESSURE_SOLVER3_H_
