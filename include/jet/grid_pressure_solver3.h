// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_PRESSURE_SOLVER3_H_
#define INCLUDE_JET_GRID_PRESSURE_SOLVER3_H_

#include <jet/collocated_vector_grid3.h>
#include <jet/constant_scalar_field3.h>
#include <jet/constants.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_boundary_condition_solver3.h>
#include <jet/scalar_grid3.h>
#include <memory>

namespace jet {

class GridPressureSolver3 {
 public:
    GridPressureSolver3();

    virtual ~GridPressureSolver3();

    virtual void solve(
        const FaceCenteredGrid3& input,
        double timeIntervalInSeconds,
        FaceCenteredGrid3* output,
        const ScalarField3& boundarySdf
            = ConstantScalarField3(kMaxD),
        const ScalarField3& fluidSdf
            = ConstantScalarField3(-kMaxD)) = 0;

    virtual GridBoundaryConditionSolver3Ptr
        suggestedBoundaryConditionSolver() const = 0;
};

typedef std::shared_ptr<GridPressureSolver3> GridPressureSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_PRESSURE_SOLVER3_H_
