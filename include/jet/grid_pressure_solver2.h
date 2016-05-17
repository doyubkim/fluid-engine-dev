// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_PRESSURE_SOLVER2_H_
#define INCLUDE_JET_GRID_PRESSURE_SOLVER2_H_

#include <jet/collocated_vector_grid2.h>
#include <jet/constant_scalar_field2.h>
#include <jet/constants.h>
#include <jet/face_centered_grid2.h>
#include <jet/grid_boundary_condition_solver2.h>
#include <jet/scalar_grid2.h>
#include <memory>

namespace jet {

class GridPressureSolver2 {
 public:
    GridPressureSolver2();

    virtual ~GridPressureSolver2();

    virtual void solve(
        const FaceCenteredGrid2& input,
        double timeIntervalInSeconds,
        FaceCenteredGrid2* output,
        const ScalarField2& boundarySdf
            = ConstantScalarField2(kMaxD),
        const ScalarField2& fluidSdf
            = ConstantScalarField2(-kMaxD)) = 0;

    virtual GridBoundaryConditionSolver2Ptr
        suggestedBoundaryConditionSolver() const = 0;
};

typedef std::shared_ptr<GridPressureSolver2> GridPressureSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_PRESSURE_SOLVER2_H_
