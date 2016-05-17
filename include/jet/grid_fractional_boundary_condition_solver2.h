// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_FRACTIONAL_BOUNDARY_CONDITION_SOLVER2_H_
#define INCLUDE_JET_GRID_FRACTIONAL_BOUNDARY_CONDITION_SOLVER2_H_

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/grid_boundary_condition_solver2.h>

#include <memory>

namespace jet {

class GridFractionalBoundaryConditionSolver2 final
    : public GridBoundaryConditionSolver2 {
 public:
    GridFractionalBoundaryConditionSolver2();

    void constrainVelocity(
        FaceCenteredGrid2* velocity,
        unsigned int extrapolationDepth = 5) override;

 protected:
    void onColliderUpdated(
        const Size2& gridSize,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin) override;

 private:
    CellCenteredScalarGrid2 _colliderSdf;
};

typedef std::shared_ptr<GridFractionalBoundaryConditionSolver2>
    GridFractionalBoundaryConditionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_FRACTIONAL_BOUNDARY_CONDITION_SOLVER2_H_
