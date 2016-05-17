// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_BLOCKED_BOUNDARY_CONDITION_SOLVER2_H_
#define INCLUDE_JET_GRID_BLOCKED_BOUNDARY_CONDITION_SOLVER2_H_

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/grid_boundary_condition_solver2.h>

#include <memory>

namespace jet {

class GridBlockedBoundaryConditionSolver2 final
    : public GridBoundaryConditionSolver2 {
 public:
    GridBlockedBoundaryConditionSolver2();

    void constrainVelocity(
        FaceCenteredGrid2* velocity,
        unsigned int extrapolationDepth = 5) override;

    const Array2<char>& marker() const;

 protected:
    void onColliderUpdated(
        const Size2& gridSize,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin) override;

 private:
    Array2<char> _marker;
    CellCenteredScalarGrid2 _colliderSdf;
};

typedef std::shared_ptr<GridBlockedBoundaryConditionSolver2>
    GridBlockedBoundaryConditionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BLOCKED_BOUNDARY_CONDITION_SOLVER2_H_
