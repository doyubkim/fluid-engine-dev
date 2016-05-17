// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER2_H_
#define INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER2_H_

#include <jet/collider2.h>
#include <jet/constants.h>
#include <jet/face_centered_grid2.h>

#include <memory>

namespace jet {

class GridBoundaryConditionSolver2 {
 public:
    GridBoundaryConditionSolver2();

    virtual ~GridBoundaryConditionSolver2();

    const Collider2Ptr& collider() const;

    void updateCollider(
        const Collider2Ptr& newCollider,
        const Size2& gridSize,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin);

    int closedDomainBoundaryFlag() const;

    void setClosedDomainBoundaryFlag(int flag);

    virtual void constrainVelocity(
        FaceCenteredGrid2* velocity,
        unsigned int extrapolationDepth = 5) = 0;

 protected:
    virtual void onColliderUpdated(
        const Size2& gridSize,
        const Vector2D& gridSpacing,
        const Vector2D& gridOrigin) = 0;

    const Size2& gridSize() const;

    const Vector2D& gridSpacing() const;

    const Vector2D& gridOrigin() const;

 private:
    Collider2Ptr _collider;
    Size2 _gridSize;
    Vector2D _gridSpacing;
    Vector2D _gridOrigin;
    int _closedDomainBoundaryFlag = kDirectionAll;
};

typedef std::shared_ptr<GridBoundaryConditionSolver2>
    GridBoundaryConditionSolver2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER2_H_
