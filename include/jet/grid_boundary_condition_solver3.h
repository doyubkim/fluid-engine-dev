// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER3_H_
#define INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER3_H_

#include <jet/collider3.h>
#include <jet/constants.h>
#include <jet/face_centered_grid3.h>

#include <memory>

namespace jet {

class GridBoundaryConditionSolver3 {
 public:
    GridBoundaryConditionSolver3();

    virtual ~GridBoundaryConditionSolver3();

    const Collider3Ptr& collider() const;

    void updateCollider(
        const Collider3Ptr& newCollider,
        const Size3& gridSize,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin);

    int closedDomainBoundaryFlag() const;

    void setClosedDomainBoundaryFlag(int flag);

    virtual void constrainVelocity(
        FaceCenteredGrid3* velocity,
        unsigned int extrapolationDepth = 5) = 0;

 protected:
    virtual void onColliderUpdated(
        const Size3& gridSize,
        const Vector3D& gridSpacing,
        const Vector3D& gridOrigin) = 0;

    const Size3& gridSize() const;

    const Vector3D& gridSpacing() const;

    const Vector3D& gridOrigin() const;

 private:
    Collider3Ptr _collider;
    Size3 _gridSize;
    Vector3D _gridSpacing;
    Vector3D _gridOrigin;
    int _closedDomainBoundaryFlag = kDirectionAll;
};

typedef std::shared_ptr<GridBoundaryConditionSolver3>
    GridBoundaryConditionSolver3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_GRID_BOUNDARY_CONDITION_SOLVER3_H_
