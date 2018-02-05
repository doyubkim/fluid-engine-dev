// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/grid_boundary_condition_solver2.h>

using namespace jet;

GridBoundaryConditionSolver2::GridBoundaryConditionSolver2() {
}

GridBoundaryConditionSolver2::~GridBoundaryConditionSolver2() {
}

const Collider2Ptr& GridBoundaryConditionSolver2::collider() const {
    return _collider;
}

void GridBoundaryConditionSolver2::updateCollider(
    const Collider2Ptr& newCollider,
    const Size2& gridSize,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin) {
    _collider = newCollider;
    _gridSize = gridSize;
    _gridSpacing = gridSpacing;
    _gridOrigin = gridOrigin;

    onColliderUpdated(gridSize, gridSpacing, gridOrigin);
}

int GridBoundaryConditionSolver2::closedDomainBoundaryFlag() const {
    return _closedDomainBoundaryFlag;
}

void GridBoundaryConditionSolver2::setClosedDomainBoundaryFlag(int flag) {
    _closedDomainBoundaryFlag = flag;
}

const Size2& GridBoundaryConditionSolver2::gridSize() const {
    return _gridSize;
}

const Vector2D& GridBoundaryConditionSolver2::gridSpacing() const {
    return _gridSpacing;
}

const Vector2D& GridBoundaryConditionSolver2::gridOrigin() const {
    return _gridOrigin;
}
