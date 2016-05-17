// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/grid_boundary_condition_solver3.h>

using namespace jet;

GridBoundaryConditionSolver3::GridBoundaryConditionSolver3() {
}

GridBoundaryConditionSolver3::~GridBoundaryConditionSolver3() {
}

const Collider3Ptr& GridBoundaryConditionSolver3::collider() const {
    return _collider;
}

void GridBoundaryConditionSolver3::updateCollider(
    const Collider3Ptr& newCollider,
    const Size3& gridSize,
    const Vector3D& gridSpacing,
    const Vector3D& gridOrigin) {
    _collider = newCollider;
    _gridSize = gridSize;
    _gridSpacing = gridSpacing;
    _gridOrigin = gridOrigin;

    onColliderUpdated(gridSize, gridSpacing, gridOrigin);
}

int GridBoundaryConditionSolver3::closedDomainBoundaryFlag() const {
    return _closedDomainBoundaryFlag;
}

void GridBoundaryConditionSolver3::setClosedDomainBoundaryFlag(int flag) {
    _closedDomainBoundaryFlag = flag;
}

const Size3& GridBoundaryConditionSolver3::gridSize() const {
    return _gridSize;
}

const Vector3D& GridBoundaryConditionSolver3::gridSpacing() const {
    return _gridSpacing;
}

const Vector3D& GridBoundaryConditionSolver3::gridOrigin() const {
    return _gridOrigin;
}
