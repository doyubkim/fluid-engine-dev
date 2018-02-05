// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <physics_helpers.h>
#include <jet/array_utils.h>
#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit2.h>
#include <algorithm>

using namespace jet;

static const char kFluid = 1;
static const char kCollider = 0;

GridBlockedBoundaryConditionSolver2::GridBlockedBoundaryConditionSolver2() {
}

void GridBlockedBoundaryConditionSolver2::constrainVelocity(
    FaceCenteredGrid2* velocity,
    unsigned int extrapolationDepth) {
    GridFractionalBoundaryConditionSolver2::constrainVelocity(
        velocity, extrapolationDepth);

    // No-flux: project the velocity at the marker interface
    Size2 size = velocity->resolution();
    auto u = velocity->uAccessor();
    auto v = velocity->vAccessor();
    auto uPos = velocity->uPosition();
    auto vPos = velocity->vPosition();

    _marker.forEachIndex([&](size_t i, size_t j) {
        if (_marker(i, j) == kCollider) {
            if (i > 0 && _marker(i - 1, j) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(uPos(i, j));
                u(i, j) = colliderVel.x;
            }
            if (i < size.x - 1 && _marker(i + 1, j) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(uPos(i + 1, j));
                u(i + 1, j) = colliderVel.x;
            }
            if (j > 0 && _marker(i, j - 1) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(vPos(i, j));
                v(i, j) = colliderVel.y;
            }
            if (j < size.y - 1 && _marker(i, j + 1) == kFluid) {
                Vector2D colliderVel = collider()->velocityAt(vPos(i, j + 1));
                v(i, j + 1) = colliderVel.y;
            }
        }
    });
}

const Array2<char>& GridBlockedBoundaryConditionSolver2::marker() const {
    return _marker;
}

void GridBlockedBoundaryConditionSolver2::onColliderUpdated(
    const Size2& gridSize,
    const Vector2D& gridSpacing,
    const Vector2D& gridOrigin) {
    GridFractionalBoundaryConditionSolver2::onColliderUpdated(
        gridSize, gridSpacing, gridOrigin);

    const auto sdf
        = std::dynamic_pointer_cast<CellCenteredScalarGrid2>(colliderSdf());

    _marker.resize(gridSize);
    _marker.parallelForEachIndex([&](size_t i, size_t j) {
        if (isInsideSdf((*sdf)(i, j))) {
            _marker(i, j) = kCollider;
        } else {
            _marker(i, j) = kFluid;
        }
    });
}
