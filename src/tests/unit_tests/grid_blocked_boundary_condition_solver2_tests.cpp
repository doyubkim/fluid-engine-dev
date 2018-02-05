// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/grid_blocked_boundary_condition_solver2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridBlockedBoundaryConditionSolver2, ClosedDomain) {
    GridBlockedBoundaryConditionSolver2 bndSolver;
    Size2 gridSize(10, 10);
    Vector2D gridSpacing(1.0, 1.0);
    Vector2D gridOrigin(-5.0, -5.0);

    bndSolver.updateCollider(nullptr, gridSize, gridSpacing, gridOrigin);

    FaceCenteredGrid2 velocity(gridSize, gridSpacing, gridOrigin);
    velocity.fill(Vector2D(1.0, 1.0));

    bndSolver.constrainVelocity(&velocity);

    velocity.forEachUIndex([&](size_t i, size_t j) {
        if (i == 0 || i == gridSize.x) {
            EXPECT_DOUBLE_EQ(0.0, velocity.u(i, j));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.u(i, j));
        }
    });

    velocity.forEachVIndex([&](size_t i, size_t j) {
        if (j == 0 || j == gridSize.y) {
            EXPECT_DOUBLE_EQ(0.0, velocity.v(i, j));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.v(i, j));
        }
    });
}

TEST(GridBlockedBoundaryConditionSolver2, OpenDomain) {
    GridBlockedBoundaryConditionSolver2 bndSolver;
    Size2 gridSize(10, 10);
    Vector2D gridSpacing(1.0, 1.0);
    Vector2D gridOrigin(-5.0, -5.0);

    // Partially open domain
    bndSolver.setClosedDomainBoundaryFlag(kDirectionLeft | kDirectionUp);
    bndSolver.updateCollider(nullptr, gridSize, gridSpacing, gridOrigin);

    FaceCenteredGrid2 velocity(gridSize, gridSpacing, gridOrigin);
    velocity.fill(Vector2D(1.0, 1.0));

    bndSolver.constrainVelocity(&velocity);

    velocity.forEachUIndex([&](size_t i, size_t j) {
        if (i == 0) {
            EXPECT_DOUBLE_EQ(0.0, velocity.u(i, j));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.u(i, j));
        }
    });

    velocity.forEachVIndex([&](size_t i, size_t j) {
        if (j == gridSize.y) {
            EXPECT_DOUBLE_EQ(0.0, velocity.v(i, j));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.v(i, j));
        }
    });
}
