// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <gtest/gtest.h>
#include <jet/grid_blocked_boundary_condition_solver3.h>
#include <jet/plane.h>
#include <jet/rigid_body_collider.h>

using namespace jet;

TEST(GridBlockedBoundaryConditionSolver3, ClosedDomain) {
    GridBlockedBoundaryConditionSolver3 bndSolver;
    Vector3UZ gridSize(10, 10, 10);
    Vector3D gridSpacing(1.0, 1.0, 1.0);
    Vector3D gridOrigin(-5.0, -5.0, -5.0);

    bndSolver.updateCollider(nullptr, gridSize, gridSpacing, gridOrigin);

    FaceCenteredGrid3 velocity(gridSize, gridSpacing, gridOrigin);
    velocity.fill(Vector3D(1.0, 1.0, 1.0));

    bndSolver.constrainVelocity(&velocity);

    velocity.forEachUIndex([&](const Vector3UZ& idx) {
        if (idx.x == 0 || idx.x == gridSize.x) {
            EXPECT_DOUBLE_EQ(0.0, velocity.u(idx));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.u(idx));
        }
    });

    velocity.forEachVIndex([&](const Vector3UZ& idx) {
        if (idx.y == 0 || idx.y == gridSize.y) {
            EXPECT_DOUBLE_EQ(0.0, velocity.v(idx));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.v(idx));
        }
    });

    velocity.forEachWIndex([&](const Vector3UZ& idx) {
        if (idx.z == 0 || idx.z == gridSize.z) {
            EXPECT_DOUBLE_EQ(0.0, velocity.w(idx));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.w(idx));
        }
    });
}

TEST(GridBlockedBoundaryConditionSolver3, OpenDomain) {
    GridBlockedBoundaryConditionSolver3 bndSolver;
    Vector3UZ gridSize(10, 10, 10);
    Vector3D gridSpacing(1.0, 1.0, 1.0);
    Vector3D gridOrigin(-5.0, -5.0, -5.0);

    // Partially open domain
    bndSolver.setClosedDomainBoundaryFlag(kDirectionLeft | kDirectionUp |
                                          kDirectionFront);
    bndSolver.updateCollider(nullptr, gridSize, gridSpacing, gridOrigin);

    FaceCenteredGrid3 velocity(gridSize, gridSpacing, gridOrigin);
    velocity.fill(Vector3D(1.0, 1.0, 1.0));

    bndSolver.constrainVelocity(&velocity);

    velocity.forEachUIndex([&](const Vector3UZ& idx) {
        if (idx.x == 0) {
            EXPECT_DOUBLE_EQ(0.0, velocity.u(idx));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.u(idx));
        }
    });

    velocity.forEachVIndex([&](const Vector3UZ& idx) {
        if (idx.y == gridSize.y) {
            EXPECT_DOUBLE_EQ(0.0, velocity.v(idx));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.v(idx));
        }
    });

    velocity.forEachWIndex([&](const Vector3UZ& idx) {
        if (idx.z == gridSize.z) {
            EXPECT_DOUBLE_EQ(0.0, velocity.w(idx));
        } else {
            EXPECT_DOUBLE_EQ(1.0, velocity.w(idx));
        }
    });
}
