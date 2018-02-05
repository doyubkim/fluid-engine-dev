// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/grid_fluid_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridFluidSolver3, Constructor) {
    GridFluidSolver3 solver;

    // Check if the sub-step solvers are present
    EXPECT_TRUE(solver.advectionSolver() != nullptr);
    EXPECT_TRUE(solver.diffusionSolver() != nullptr);
    EXPECT_TRUE(solver.pressureSolver() != nullptr);

    // Check default parameters
    EXPECT_GE(solver.viscosityCoefficient(), 0.0);
    EXPECT_GT(solver.maxCfl(), 0.0);
    EXPECT_EQ(kDirectionAll, solver.closedDomainBoundaryFlag());

    // Check grid system data
    EXPECT_TRUE(solver.gridSystemData() != nullptr);
    EXPECT_EQ(1u, solver.gridSystemData()->resolution().x);
    EXPECT_EQ(1u, solver.gridSystemData()->resolution().y);
    EXPECT_EQ(1u, solver.gridSystemData()->resolution().z);
    EXPECT_EQ(solver.gridSystemData()->velocity(), solver.velocity());

    // Collider should be null
    EXPECT_TRUE(solver.collider() == nullptr);
}

TEST(GridFluidSolver3, ResizeGridSystemData) {
    GridFluidSolver3 solver;

    solver.resizeGrid(
        Size3(1, 2, 3),
        Vector3D(4.0, 5.0, 6.0),
        Vector3D(7.0, 8.0, 9.0));

    EXPECT_EQ(1u, solver.resolution().x);
    EXPECT_EQ(2u, solver.resolution().y);
    EXPECT_EQ(3u, solver.resolution().z);
    EXPECT_EQ(4.0, solver.gridSpacing().x);
    EXPECT_EQ(5.0, solver.gridSpacing().y);
    EXPECT_EQ(6.0, solver.gridSpacing().z);
    EXPECT_EQ(7.0, solver.gridOrigin().x);
    EXPECT_EQ(8.0, solver.gridOrigin().y);
    EXPECT_EQ(9.0, solver.gridOrigin().z);
}

TEST(GridFluidSolver3, MinimumResolution) {
    GridFluidSolver3 solver;

    solver.resizeGrid(Size3(1, 1, 1), Vector3D(1.0, 1.0, 1.0), Vector3D());
    solver.velocity()->fill(Vector3D());

    Frame frame(0, 1.0 / 60.0);
    frame.timeIntervalInSeconds = 0.01;
    solver.update(frame);
}

TEST(GridFluidSolver3, GravityOnly) {
    GridFluidSolver3 solver;
    solver.setGravity(Vector3D(0, -10, 0.0));
    solver.setAdvectionSolver(nullptr);
    solver.setDiffusionSolver(nullptr);
    solver.setPressureSolver(nullptr);

    solver.resizeGrid(
        Size3(3, 3, 3),
        Vector3D(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        Vector3D());
    solver.velocity()->fill(Vector3D());

    Frame frame(0, 0.01);
    solver.update(frame);

    solver.velocity()->forEachUIndex([&](size_t i, size_t j, size_t k) {
        EXPECT_NEAR(0.0, solver.velocity()->u(i, j, k), 1e-8);
    });

    solver.velocity()->forEachVIndex([&](size_t i, size_t j, size_t k) {
        if (j == 0 || j == 3) {
            EXPECT_NEAR(0.0, solver.velocity()->v(i, j, k), 1e-8);
        } else {
            EXPECT_NEAR(-0.1, solver.velocity()->v(i, j, k), 1e-8);
        }
    });

    solver.velocity()->forEachWIndex([&](size_t i, size_t j, size_t k) {
        EXPECT_NEAR(0.0, solver.velocity()->w(i, j, k), 1e-8);
    });
}
