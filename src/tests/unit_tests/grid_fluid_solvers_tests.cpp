// Copyright (c) 2016 Doyub Kim

#include <jet/grid_fluid_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridFluidSolver2, MinimumResolution) {
    GridFluidSolver2 solver;

    GridSystemData2Ptr data = solver.gridSystemData();
    data->resize(Size2(1, 1), Vector2D(1.0, 1.0), Vector2D());
    data->velocity()->fill(Vector2D());

    Frame frame(1, 1.0 / 60.0);
    frame.timeIntervalInSeconds = 0.01;
    solver.update(frame);
}

TEST(GridFluidSolver2, GravityOnly) {
    GridFluidSolver2 solver;
    solver.setGravity(Vector2D(0, -10));
    solver.setAdvectionSolver(nullptr);
    solver.setDiffusionSolver(nullptr);
    solver.setPressureSolver(nullptr);

    GridSystemData2Ptr data = solver.gridSystemData();
    data->resize(Size2(3, 3), Vector2D(1.0 / 3.0, 1.0 / 3.0), Vector2D());
    data->velocity()->fill(Vector2D());

    Frame frame(1, 1.0 / 60.0);
    frame.timeIntervalInSeconds = 0.01;
    solver.update(frame);

    data->velocity()->forEachUIndex([&](size_t i, size_t j) {
        EXPECT_NEAR(0.0, data->velocity()->u(i, j), 1e-8);
    });

    data->velocity()->forEachVIndex([&](size_t i, size_t j) {
        if (j == 0 || j == 3) {
            EXPECT_NEAR(0.0, data->velocity()->v(i, j), 1e-8);
        } else {
            EXPECT_NEAR(-0.1, data->velocity()->v(i, j), 1e-8);
        }
    });
}
