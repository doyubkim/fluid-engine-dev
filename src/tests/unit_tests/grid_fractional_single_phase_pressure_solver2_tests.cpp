// Copyright (c) 2016 Doyub Kim

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/face_centered_grid2.h>
#include <jet/grid_fractional_single_phase_pressure_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridFractionalSinglePhasePressureSolver2, SolveFreeSurface) {
    FaceCenteredGrid2 vel(3, 3);
    CellCenteredScalarGrid2 fluidSdf(3, 3);

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            vel.u(i, j) = 0.0;
        }
    }

    for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            if (j == 0 || j == 3) {
                vel.v(i, j) = 0.0;
            } else {
                vel.v(i, j) = 1.0;
            }
        }
    }

    fluidSdf.fill([&](const Vector2D& x) {
        return x.y - 2.0;
    });

    GridFractionalSinglePhasePressureSolver2 solver;
    solver.solve(
        vel,
        1.0,
        &vel,
        ConstantScalarField2(kMaxD),
        ConstantVectorField2({0, 0}),
        fluidSdf);

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(0.0, vel.u(i, j), 1e-6);
        }
    }

    for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(0.0, vel.v(i, j), 1e-6);
        }
    }

    const auto& pressure = solver.pressure();
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(1.5, pressure(i, 0), 1e-6);
        EXPECT_NEAR(0.5, pressure(i, 1), 1e-6);
        EXPECT_NEAR(0.0, pressure(i, 2), 1e-6);
    }
}
