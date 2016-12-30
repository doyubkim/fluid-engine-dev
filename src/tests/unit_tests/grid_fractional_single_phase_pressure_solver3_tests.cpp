// Copyright (c) 2016 Doyub Kim

#include <jet/cell_centered_scalar_grid3.h>
#include <jet/face_centered_grid3.h>
#include <jet/grid_fractional_single_phase_pressure_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridFractionalSinglePhasePressureSolver3, SolveFreeSurface) {
    FaceCenteredGrid3 vel(3, 3, 3);
    CellCenteredScalarGrid3 fluidSdf(3, 3, 3);

    vel.fill(Vector3D());

    for (size_t k = 0; k < 3; ++k) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                if (j == 0 || j == 3) {
                    vel.v(i, j, k) = 0.0;
                } else {
                    vel.v(i, j, k) = 1.0;
                }
            }
        }
    }

    fluidSdf.fill([&](const Vector3D& x) {
        return x.y - 2.0;
    });

    GridFractionalSinglePhasePressureSolver3 solver;
    solver.solve(
        vel,
        1.0,
        &vel,
        ConstantScalarField3(kMaxD),
        ConstantVectorField3({0, 0, 0}),
        fluidSdf);

    for (size_t k = 0; k < 3; ++k) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 4; ++i) {
                EXPECT_NEAR(0.0, vel.u(i, j, k), 1e-6);
            }
        }
    }

    for (size_t k = 0; k < 3; ++k) {
        for (size_t j = 0; j < 4; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                EXPECT_NEAR(0.0, vel.v(i, j, k), 1e-6);
            }
        }
    }

    for (size_t k = 0; k < 4; ++k) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                EXPECT_NEAR(0.0, vel.w(i, j, k), 1e-6);
            }
        }
    }

    const auto& pressure = solver.pressure();
    for (size_t k = 0; k < 3; ++k) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                double p = static_cast<double>(1.5 - j);
                EXPECT_NEAR(p, pressure(i, j, k), 1e-6);
            }
        }
    }
}
