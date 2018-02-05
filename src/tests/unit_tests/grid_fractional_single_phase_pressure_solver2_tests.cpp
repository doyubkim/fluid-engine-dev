// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/face_centered_grid2.h>
#include <jet/fdm_mg_solver2.h>
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

    fluidSdf.fill([&](const Vector2D& x) { return x.y - 2.0; });

    GridFractionalSinglePhasePressureSolver2 solver;
    solver.solve(vel, 1.0, &vel, ConstantScalarField2(kMaxD),
                 ConstantVectorField2({0, 0}), fluidSdf);

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

TEST(GridFractionalSinglePhasePressureSolver2, SolveFreeSurfaceCompressed) {
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

    fluidSdf.fill([&](const Vector2D& x) { return x.y - 2.0; });

    GridFractionalSinglePhasePressureSolver2 solver;
    solver.solve(vel, 1.0, &vel, ConstantScalarField2(kMaxD),
                 ConstantVectorField2({0, 0}), fluidSdf, true);

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

TEST(GridFractionalSinglePhasePressureSolver2, SolveFreeSurfaceMg) {
    FaceCenteredGrid2 vel(32, 32);
    CellCenteredScalarGrid2 fluidSdf(32, 32);

    for (size_t j = 0; j < 32; ++j) {
        for (size_t i = 0; i < 33; ++i) {
            vel.u(i, j) = 0.0;
        }
    }

    for (size_t j = 0; j < 33; ++j) {
        for (size_t i = 0; i < 32; ++i) {
            if (j == 0 || j == 32) {
                vel.v(i, j) = 0.0;
            } else {
                vel.v(i, j) = 1.0;
            }
        }
    }

    fluidSdf.fill([&](const Vector2D& x) { return x.y - 16.0; });

    GridFractionalSinglePhasePressureSolver2 solver;
    solver.setLinearSystemSolver(
        std::make_shared<FdmMgSolver2>(5, 50, 50, 50, 50));
    solver.solve(vel, 1.0, &vel, ConstantScalarField2(kMaxD),
                 ConstantVectorField2({0, 0}), fluidSdf, true);

    for (size_t j = 0; j < 32; ++j) {
        for (size_t i = 0; i < 33; ++i) {
            EXPECT_NEAR(0.0, vel.u(i, j), 0.002);
        }
    }

    for (size_t j = 0; j < 16; ++j) {
        for (size_t i = 0; i < 32; ++i) {
            EXPECT_NEAR(0.0, vel.v(i, j), 0.002);
        }
    }

    const auto& pressure = solver.pressure();
    for (size_t j = 0; j < 32; ++j) {
        for (size_t i = 16; i < 17; ++i) {
            if (j < 16) {
                EXPECT_NEAR(15.5 - static_cast<double>(j), pressure(i, j), 0.1);
            } else {
                EXPECT_NEAR(0.0, pressure(i, j), 0.1);
            }
        }
    }
}
