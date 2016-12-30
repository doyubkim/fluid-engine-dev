// Copyright (c) 2016 Doyub Kim

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/face_centered_grid2.h>
#include <jet/grid_single_phase_pressure_solver2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(GridSinglePhasePressureSolver2, SolveSinglePhase) {
    FaceCenteredGrid2 vel(3, 3);

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

    GridSinglePhasePressureSolver2 solver;
    solver.solve(vel, 1.0, &vel);

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
    for (size_t j = 0; j < 2; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_NEAR(pressure(i, j + 1) - pressure(i, j), -1.0, 1e-6);
        }
    }
}

TEST(GridSinglePhasePressureSolver2, SolveSinglePhaseWithBoundary) {
    FaceCenteredGrid2 vel(3, 3);
    CellCenteredScalarGrid2 boundarySdf(3, 3);

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

    // Wall on the right-most column
    boundarySdf.fill([&](const Vector2D& x) {
        return -x.x + 2.0;
    });

    GridSinglePhasePressureSolver2 solver;
    solver.solve(vel, 1.0, &vel, boundarySdf);

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(0.0, vel.u(i, j), 1e-6);
        }
    }

    for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            if (i == 2 && (j == 1 || j == 2)) {
                EXPECT_NEAR(1.0, vel.v(i, j), 1e-6);
            } else {
                EXPECT_NEAR(0.0, vel.v(i, j), 1e-6);
            }
        }
    }

    const auto& pressure = solver.pressure();
    for (size_t j = 0; j < 2; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_NEAR(pressure(i, j + 1) - pressure(i, j), -1.0, 1e-6);
        }
    }
}

TEST(GridSinglePhasePressureSolver2, SolveFreeSurface) {
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

    GridSinglePhasePressureSolver2 solver;
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
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            double p = static_cast<double>(2 - j);
            EXPECT_NEAR(p, pressure(i, j), 1e-6);
        }
    }
}

TEST(GridSinglePhasePressureSolver2, SolveFreeSurfaceWithBoundary) {
    FaceCenteredGrid2 vel(3, 3);
    CellCenteredScalarGrid2 fluidSdf(3, 3);
    CellCenteredScalarGrid2 boundarySdf(3, 3);

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

    // Wall on the right-most column
    boundarySdf.fill([&](const Vector2D& x) {
        return -x.x + 2.0;
    });
    fluidSdf.fill([&](const Vector2D& x) {
        return x.y - 2.0;
    });

    GridSinglePhasePressureSolver2 solver;
    solver.solve(
        vel,
        1.0,
        &vel,
        boundarySdf,
        ConstantVectorField2({0, 0}),
        fluidSdf);

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_NEAR(0.0, vel.u(i, j), 1e-6);
        }
    }

    for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            if (i == 2 && (j == 1 || j == 2)) {
                EXPECT_NEAR(1.0, vel.v(i, j), 1e-6);
            } else {
                EXPECT_NEAR(0.0, vel.v(i, j), 1e-6);
            }
        }
    }

    const auto& pressure = solver.pressure();
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            double p = static_cast<double>(2 - j);
            EXPECT_NEAR(p, pressure(i, j), 1e-6);
        }
    }
}
