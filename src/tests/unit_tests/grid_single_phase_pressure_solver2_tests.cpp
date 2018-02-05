// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/face_centered_grid2.h>
#include <jet/fdm_mg_solver2.h>
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
    solver.solve(vel, 1.0, &vel, ConstantScalarField2(kMaxD),
                 ConstantVectorField2({0, 0}), ConstantScalarField2(-kMaxD),
                 false);

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

TEST(GridSinglePhasePressureSolver2, SolveSinglePhaseCompressed) {
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
    solver.solve(vel, 1.0, &vel, ConstantScalarField2(kMaxD),
                 ConstantVectorField2({0, 0}), ConstantScalarField2(-kMaxD),
                 true);

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
    boundarySdf.fill([&](const Vector2D& x) { return -x.x + 2.0; });

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

    fluidSdf.fill([&](const Vector2D& x) { return x.y - 2.0; });

    GridSinglePhasePressureSolver2 solver;
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
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 3; ++i) {
            double p = static_cast<double>(2 - j);
            EXPECT_NEAR(p, pressure(i, j), 1e-6);
        }
    }
}

TEST(GridSinglePhasePressureSolver2, SolveFreeSurfaceCompressed) {
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

    GridSinglePhasePressureSolver2 solver;
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
    boundarySdf.fill([&](const Vector2D& x) { return -x.x + 2.0; });
    fluidSdf.fill([&](const Vector2D& x) { return x.y - 2.0; });

    GridSinglePhasePressureSolver2 solver;
    solver.solve(vel, 1.0, &vel, boundarySdf, ConstantVectorField2({0, 0}),
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

TEST(GridSinglePhasePressureSolver2, SolveSinglePhaseWithMg) {
    size_t n = 64;
    FaceCenteredGrid2 vel(n, n);

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n + 1; ++i) {
            vel.u(i, j) = 0.0;
        }
    }

    for (size_t j = 0; j < n + 1; ++j) {
        for (size_t i = 0; i < n; ++i) {
            if (j == 0 || j == n) {
                vel.v(i, j) = 0.0;
            } else {
                vel.v(i, j) = 1.0;
            }
        }
    }

    GridSinglePhasePressureSolver2 solver;
    solver.setLinearSystemSolver(
        std::make_shared<FdmMgSolver2>(5, 10, 10, 40, 10));

    solver.solve(vel, 1.0, &vel);

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n + 1; ++i) {
            EXPECT_NEAR(0.0, vel.u(i, j), 0.01);
        }
    }

    for (size_t j = 0; j < n + 1; ++j) {
        for (size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(0.0, vel.v(i, j), 0.05);
        }
    }
}
