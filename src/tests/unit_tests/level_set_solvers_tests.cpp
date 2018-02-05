// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_utils.h>
#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/eno_level_set_solver2.h>
#include <jet/eno_level_set_solver3.h>
#include <jet/fdm_utils.h>
#include <jet/fmm_level_set_solver2.h>
#include <jet/fmm_level_set_solver3.h>
#include <jet/upwind_level_set_solver2.h>
#include <jet/upwind_level_set_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(UpwindLevelSetSolver2, Reinitialize) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });

    UpwindLevelSetSolver2 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            EXPECT_NEAR(sdf(i, j), temp(i, j), 0.5);
        }
    }
}

TEST(UpwindLevelSetSolver2, Extrapolate) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);
    CellCenteredScalarGrid2 field(40, 30);

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });
    field.fill(5.0);

    UpwindLevelSetSolver2 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            EXPECT_DOUBLE_EQ(5.0, temp(i, j));
        }
    }
}

TEST(UpwindLevelSetSolver3, Reinitialize) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });

    UpwindLevelSetSolver3 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    for (size_t k = 0; k < 50; ++k) {
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < 40; ++i) {
                EXPECT_NEAR(sdf(i, j, k), temp(i, j, k), 0.7)
                    << i << ", " << j << ", " << k;
            }
        }
    }
}

TEST(UpwindLevelSetSolver3, Extrapolate) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);
    CellCenteredScalarGrid3 field(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });
    field.fill(5.0);

    UpwindLevelSetSolver3 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    for (size_t k = 0; k < 50; ++k) {
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < 40; ++i) {
                EXPECT_DOUBLE_EQ(5.0, temp(i, j, k))
                    << i << ", " << j << ", " << k;
            }
        }
    }
}


TEST(EnoLevelSetSolver2, Reinitialize) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });

    EnoLevelSetSolver2 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            EXPECT_NEAR(sdf(i, j), temp(i, j), 0.2);
        }
    }
}

TEST(EnoLevelSetSolver2, Extrapolate) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);
    CellCenteredScalarGrid2 field(40, 30);

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });
    field.fill(5.0);

    EnoLevelSetSolver2 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            EXPECT_DOUBLE_EQ(5.0, temp(i, j));
        }
    }
}

TEST(EnoLevelSetSolver3, Reinitialize) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });

    EnoLevelSetSolver3 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    for (size_t k = 0; k < 50; ++k) {
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < 40; ++i) {
                EXPECT_NEAR(sdf(i, j, k), temp(i, j, k), 0.5)
                    << i << ", " << j << ", " << k;;
            }
        }
    }
}

TEST(EnoLevelSetSolver3, Extrapolate) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);
    CellCenteredScalarGrid3 field(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });
    field.fill(5.0);

    EnoLevelSetSolver3 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    for (size_t k = 0; k < 50; ++k) {
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < 40; ++i) {
                EXPECT_DOUBLE_EQ(5.0, temp(i, j, k))
                    << i << ", " << j << ", " << k;;
            }
        }
    }
}


TEST(FmmLevelSetSolver2, Reinitialize) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });

    FmmLevelSetSolver2 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            EXPECT_NEAR(sdf(i, j), temp(i, j), 0.6);
        }
    }
}

TEST(FmmLevelSetSolver2, Extrapolate) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);
    CellCenteredScalarGrid2 field(40, 30);

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });
    field.fill(5.0);

    FmmLevelSetSolver2 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            EXPECT_DOUBLE_EQ(5.0, temp(i, j));
        }
    }
}

TEST(FmmLevelSetSolver3, Reinitialize) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });

    FmmLevelSetSolver3 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    for (size_t k = 0; k < 50; ++k) {
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < 40; ++i) {
                EXPECT_NEAR(sdf(i, j, k), temp(i, j, k), 0.9)
                    << i << ", " << j << ", " << k;;
            }
        }
    }
}

TEST(FmmLevelSetSolver3, Extrapolate) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);
    CellCenteredScalarGrid3 field(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });
    field.fill(5.0);

    FmmLevelSetSolver3 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    for (size_t k = 0; k < 50; ++k) {
        for (size_t j = 0; j < 30; ++j) {
            for (size_t i = 0; i < 40; ++i) {
                EXPECT_DOUBLE_EQ(5.0, temp(i, j, k))
                    << i << ", " << j << ", " << k;;
            }
        }
    }
}
