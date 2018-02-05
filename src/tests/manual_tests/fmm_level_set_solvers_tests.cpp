// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array_utils.h>
#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/fmm_level_set_solver2.h>
#include <jet/fmm_level_set_solver3.h>

using namespace jet;

JET_TESTS(FmmLevelSetSolver2);

JET_BEGIN_TEST_F(FmmLevelSetSolver2, ReinitializeSmall) {
    CellCenteredScalarGrid2 sdf(40, 30), temp(40, 30);
    FmmLevelSetSolver2 solver;

    // Starting from constant field
    sdf.fill([](const Vector2D& x) {
        return 1.0;
    });
    saveData(sdf.constDataAccessor(), "constant0_#grid2,iso.npy");

    solver.reinitialize(sdf, 40.0, &temp);

    saveData(temp.constDataAccessor(), "constant1_#grid2,iso.npy");

    // Starting from SDF field
    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });
    saveData(sdf.constDataAccessor(), "sdf0_#grid2,iso.npy");

    solver.reinitialize(sdf, 40.0, &temp);

    saveData(temp.constDataAccessor(), "sdf1_#grid2,iso.npy");

    // Starting from scaled SDF field
    sdf.fill([](const Vector2D& x) {
        double r = (x - Vector2D(20, 20)).length() - 8.0;
        return 2.0 * r;
    });
    saveData(sdf.constDataAccessor(), "scaled0_#grid2,iso.npy");

    solver.reinitialize(sdf, 40.0, &temp);

    saveData(temp.constDataAccessor(), "scaled1_#grid2,iso.npy");

    // Starting from scaled SDF field
    sdf.fill([](const Vector2D& x) {
        double r = (x - Vector2D(20, 20)).length() - 8.0;
        return (r < 0.0) ? -0.5 : 0.5;
    });
    saveData(sdf.constDataAccessor(), "unit_step0_#grid2,iso.npy");

    solver.reinitialize(sdf, 40.0, &temp);

    saveData(temp.constDataAccessor(), "unit_step1_#grid2,iso.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FmmLevelSetSolver2, Reinitialize) {
    CellCenteredScalarGrid2 sdf(160, 120), temp(160, 120);
    FmmLevelSetSolver2 solver;

    // Starting from constant field
    sdf.fill([](const Vector2D& x) {
        return 1.0;
    });
    saveData(sdf.constDataAccessor(), "constant0_#grid2,iso.npy");

    solver.reinitialize(sdf, 160.0, &temp);

    saveData(temp.constDataAccessor(), "constant1_#grid2,iso.npy");

    // Starting from SDF field
    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(80, 80)).length() - 32.0;
    });
    saveData(sdf.constDataAccessor(), "sdf0_#grid2,iso.npy");

    solver.reinitialize(sdf, 160.0, &temp);

    saveData(temp.constDataAccessor(), "sdf1_#grid2,iso.npy");

    // Starting from scaled SDF field
    sdf.fill([](const Vector2D& x) {
        double r = (x - Vector2D(80, 80)).length() - 32.0;
        return 2.0 * r;
    });
    saveData(sdf.constDataAccessor(), "scaled0_#grid2,iso.npy");

    solver.reinitialize(sdf, 160.0, &temp);

    saveData(temp.constDataAccessor(), "scaled1_#grid2,iso.npy");

    // Starting from scaled SDF field
    sdf.fill([](const Vector2D& x) {
        double r = (x - Vector2D(80, 80)).length() - 32.0;
        return (r < 0.0) ? -0.5 : 0.5;
    });
    saveData(sdf.constDataAccessor(), "unit_step0_#grid2,iso.npy");

    solver.reinitialize(sdf, 160.0, &temp);

    saveData(temp.constDataAccessor(), "unit_step1_#grid2,iso.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FmmLevelSetSolver2, Extrapolate) {
    Size2 size(160, 120);
    Vector2D gridSpacing(1.0/size.x, 1.0/size.x);
    double maxDistance = 20.0 * gridSpacing.x;

    FmmLevelSetSolver2 solver;
    CellCenteredScalarGrid2 sdf(size, gridSpacing);
    CellCenteredScalarGrid2 input(size, gridSpacing);
    CellCenteredScalarGrid2 output(size, gridSpacing);

    sdf.fill([&](const Vector2D& x) {
        return (x - Vector2D(0.75, 0.5)).length() - 0.3;
    });

    input.fill([&](const Vector2D& x) {
        if ((x - Vector2D(0.75, 0.5)).length() <= 0.3) {
            double p = 10.0 * kPiD;
            return 0.5 * 0.25 * std::sin(p * x.x) * std::sin(p * x.y);
        } else {
            return 0.0;
        }
    });

    solver.extrapolate(input, sdf, maxDistance, &output);

    saveData(sdf.constDataAccessor(), "sdf_#grid2,iso.npy");
    saveData(input.constDataAccessor(), "input_#grid2.npy");
    saveData(output.constDataAccessor(), "output_#grid2.npy");
}
JET_END_TEST_F


JET_TESTS(FmmLevelSetSolver3);

JET_BEGIN_TEST_F(FmmLevelSetSolver3, ReinitializeSmall) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });

    FmmLevelSetSolver3 solver;
    solver.reinitialize(sdf, 5.0, &temp);

    Array2<double> sdf2(40, 30);
    Array2<double> temp2(40, 30);
    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            sdf2(i, j) = sdf(i, j, 10);
            temp2(i, j) = temp(i, j, 10);
        }
    }

    saveData(sdf2.constAccessor(), "sdf_#grid2,iso.npy");
    saveData(temp2.constAccessor(), "temp_#grid2,iso.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FmmLevelSetSolver3, ExtrapolateSmall) {
    CellCenteredScalarGrid3 sdf(40, 30, 50), temp(40, 30, 50);
    CellCenteredScalarGrid3 field(40, 30, 50);

    sdf.fill([](const Vector3D& x) {
        return (x - Vector3D(20, 20, 20)).length() - 8.0;
    });
    field.fill([&](const Vector3D& x) {
        if ((x - Vector3D(20, 20, 20)).length() <= 8.0) {
            return 0.5 * 0.25 * std::sin(x.x) * std::sin(x.y) * std::sin(x.z);
        } else {
            return 0.0;
        }
    });

    FmmLevelSetSolver3 solver;
    solver.extrapolate(field, sdf, 5.0, &temp);

    Array2<double> field2(40, 30);
    Array2<double> temp2(40, 30);
    for (size_t j = 0; j < 30; ++j) {
        for (size_t i = 0; i < 40; ++i) {
            field2(i, j) = field(i, j, 12);
            temp2(i, j) = temp(i, j, 12);
        }
    }

    saveData(field2.constAccessor(), "field_#grid2.npy");
    saveData(temp2.constAccessor(), "temp_#grid2.npy");
}
JET_END_TEST_F
