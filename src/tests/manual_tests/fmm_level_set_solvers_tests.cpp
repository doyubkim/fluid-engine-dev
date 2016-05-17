// Copyright (c) 2016 Doyub Kim

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

    sdf.fill([](const Vector2D& x) {
        return (x - Vector2D(20, 20)).length() - 8.0;
    });

    FmmLevelSetSolver2 solver;
    solver.reinitialize(sdf, 20.0 /* * gridSpacing.x */, &temp);

    saveData(sdf.constDataAccessor(), "sdf_#grid2,iso.npy");
    saveData(temp.constDataAccessor(), "temp_#grid2,iso.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(FmmLevelSetSolver2, Reinitialize) {
    Size2 size(160, 120);
    Vector2D gridSpacing(1.0/size.x, 1.0/size.x);
    double maxDistance = 10.0 * gridSpacing.x;

    FmmLevelSetSolver2 solver;
    CellCenteredScalarGrid2 data(size, gridSpacing);
    CellCenteredScalarGrid2 buffer(size, gridSpacing);

    // Starting from constant field
    data.fill(1.0);

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "constant_#grid2,iso.npy");

    // Starting from unit-step function
    data.fill([gridSpacing](const Vector2D& x) {
        double r = (x - Vector2D(0.75, 0.5)).length() - 0.3;
        if (r < 0.0) {
            return -0.5 * gridSpacing.x;
        } else {
            return 0.5 * gridSpacing.x;
        }
    });

    saveData(data.constDataAccessor(), "unit_step0_#grid2,iso.npy");

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "unit_step1_#grid2,iso.npy");

    data.swap(&buffer);

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "unit_step2_#grid2,iso.npy");

    // Starting from SDF
    data.fill([gridSpacing](const Vector2D& x) {
        return (x - Vector2D(0.75, 0.5)).length() - 0.3;
    });

    saveData(data.constDataAccessor(), "sdf0_#grid2,iso.npy");

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "sdf1_#grid2,iso.npy");

    data.swap(&buffer);

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "sdf2_#grid2,iso.npy");

    // Starting from scaled SDF
    data.fill([gridSpacing](const Vector2D& x) {
        return 4.0 * ((x - Vector2D(0.75, 0.5)).length() - 0.3);
    });

    saveData(data.constDataAccessor(), "sdf_scaled0_#grid2,iso.npy");

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "sdf_scaled1_#grid2,iso.npy");

    data.swap(&buffer);

    solver.reinitialize(data, maxDistance, &buffer);

    saveData(buffer.constDataAccessor(), "sdf_scaled2_#grid2,iso.npy");
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
