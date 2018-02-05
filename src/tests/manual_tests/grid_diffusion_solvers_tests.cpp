// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array2.h>
#include <jet/grid_backward_euler_diffusion_solver2.h>
#include <jet/grid_backward_euler_diffusion_solver3.h>
#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/custom_scalar_field2.h>
#include <jet/custom_scalar_field3.h>
#include <jet/grid_forward_euler_diffusion_solver2.h>
#include <jet/grid_forward_euler_diffusion_solver3.h>

using namespace jet;

JET_TESTS(GridForwardEulerDiffusionSolver3);

JET_BEGIN_TEST_F(GridForwardEulerDiffusionSolver3, Solve) {
    Size3 size(160, 120, 150);
    Vector3D gridSpacing(1.0/size.x, 1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid3 src(size, gridSpacing);
    CellCenteredScalarGrid3 dst(size, gridSpacing);
    Array2<double> data(160, 120);

    src.fill([&](const Vector3D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    double timeStep = 0.01;
    double diffusionCoeff = square(gridSpacing.x) / timeStep / 12.0;

    GridForwardEulerDiffusionSolver3 diffusionSolver;

    diffusionSolver.solve(src, diffusionCoeff, timeStep, &dst);
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridForwardEulerDiffusionSolver3, Unstable) {
    Size3 size(160, 120, 150);
    Vector3D gridSpacing(1.0/size.x, 1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid3 src(size, gridSpacing);
    CellCenteredScalarGrid3 dst(size, gridSpacing);
    Array2<double> data(160, 120);

    src.fill([&](const Vector3D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    double timeStep = 0.01;
    double diffusionCoeff = square(gridSpacing.x) / timeStep / 12.0;

    GridForwardEulerDiffusionSolver3 diffusionSolver;

    diffusionSolver.solve(
        src,
        10.0 * diffusionCoeff,
        timeStep,
        &dst);
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F


JET_TESTS(GridBackwardEulerDiffusionSolver2);

JET_BEGIN_TEST_F(GridBackwardEulerDiffusionSolver2, Solve) {
    Size2 size(160, 120);
    Vector2D gridSpacing(1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid2 src(size, gridSpacing);
    CellCenteredScalarGrid2 dst(size, gridSpacing);
    Array2<double> data(160, 120);

    src.fill([&](const Vector2D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j);
    });

    double timeStep = 0.01;
    double diffusionCoeff = square(gridSpacing.x) / timeStep / 12.0;

    GridBackwardEulerDiffusionSolver2 diffusionSolver;

    diffusionSolver.solve(
        src,
        100.0 * diffusionCoeff,
        timeStep,
        &dst,
        ConstantScalarField2(kMaxD),
        CustomScalarField2([&](const Vector2D& pt) {
            Vector2D md = src.boundingBox().midPoint();
            return pt.x - md.x;
        }));
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F


JET_TESTS(GridBackwardEulerDiffusionSolver3);

JET_BEGIN_TEST_F(GridBackwardEulerDiffusionSolver3, Solve) {
    Size3 size(160, 120, 150);
    Vector3D gridSpacing(1.0/size.x, 1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid3 src(size, gridSpacing);
    CellCenteredScalarGrid3 dst(size, gridSpacing);
    Array2<double> data(160, 120);

    src.fill([&](const Vector3D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    double timeStep = 0.01;
    double diffusionCoeff = square(gridSpacing.x) / timeStep / 12.0;

    GridBackwardEulerDiffusionSolver3 diffusionSolver;

    diffusionSolver.solve(src, diffusionCoeff, timeStep, &dst);
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridBackwardEulerDiffusionSolver3, Stable) {
    Size3 size(160, 120, 150);
    Vector3D gridSpacing(1.0/size.x, 1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid3 src(size, gridSpacing);
    CellCenteredScalarGrid3 dst(size, gridSpacing);
    Array2<double> data(160, 120);

    src.fill([&](const Vector3D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    double timeStep = 0.01;
    double diffusionCoeff = square(gridSpacing.x) / timeStep / 12.0;

    GridBackwardEulerDiffusionSolver3 diffusionSolver;

    diffusionSolver.solve(
        src,
        10.0 * diffusionCoeff,
        timeStep,
        &dst);
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, 75);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(
    GridBackwardEulerDiffusionSolver3,
    SolveWithBoundaryDirichlet) {
    Size3 size(80, 60, 75);
    Vector3D gridSpacing(1.0/size.x, 1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid3 src(size, gridSpacing);
    CellCenteredScalarGrid3 dst(size, gridSpacing);

    Vector3D boundaryCenter = src.boundingBox().midPoint();
    CustomScalarField3 boundarySdf(
        [&](const Vector3D& x) {
            return boundaryCenter.x - x.x;
        });

    Array2<double> data(size.x, size.y);

    src.fill([&](const Vector3D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, size.z / 2);
    });

    double timeStep = 0.01;
    double diffusionCoeff = 100 * square(gridSpacing.x) / timeStep / 12.0;

    GridBackwardEulerDiffusionSolver3 diffusionSolver(
        GridBackwardEulerDiffusionSolver3::Dirichlet);

    diffusionSolver.solve(src, diffusionCoeff, timeStep, &dst, boundarySdf);
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, size.z / 2);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(GridBackwardEulerDiffusionSolver3, SolveWithBoundaryNeumann) {
    Size3 size(80, 60, 75);
    Vector3D gridSpacing(1.0/size.x, 1.0/size.x, 1.0/size.x);

    CellCenteredScalarGrid3 src(size, gridSpacing);
    CellCenteredScalarGrid3 dst(size, gridSpacing);

    Vector3D boundaryCenter = src.boundingBox().midPoint();
    CustomScalarField3 boundarySdf(
        [&](const Vector3D& x) {
            return boundaryCenter.x - x.x;
        });

    Array2<double> data(size.x, size.y);

    src.fill([&](const Vector3D& x) {
        return (x.distanceTo(src.boundingBox().midPoint()) < 0.2) ? 1.0 : 0.0;
    });

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, size.z / 2);
    });

    double timeStep = 0.01;
    double diffusionCoeff = 100 * square(gridSpacing.x) / timeStep / 12.0;

    GridBackwardEulerDiffusionSolver3 diffusionSolver(
        GridBackwardEulerDiffusionSolver3::Neumann);

    diffusionSolver.solve(src, diffusionCoeff, timeStep, &dst, boundarySdf);
    dst.swap(&src);

    saveData(data.constAccessor(), "src_#grid2.npy");

    data.forEachIndex([&](size_t i, size_t j) {
        data(i, j) = src(i, j, size.z / 2);
    });

    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F
