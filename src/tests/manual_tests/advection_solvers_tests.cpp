// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array3.h>
#include <jet/box2.h>
#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/constant_vector_field2.h>
#include <jet/constants.h>
#include <jet/cubic_semi_lagrangian2.h>
#include <jet/custom_scalar_field2.h>
#include <jet/custom_vector_field2.h>
#include <jet/semi_lagrangian2.h>

#include <algorithm>

using namespace jet;

JET_TESTS(SemiLagrangian2);

JET_BEGIN_TEST_F(SemiLagrangian2, Boundary) {
    CellCenteredVectorGrid2 src(200, 200, 1.0/200.0, 1.0/200.0);
    CellCenteredVectorGrid2 dst(200, 200, 1.0/200.0, 1.0/200.0);
    src.fill([&](const Vector2D& pt) -> Vector2D {
        return {
            0.5 * (std::sin(15 * pt.x) + 1.0),
            0.5 * (std::sin(15 * pt.y) + 1.0)};
    });

    ConstantVectorField2 flow(Vector2D(1.0, 1.0));
    CustomScalarField2 boundarySdf([](const Vector2D& pt) {
        return Vector2D(0.5, 0.5).distanceTo(pt) - 0.25;
    });

    Array3<double> data(3, src.resolution().x, src.resolution().y);
    data.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (i < 2) {
            data(i, j, k) = src(j, k)[i];
        }
    });
    saveData(data.constAccessor(), "src_#grid2.npy");

    SemiLagrangian2 solver;
    solver.advect(src, flow, 0.1, &dst, boundarySdf);

    data.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (i < 2) {
            data(i, j, k) = dst(j, k)[i];
        }
    });
    saveData(data.constAccessor(), "dst_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(SemiLagrangian2, Zalesak) {
    Box2 box(Vector2D(0.5 - 0.025, 0.6), Vector2D(0.5 + 0.025, 0.85));
    CellCenteredScalarGrid2 sdf(200, 200, 1.0/200.0, 1.0/200.0);
    CellCenteredScalarGrid2 sdf2(200, 200, 1.0/200.0, 1.0/200.0);
    sdf.fill([box](const Vector2D& pt) {
        double disk = pt.distanceTo(Vector2D(0.5, 0.75)) - 0.15;
        double slot = box.closestDistance(pt);
        if (!box.boundingBox().contains(pt)) {
            slot *= -1.0;
        }
        return std::max(disk, slot);
    });

    CustomVectorField2 flow([](const Vector2D& pt) {
        return Vector2D(kPiD / 3.14 * (0.5 - pt.y), kPiD / 3.14 * (pt.x - 0.5));
    });

    saveData(sdf.constDataAccessor(), "orig_#grid2,iso.npy");

    SemiLagrangian2 solver;

    for (int i = 0; i < 628; ++i) {
        solver.advect(sdf, flow, 0.02, &sdf2);
        sdf.swap(&sdf2);
    }

    saveData(sdf.constDataAccessor(), "rev0628_#grid2,iso.npy");
}
JET_END_TEST_F

JET_TESTS(CubicSemiLagrangian2);

JET_BEGIN_TEST_F(CubicSemiLagrangian2, Zalesak) {
    Box2 box(Vector2D(0.5 - 0.025, 0.6), Vector2D(0.5 + 0.025, 0.85));
    CellCenteredScalarGrid2 sdf(200, 200, 1.0/200.0, 1.0/200.0);
    CellCenteredScalarGrid2 sdf2(200, 200, 1.0/200.0, 1.0/200.0);
    sdf.fill([box](const Vector2D& pt) {
        double disk = pt.distanceTo(Vector2D(0.5, 0.75)) - 0.15;
        double slot = box.closestDistance(pt);
        if (!box.boundingBox().contains(pt)) {
            slot *= -1.0;
        }
        return std::max(disk, slot);
    });

    CustomVectorField2 flow([](const Vector2D& pt) {
        return Vector2D(kPiD / 3.14 * (0.5 - pt.y), kPiD / 3.14 * (pt.x - 0.5));
    });

    saveData(sdf.constDataAccessor(), "orig_#grid2,iso.npy");

    CubicSemiLagrangian2 solver;

    for (int i = 0; i < 628; ++i) {
        solver.advect(sdf, flow, 0.02, &sdf2);
        sdf.swap(&sdf2);
    }

    saveData(sdf.constDataAccessor(), "rev0628_#grid2,iso.npy");
}
JET_END_TEST_F
