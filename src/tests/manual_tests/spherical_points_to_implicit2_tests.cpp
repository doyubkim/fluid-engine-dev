// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/spherical_points_to_implicit2.h>

#include <random>

using namespace jet;

JET_TESTS(SphericalPointsToImplicit2);

JET_BEGIN_TEST_F(SphericalPointsToImplicit2, ConvertTwo) {
    Array1<Vector2D> points;

    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist(0.2, 0.8);
    for (size_t i = 0; i < 2; ++i) {
        points.append({dist(rng), dist(rng)});
    }

    CellCenteredScalarGrid2 grid(512, 512, 1.0 / 512, 1.0 / 512);

    SphericalPointsToImplicit2 converter(0.1);
    converter.convert(points.constAccessor(), &grid);

    saveData(grid.constDataAccessor(), "data_#grid2.npy");
    saveData(grid.constDataAccessor(), "data_#grid2,iso.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(SphericalPointsToImplicit2, ConvertMany) {
    Array1<Vector2D> points;

    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist(0.2, 0.8);
    for (size_t i = 0; i < 200; ++i) {
        points.append({dist(rng), dist(rng)});
    }

    CellCenteredScalarGrid2 grid(512, 512, 1.0 / 512, 1.0 / 512);

    SphericalPointsToImplicit2 converter(0.1);
    converter.convert(points.constAccessor(), &grid);

    saveData(grid.constDataAccessor(), "data_#grid2.npy");
    saveData(grid.constDataAccessor(), "data_#grid2,iso.npy");
}
JET_END_TEST_F
