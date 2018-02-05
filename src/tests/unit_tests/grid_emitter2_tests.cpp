// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/custom_vector_field2.h>
#include <jet/level_set_utils.h>
#include <jet/sphere2.h>
#include <jet/volume_grid_emitter2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(VolumeGridEmitter2, Velocity) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.75})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(sphere)
        .makeShared();

    auto grid = CellCenteredVectorGrid2::builder()
        .withResolution({16, 16})
        .withGridSpacing({1.0/16.0, 1.0/16.0})
        .withOrigin({0, 0})
        .makeShared();

    auto mapper = [] (double sdf, const Vector2D& pt, const Vector2D& oldVal) {
        if (sdf < 0.0) {
            return Vector2D(pt.y, -pt.x);
        } else {
            return Vector2D(oldVal);
        }
    };

    emitter->addTarget(grid, mapper);

    emitter->update(0.0, 0.01);

    auto pos = grid->dataPosition();
    grid->forEachDataPointIndex([&] (size_t i, size_t j) {
        Vector2D gx = pos(i, j);
        double sdf = emitter->sourceRegion()->signedDistance(gx);
        if (isInsideSdf(sdf)) {
            Vector2D answer{gx.y, -gx.x};
            Vector2D acttual = (*grid)(i, j);

            EXPECT_NEAR(answer.x, acttual.x, 1e-6);
            EXPECT_NEAR(answer.y, acttual.y, 1e-6);
        }
    });
}

TEST(VolumeGridEmitter2, SignedDistance) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.75})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(sphere)
        .makeShared();

    auto grid = CellCenteredScalarGrid2::builder()
        .withResolution({16, 16})
        .withGridSpacing({1.0/16.0, 1.0/16.0})
        .withOrigin({0, 0})
        .withInitialValue(kMaxD)
        .makeShared();

    emitter->addSignedDistanceTarget(grid);

    emitter->update(0.0, 0.01);

    auto pos = grid->dataPosition();
    grid->forEachDataPointIndex([&] (size_t i, size_t j) {
        Vector2D gx = pos(i, j);
        double answer = (sphere->center - gx).length() - 0.15;
        double acttual = (*grid)(i, j);

        EXPECT_NEAR(answer, acttual, 1e-6);
    });
}

TEST(VolumeGridEmitter2, StepFunction) {
    auto sphere = Sphere2::builder()
        .withCenter({0.5, 0.75})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeGridEmitter2::builder()
        .withSourceRegion(sphere)
        .makeShared();

    auto grid = CellCenteredScalarGrid2::builder()
        .withResolution({16, 16})
        .withGridSpacing({1.0/16.0, 1.0/16.0})
        .withOrigin({0, 0})
        .makeShared();

    emitter->addStepFunctionTarget(grid, 3.0, 7.0);

    emitter->update(0.0, 0.01);

    auto pos = grid->dataPosition();
    grid->forEachDataPointIndex([&] (size_t i, size_t j) {
        Vector2D gx = pos(i, j);
        double answer = (sphere->center - gx).length() - 0.15;
        answer = 4.0 * (1.0 - smearedHeavisideSdf(answer * 16.0)) + 3.0;
        double acttual = (*grid)(i, j);

        EXPECT_NEAR(answer, acttual, 1e-6);
    });
}
