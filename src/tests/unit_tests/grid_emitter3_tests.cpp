// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid3.h>
#include <jet/cell_centered_vector_grid3.h>
#include <jet/custom_vector_field3.h>
#include <jet/level_set_utils.h>
#include <jet/sphere3.h>
#include <jet/volume_grid_emitter3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(VolumeGridEmitter3, Velocity) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.75, 0.5})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(sphere)
        .makeShared();

    auto grid = CellCenteredVectorGrid3::builder()
        .withResolution({16, 16, 16})
        .withGridSpacing({1.0/16.0, 1.0/16.0, 1.0/16.0})
        .withOrigin({0, 0, 0})
        .makeShared();

    auto mapper = [] (double sdf, const Vector3D& pt, const Vector3D& oldVal) {
        if (sdf < 0.0) {
            return Vector3D(pt.y, -pt.x, 3.5);
        } else {
            return Vector3D(oldVal);
        }
    };

    emitter->addTarget(grid, mapper);

    emitter->update(0.0, 0.01);

    auto pos = grid->dataPosition();
    grid->forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        Vector3D gx = pos(i, j, k);
        double sdf = emitter->sourceRegion()->signedDistance(gx);
        if (isInsideSdf(sdf)) {
            Vector3D answer{gx.y, -gx.x, 3.5};
            Vector3D acttual = (*grid)(i, j, k);

            EXPECT_NEAR(answer.x, acttual.x, 1e-6);
            EXPECT_NEAR(answer.y, acttual.y, 1e-6);
        }
    });
}

TEST(VolumeGridEmitter3, SignedDistance) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.75, 0.5})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(sphere)
        .makeShared();

    auto grid = CellCenteredScalarGrid3::builder()
        .withResolution({16, 16, 16})
        .withGridSpacing({1.0/16.0, 1.0/16.0, 1.0/16.0})
        .withOrigin({0, 0, 0})
        .withInitialValue(kMaxD)
        .makeShared();

    emitter->addSignedDistanceTarget(grid);

    emitter->update(0.0, 0.01);

    auto pos = grid->dataPosition();
    grid->forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        Vector3D gx = pos(i, j, k);
        double answer = (sphere->center - gx).length() - 0.15;
        double acttual = (*grid)(i, j, k);

        EXPECT_NEAR(answer, acttual, 1e-6);
    });
}

TEST(VolumeGridEmitter3, StepFunction) {
    auto sphere = Sphere3::builder()
        .withCenter({0.5, 0.75, 0.5})
        .withRadius(0.15)
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(sphere)
        .makeShared();

    auto grid = CellCenteredScalarGrid3::builder()
        .withResolution({16, 16, 16})
        .withGridSpacing({1.0/16.0, 1.0/16.0, 1.0/16.0})
        .withOrigin({0, 0, 0})
        .makeShared();

    emitter->addStepFunctionTarget(grid, 3.0, 7.0);

    emitter->update(0.0, 0.01);

    auto pos = grid->dataPosition();
    grid->forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        Vector3D gx = pos(i, j, k);
        double answer = (sphere->center - gx).length() - 0.15;
        answer = 4.0 * (1.0 - smearedHeavisideSdf(answer * 16.0)) + 3.0;
        double acttual = (*grid)(i, j, k);

        EXPECT_NEAR(answer, acttual, 1e-6);
    });
}
