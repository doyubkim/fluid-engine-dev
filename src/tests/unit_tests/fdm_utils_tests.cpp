// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/cell_centered_vector_grid3.h>
#include <jet/fdm_utils.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(FdmUtils, ScalarToGradient2) {
    CellCenteredScalarGrid2 grid(10, 10, 2.0, 3.0, -1.0, 4.0);
    grid.fill([&](const Vector2D& x) {
        return -5.0 * x.x + 4.0 * x.y;
    });

    Vector2D grad = gradient2(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3);
    EXPECT_DOUBLE_EQ(-5.0, grad.x);
    EXPECT_DOUBLE_EQ(4.0, grad.y);
}

TEST(FdmUtils, VectorToGradient2) {
    CellCenteredVectorGrid2 grid(10, 10, 2.0, 3.0, -1.0, 4.0);
    grid.fill([&](const Vector2D& x) {
        return Vector2D(-5.0 * x.x + 4.0 * x.y, 2.0 * x.x - 7.0 * x.y);
    });

    auto grad = gradient2(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3);
    EXPECT_DOUBLE_EQ(-5.0, grad[0].x);
    EXPECT_DOUBLE_EQ(4.0, grad[0].y);
    EXPECT_DOUBLE_EQ(2.0, grad[1].x);
    EXPECT_DOUBLE_EQ(-7.0, grad[1].y);
}

TEST(FdmUtils, ScalarToGradient3) {
    CellCenteredScalarGrid3 grid(10, 10, 10, 2.0, 3.0, 0.5, -1.0, 4.0, 2.0);
    grid.fill([&](const Vector3D& x) {
        return -5.0 * x.x + 4.0 * x.y + 2.0 * x.z;
    });

    Vector3D grad = gradient3(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3, 4);
    EXPECT_DOUBLE_EQ(-5.0, grad.x);
    EXPECT_DOUBLE_EQ(4.0, grad.y);
    EXPECT_DOUBLE_EQ(2.0, grad.z);
}

TEST(FdmUtils, VectorToGradient3) {
    CellCenteredVectorGrid3 grid(10, 10, 10, 2.0, 3.0, 0.5, -1.0, 4.0, 2.0);
    grid.fill([&](const Vector3D& x) {
        return Vector3D(
            -5.0 * x.x + 4.0 * x.y + 2.0 * x.z,
            2.0 * x.x - 7.0 * x.y,
            x.y + 3.0 * x.z);
    });

    auto grad = gradient3(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3, 4);
    EXPECT_DOUBLE_EQ(-5.0, grad[0].x);
    EXPECT_DOUBLE_EQ(4.0, grad[0].y);
    EXPECT_DOUBLE_EQ(2.0, grad[0].z);
    EXPECT_DOUBLE_EQ(2.0, grad[1].x);
    EXPECT_DOUBLE_EQ(-7.0, grad[1].y);
    EXPECT_DOUBLE_EQ(0.0, grad[1].z);
    EXPECT_DOUBLE_EQ(0.0, grad[2].x);
    EXPECT_DOUBLE_EQ(1.0, grad[2].y);
    EXPECT_DOUBLE_EQ(3.0, grad[2].z);
}

TEST(FdmUtils, ScalarToLaplacian2) {
    CellCenteredScalarGrid2 grid(10, 10, 2.0, 3.0, -1.0, 4.0);
    grid.fill([&](const Vector2D& x) {
        return -5.0 * x.x * x.x + 4.0 * x.y * x.y;
    });

    double lapl = laplacian2(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3);
    EXPECT_DOUBLE_EQ(-2.0, lapl);
}

TEST(FdmUtils, VectorToLaplacian2) {
    CellCenteredVectorGrid2 grid(10, 10, 2.0, 3.0, -1.0, 4.0);
    grid.fill([&](const Vector2D& x) {
        return Vector2D(
            -5.0 * x.x * x.x + 4.0 * x.y * x.y,
            2.0 * x.x * x.x - 7.0 * x.y * x.y);
    });

    auto lapl = laplacian2(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3);
    EXPECT_DOUBLE_EQ(-2.0, lapl.x);
    EXPECT_DOUBLE_EQ(-10.0, lapl.y);
}

TEST(FdmUtils, ScalarToLaplacian3) {
    CellCenteredScalarGrid3 grid(10, 10, 10, 2.0, 3.0, 0.5, -1.0, 4.0, 2.0);
    grid.fill([&](const Vector3D& x) {
        return -5.0 * x.x * x.x + 4.0 * x.y * x.y - 3.0 * x.z * x.z;
    });

    double lapl = laplacian3(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3, 4);
    EXPECT_DOUBLE_EQ(-8.0, lapl);
}

TEST(FdmUtils, VectorToLaplacian3) {
    CellCenteredVectorGrid3 grid(10, 10, 10, 2.0, 3.0, 0.5, -1.0, 4.0, 2.0);
    grid.fill([&](const Vector3D& x) {
        return Vector3D(
            -5.0 * x.x * x.x + 4.0 * x.y * x.y + 2.0 * x.z * x.z,
            2.0 * x.x * x.x - 7.0 * x.y * x.y,
            x.y * x.y + 3.0 * x.z * x.z);
    });

    auto lapl = laplacian3(
        grid.constDataAccessor(), grid.gridSpacing(), 5, 3, 4);
    EXPECT_DOUBLE_EQ(2.0, lapl.x);
    EXPECT_DOUBLE_EQ(-10.0, lapl.y);
    EXPECT_DOUBLE_EQ(8.0, lapl.z);
}
