// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/box3.h>
#include <jet/triangle_mesh_to_sdf.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(TriangleMeshToSdf, TriangleMeshToSdf) {
    TriangleMesh3 mesh;

    // Build a cube
    mesh.addPoint({0.0, 0.0, 0.0});
    mesh.addPoint({0.0, 0.0, 1.0});
    mesh.addPoint({0.0, 1.0, 0.0});
    mesh.addPoint({0.0, 1.0, 1.0});
    mesh.addPoint({1.0, 0.0, 0.0});
    mesh.addPoint({1.0, 0.0, 1.0});
    mesh.addPoint({1.0, 1.0, 0.0});
    mesh.addPoint({1.0, 1.0, 1.0});

    mesh.addPointTriangle({0, 1, 3});
    mesh.addPointTriangle({0, 3, 2});
    mesh.addPointTriangle({4, 6, 7});
    mesh.addPointTriangle({4, 7, 5});
    mesh.addPointTriangle({0, 4, 5});
    mesh.addPointTriangle({0, 5, 1});
    mesh.addPointTriangle({2, 3, 7});
    mesh.addPointTriangle({2, 7, 6});
    mesh.addPointTriangle({0, 2, 6});
    mesh.addPointTriangle({0, 6, 4});
    mesh.addPointTriangle({1, 5, 7});
    mesh.addPointTriangle({1, 7, 3});

    CellCenteredScalarGrid3 grid(
        3, 3, 3,
        1.0, 1.0, 1.0,
        -1.0, -1.0, -1.0);

    triangleMeshToSdf(mesh, &grid, 10);

    Box3 box(Vector3D(), Vector3D(1.0, 1.0, 1.0));

    auto gridPos = grid.dataPosition();
    grid.forEachDataPointIndex(
        [&](size_t i, size_t j, size_t k) {
            auto pos = gridPos(i, j, k);
            double ans = box.closestDistance(pos);
            ans *= box.bound.contains(pos) ? -1.0 : 1.0;
            EXPECT_DOUBLE_EQ(ans, grid(i, j, k));
        });
}
