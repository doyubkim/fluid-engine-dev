// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array2.h>
#include <jet/marching_cubes.h>
#include <jet/triangle_mesh_to_sdf.h>
#include <jet/vertex_centered_scalar_grid3.h>

using namespace jet;

JET_TESTS(TriangleMeshToSdf);

JET_BEGIN_TEST_F(TriangleMeshToSdf, Cube) {
    TriangleMesh3 triMesh;

    triMesh.addPoint({0, 0, 0});
    triMesh.addPoint({0, 0, 1});
    triMesh.addPoint({0, 1, 0});
    triMesh.addPoint({0, 1, 1});
    triMesh.addPoint({1, 0, 0});
    triMesh.addPoint({1, 0, 1});
    triMesh.addPoint({1, 1, 0});
    triMesh.addPoint({1, 1, 1});

    // -x
    triMesh.addPointTriangle({0, 1, 3});
    triMesh.addPointTriangle({0, 3, 2});

    // +x
    triMesh.addPointTriangle({4, 6, 7});
    triMesh.addPointTriangle({4, 7, 5});

    // -y
    triMesh.addPointTriangle({0, 4, 5});
    triMesh.addPointTriangle({0, 5, 1});

    // +y
    triMesh.addPointTriangle({2, 3, 7});
    triMesh.addPointTriangle({2, 7, 6});

    // -z
    triMesh.addPointTriangle({0, 2, 6});
    triMesh.addPointTriangle({0, 6, 4});

    // +z
    triMesh.addPointTriangle({1, 5, 7});
    triMesh.addPointTriangle({1, 7, 3});

    VertexCenteredScalarGrid3 grid(
        64, 64, 64, 3.0/64, 3.0/64, 3.0/64, -1.0, -1.0, -1.0);

    triangleMeshToSdf(triMesh, &grid);

    Array2<double> temp(64, 64);
    for (size_t j = 0; j < 64; ++j) {
        for (size_t i = 0; i < 64; ++i) {
            temp(i, j) = grid(i, j, 32);
        }
    }

    saveData(temp.constAccessor(), "sdf_#grid2.npy");

    TriangleMesh3 triMesh2;
    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh2,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh2, "cube.obj");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(TriangleMeshToSdf, Bunny) {
    TriangleMesh3 triMesh;

    std::ifstream file(RESOURCES_DIR "/bunny.obj");
    if (file) {
        triMesh.readObj(&file);
        file.close();
    }

    BoundingBox3D box = triMesh.boundingBox();
    Vector3D scale(box.width(), box.height(), box.depth());
    box.lowerCorner -= 0.2 * scale;
    box.upperCorner += 0.2 * scale;

    VertexCenteredScalarGrid3 grid(
        100, 100, 100,
        box.width() / 100, box.height() / 100, box.depth() / 100,
        box.lowerCorner.x, box.lowerCorner.y, box.lowerCorner.z);

    triangleMeshToSdf(triMesh, &grid);

    TriangleMesh3 triMesh2;
    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh2,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh2, "bunny.obj");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(TriangleMeshToSdf, Dragon) {
    TriangleMesh3 triMesh;

    std::ifstream file(RESOURCES_DIR "/dragon.obj");
    if (file) {
        triMesh.readObj(&file);
        file.close();
    }

    BoundingBox3D box = triMesh.boundingBox();
    Vector3D scale(box.width(), box.height(), box.depth());
    box.lowerCorner -= 0.2 * scale;
    box.upperCorner += 0.2 * scale;

    VertexCenteredScalarGrid3 grid(
        100, 100, 100,
        box.width() / 100, box.height() / 100, box.depth() / 100,
        box.lowerCorner.x, box.lowerCorner.y, box.lowerCorner.z);

    triangleMeshToSdf(triMesh, &grid);

    TriangleMesh3 triMesh2;
    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh2,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh2, "dragon.obj");
}
JET_END_TEST_F
