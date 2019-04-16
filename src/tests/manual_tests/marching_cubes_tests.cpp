// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array3.h>
#include <jet/marching_cubes.h>
#include <jet/vertex_centered_scalar_grid3.h>

using namespace jet;

JET_TESTS(MarchingCubes);

JET_BEGIN_TEST_F(MarchingCubes, SingleCube) {
    TriangleMesh3 triMesh;

    Array3<double> grid(2, 2, 2);
    grid(0, 0, 0) = -0.5;
    grid(0, 0, 1) = -0.5;
    grid(0, 1, 0) =  0.5;
    grid(0, 1, 1) =  0.5;
    grid(1, 0, 0) = -0.5;
    grid(1, 0, 1) = -0.5;
    grid(1, 1, 0) =  0.5;
    grid(1, 1, 1) =  0.5;

    marchingCubes(
        grid,
        Vector3D(1, 1, 1),
        Vector3D(),
        &triMesh,
        0,
        kDirectionAll,
        kDirectionAll);

    saveTriangleMeshData(triMesh, "single_cube.obj");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(MarchingCubes, FourCubes) {
    TriangleMesh3 triMesh;

    VertexCenteredScalarGrid3 grid(2, 1, 2);
    grid.fill([](const Vector3D& x) {
        return x.y - 0.5;
    });

    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh, "four_cubes.obj");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(MarchingCubes, Sphere) {
    TriangleMesh3 triMesh;

    VertexCenteredScalarGrid3 grid(16, 16, 16);
    grid.fill([](const Vector3D& x) {
        return x.distanceTo({8.0, 8.0, 8.0}) - 3.0;
    });

    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh, "sphere.obj");

    grid.fill([](const Vector3D& x) {
        return x.distanceTo({0.0, 4.0, 3.0}) - 6.0;
    });

    triMesh.clear();

    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh, "clamped_sphere.obj");

    grid.fill([](const Vector3D& x) {
        return x.distanceTo({11.0, 14.0, 12.0}) - 6.0;
    });

    triMesh.clear();

    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh, "clamped_sphere2.obj");
}
JET_END_TEST_F
