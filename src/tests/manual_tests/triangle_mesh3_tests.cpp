// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/triangle_mesh3.h>

using namespace jet;

JET_TESTS(TriangleMesh3);

JET_BEGIN_TEST_F(TriangleMesh3, PointsOnlyGeometries) {
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
    triMesh.addPointFace({0, 1, 3});
    triMesh.addPointFace({0, 3, 2});

    // +x
    triMesh.addPointFace({4, 6, 7});
    triMesh.addPointFace({4, 7, 5});

    // -y
    triMesh.addPointFace({0, 4, 5});
    triMesh.addPointFace({0, 5, 1});

    // +y
    triMesh.addPointFace({2, 3, 7});
    triMesh.addPointFace({2, 7, 6});

    // -z
    triMesh.addPointFace({0, 2, 6});
    triMesh.addPointFace({0, 6, 4});

    // +z
    triMesh.addPointFace({1, 5, 7});
    triMesh.addPointFace({1, 7, 3});

    saveTriangleMeshData(triMesh, "cube.obj");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(TriangleMesh3, PointsAndNormalGeometries) {
    TriangleMesh3 triMesh;

    triMesh.addPoint({0, 0, 0});
    triMesh.addPoint({0, 0, 1});
    triMesh.addPoint({0, 1, 0});
    triMesh.addPoint({0, 1, 1});
    triMesh.addPoint({1, 0, 0});
    triMesh.addPoint({1, 0, 1});
    triMesh.addPoint({1, 1, 0});
    triMesh.addPoint({1, 1, 1});

    triMesh.addNormal({-1, 0, 0});
    triMesh.addNormal({1, 0, 0});
    triMesh.addNormal({0, -1, 0});
    triMesh.addNormal({0, 1, 0});
    triMesh.addNormal({0, 0, -1});
    triMesh.addNormal({0, 0, 1});

    // -x
    triMesh.addPointNormalFace({0, 1, 3}, {0, 0, 0});
    triMesh.addPointNormalFace({0, 3, 2}, {0, 0, 0});

    // +x
    triMesh.addPointNormalFace({4, 6, 7}, {1, 1, 1});
    triMesh.addPointNormalFace({4, 7, 5}, {1, 1, 1});

    // -y
    triMesh.addPointNormalFace({0, 4, 5}, {2, 2, 2});
    triMesh.addPointNormalFace({0, 5, 1}, {2, 2, 2});

    // +y
    triMesh.addPointNormalFace({2, 3, 7}, {3, 3, 3});
    triMesh.addPointNormalFace({2, 7, 6}, {3, 3, 3});

    // -z
    triMesh.addPointNormalFace({0, 2, 6}, {4, 4, 4});
    triMesh.addPointNormalFace({0, 6, 4}, {4, 4, 4});

    // +z
    triMesh.addPointNormalFace({1, 5, 7}, {5, 5, 5});
    triMesh.addPointNormalFace({1, 7, 3}, {5, 5, 5});

    saveTriangleMeshData(triMesh, "cube_with_normal.obj");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(TriangleMesh3, BasicIO) {
    TriangleMesh3 triMesh;

    std::ifstream file("resources/bunny.obj");
    if (file) {
        triMesh.readObj(&file);
        file.close();
    }
}
JET_END_TEST_F
