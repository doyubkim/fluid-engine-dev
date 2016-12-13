// Copyright (c) 2016 Doyub Kim

#include <jet/triangle_mesh3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(TriangleMesh3, Constructors) {
    TriangleMesh3 mesh1;
    EXPECT_EQ(0u, mesh1.numberOfPoints());
    EXPECT_EQ(0u, mesh1.numberOfNormals());
    EXPECT_EQ(0u, mesh1.numberOfUvs());
    EXPECT_EQ(0u, mesh1.numberOfTriangles());
}

TEST(TriangleMesh3, Builder) {
    TriangleMesh3::PointArray points = {
        Vector3D(1, 2, 3),
        Vector3D(4, 5, 6),
        Vector3D(7, 8, 9),
        Vector3D(10, 11, 12)
    };

    TriangleMesh3::NormalArray normals = {
        Vector3D(10, 11, 12),
        Vector3D(7, 8, 9),
        Vector3D(4, 5, 6),
        Vector3D(1, 2, 3)
    };

    TriangleMesh3::UvArray uvs = {
        Vector2D(13, 14),
        Vector2D(15, 16)
    };

    TriangleMesh3::IndexArray pointIndices = {
        Point3UI(0, 1, 2),
        Point3UI(0, 1, 3)
    };

    TriangleMesh3::IndexArray normalIndices = {
        Point3UI(1, 2, 3),
        Point3UI(2, 1, 0)
    };

    TriangleMesh3::IndexArray uvIndices = {
        Point3UI(1, 0, 2),
        Point3UI(3, 1, 0)
    };

    TriangleMesh3 mesh = TriangleMesh3::builder()
        .withPoints(points)
        .withNormals(normals)
        .withUvs(uvs)
        .withPointIndices(pointIndices)
        .withNormalIndices(normalIndices)
        .withUvIndices(uvIndices)
        .build();

    EXPECT_EQ(4u, mesh.numberOfPoints());
    EXPECT_EQ(4u, mesh.numberOfNormals());
    EXPECT_EQ(2u, mesh.numberOfUvs());
    EXPECT_EQ(2u, mesh.numberOfTriangles());

    for (size_t i = 0; i < mesh.numberOfPoints(); ++i) {
        EXPECT_EQ(points[i], mesh.point(i));
    }

    for (size_t i = 0; i < mesh.numberOfNormals(); ++i) {
        EXPECT_EQ(normals[i], mesh.normal(i));
    }

    for (size_t i = 0; i < mesh.numberOfUvs(); ++i) {
        EXPECT_EQ(uvs[i], mesh.uv(i));
    }

    for (size_t i = 0; i < mesh.numberOfTriangles(); ++i) {
        EXPECT_EQ(pointIndices[i], mesh.pointIndex(i));
        EXPECT_EQ(normalIndices[i], mesh.normalIndex(i));
        EXPECT_EQ(uvIndices[i], mesh.uvIndex(i));
    }
}
