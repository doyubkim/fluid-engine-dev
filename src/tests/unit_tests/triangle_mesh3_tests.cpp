// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <unit_tests_utils.h>

#include <jet/triangle_mesh3.h>

using namespace jet;

TEST(TriangleMesh3, Constructors) {
    TriangleMesh3 mesh1;
    EXPECT_EQ(0u, mesh1.numberOfPoints());
    EXPECT_EQ(0u, mesh1.numberOfNormals());
    EXPECT_EQ(0u, mesh1.numberOfUvs());
    EXPECT_EQ(0u, mesh1.numberOfTriangles());
}

TEST(TriangleMesh3, ReadObj) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    EXPECT_EQ(56u, mesh.numberOfPoints());
    EXPECT_EQ(96u, mesh.numberOfNormals());
    EXPECT_EQ(76u, mesh.numberOfUvs());
    EXPECT_EQ(108u, mesh.numberOfTriangles());
}

TEST(TriangleMesh3, ClosestPoint) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    const auto bruteForceSearch = [&](const Vector3D& pt) {
        double minDist2 = kMaxD;
        Vector3D result;
        for (size_t i = 0; i < mesh.numberOfTriangles(); ++i) {
            Triangle3 tri = mesh.triangle(i);
            auto localResult = tri.closestPoint(pt);
            double localDist2 = pt.distanceSquaredTo(localResult);
            if (localDist2 < minDist2) {
                minDist2 = localDist2;
                result = localResult;
            }
        }
        return result;
    };

    size_t numSamples = getNumberOfSamplePoints3();
    for (size_t i = 0; i < numSamples; ++i) {
        auto actual = mesh.closestPoint(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_VECTOR3_EQ(expected, actual);
    }
}

TEST(TriangleMesh3, ClosestNormal) {
    std::string objStr = getSphereTriMesh5x5Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    const auto bruteForceSearch = [&](const Vector3D& pt) {
        double minDist2 = kMaxD;
        Vector3D result;
        for (size_t i = 0; i < mesh.numberOfTriangles(); ++i) {
            Triangle3 tri = mesh.triangle(i);
            auto localResult = tri.closestNormal(pt);
            auto closestPt = tri.closestPoint(pt);
            double localDist2 = pt.distanceSquaredTo(closestPt);
            if (localDist2 < minDist2) {
                minDist2 = localDist2;
                result = localResult;
            }
        }
        return result;
    };

    size_t numSamples = getNumberOfSamplePoints3();
    for (size_t i = 0; i < numSamples; ++i) {
        auto actual = mesh.closestNormal(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_VECTOR3_NEAR(expected, actual, 1e-9);
    }
}

TEST(TriangleMesh3, ClosestDistance) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    const auto bruteForceSearch = [&](const Vector3D& pt) {
        double minDist = kMaxD;
        for (size_t i = 0; i < mesh.numberOfTriangles(); ++i) {
            Triangle3 tri = mesh.triangle(i);
            auto localResult = tri.closestDistance(pt);
            if (localResult < minDist) {
                minDist = localResult;
            }
        }
        return minDist;
    };

    size_t numSamples = getNumberOfSamplePoints3();
    for (size_t i = 0; i < numSamples; ++i) {
        auto actual = mesh.closestDistance(getSamplePoints3()[i]);
        auto expected = bruteForceSearch(getSamplePoints3()[i]);
        EXPECT_DOUBLE_EQ(expected, actual);
    }
}


TEST(TriangleMesh3, Intersects) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    size_t numSamples = getNumberOfSamplePoints3();

    const auto bruteForceTest = [&](const Ray3D& ray) {
        for (size_t i = 0; i < mesh.numberOfTriangles(); ++i) {
            Triangle3 tri = mesh.triangle(i);
            if (tri.intersects(ray)) {
                return true;
            }
        }
        return false;
    };

    for (size_t i = 0; i < numSamples; ++i) {
        Ray3D ray(getSamplePoints3()[i], getSampleDirs3()[i]);
        bool actual = mesh.intersects(ray);
        bool expected = bruteForceTest(ray);
        EXPECT_EQ(expected, actual);
    }
}

TEST(TriangleMesh3, ClosestIntersection) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    size_t numSamples = getNumberOfSamplePoints3();

    const auto bruteForceTest = [&](const Ray3D& ray) {
        SurfaceRayIntersection3 result{};
        for (size_t i = 0; i < mesh.numberOfTriangles(); ++i) {
            Triangle3 tri = mesh.triangle(i);
            auto localResult = tri.closestIntersection(ray);
            if (localResult.distance < result.distance) {
                result = localResult;
            }
        }
        return result;
    };

    for (size_t i = 0; i < numSamples; ++i) {
        Ray3D ray(getSamplePoints3()[i], getSampleDirs3()[i]);
        auto actual = mesh.closestIntersection(ray);
        auto expected = bruteForceTest(ray);
        EXPECT_DOUBLE_EQ(expected.distance, actual.distance);
        EXPECT_VECTOR3_EQ(expected.point, actual.point);
        EXPECT_VECTOR3_EQ(expected.normal, actual.normal);
        EXPECT_EQ(expected.isIntersecting, actual.isIntersecting);
    }
}

TEST(TriangleMesh3, IsInside) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    size_t numSamples = getNumberOfSamplePoints3();

    for (size_t i = 0; i < numSamples; ++i) {
        Vector3D p = getSamplePoints3()[i];
        auto actual = mesh.isInside(p);
        auto expected = mesh.boundingBox().contains(p);
        EXPECT_EQ(expected, actual);
    }
}

TEST(TriangleMesh3, BoundingBox) {
    std::string objStr = getCubeTriMesh3x3x3Obj();
    std::istringstream objStream(objStr);

    TriangleMesh3 mesh;
    mesh.readObj(&objStream);

    EXPECT_BOUNDING_BOX3_EQ(
        BoundingBox3D({-0.5, -0.5, -0.5}, {0.5, 0.5, 0.5}),
        mesh.boundingBox());
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
