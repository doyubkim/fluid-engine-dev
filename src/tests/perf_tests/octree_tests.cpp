// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <perf_tests.h>

#include <jet/octree.h>
#include <jet/timer.h>
#include <jet/triangle_mesh3.h>

#include <gtest/gtest.h>

#include <random>

using namespace jet;

TEST(Octree, Nearest) {
    TriangleMesh3 triMesh;

    std::ifstream file(RESOURCES_DIR "bunny.obj");
    ASSERT_TRUE(file);

    if (file) {
        triMesh.readObj(&file);
        file.close();
    }

    std::vector<Triangle3> triangles;
    BoundingBox3D bound;
    for (size_t i = 0; i < triMesh.numberOfTriangles(); ++i) {
        auto tri = triMesh.triangle(i);
        triangles.push_back(tri);
        bound.merge(tri.boundingBox());
    }

    const auto triBoxTestFunc = [](const Triangle3& tri, const BoundingBox3D& box) {
        // TODO: Implement actual intersecting test
        return tri.boundingBox().overlaps(box);
    };

    Octree<Triangle3> octree;
    octree.build(triangles, bound, triBoxTestFunc, 4);

    std::mt19937 rng(0);
    std::uniform_real_distribution<> d(0.0, 1.0);
    const auto makeVec = [&]() { return Vector3D(d(rng), d(rng), d(rng)); };

    const auto distanceFunc = [](const Triangle3& tri, const Vector3D& pt) {
        return tri.closestDistance(pt);
    };

    std::vector<NearestNeighborQueryResult3<Triangle3>> results;

    Timer timer;
    size_t n = 1000;
    for (size_t i = 0; i < n; ++i) {
        results.push_back(octree.nearest(makeVec(), distanceFunc));
    }

    JET_PRINT_INFO("Octree::nearest calls took %f sec.\n",
                   timer.durationInSeconds());

    EXPECT_EQ(n, results.size());
}

TEST(Octree, RayIntersects) {
    TriangleMesh3 triMesh;

    std::ifstream file(RESOURCES_DIR "bunny.obj");
    ASSERT_TRUE(file);

    if (file) {
        triMesh.readObj(&file);
        file.close();
    }

    std::vector<Triangle3> triangles;
    BoundingBox3D bound;
    for (size_t i = 0; i < triMesh.numberOfTriangles(); ++i) {
        auto tri = triMesh.triangle(i);
        triangles.push_back(tri);
        bound.merge(tri.boundingBox());
    }

    const auto triBoxTestFunc = [](const Triangle3& tri, const BoundingBox3D& box) {
        // TODO: Implement actual intersecting test
        return tri.boundingBox().overlaps(box);
    };

    Octree<Triangle3> octree;
    octree.build(triangles, bound, triBoxTestFunc, 4);

    std::mt19937 rng(0);
    std::uniform_real_distribution<> d(0.0, 1.0);
    const auto makeVec = [&]() { return Vector3D(d(rng), d(rng), d(rng)); };

    const auto testFunc = [](const Triangle3& tri, const Ray3D& ray) {
        return tri.intersects(ray);
    };

    std::vector<bool> results;

    Timer timer;
    size_t n = 1000;
    for (size_t i = 0; i < n; ++i) {
        results.push_back(
            octree.intersects(Ray3D(makeVec(), makeVec().normalized()), testFunc));
    }

    JET_PRINT_INFO("Octree::intersects calls took %f sec.\n",
                   timer.durationInSeconds());

    EXPECT_EQ(n, results.size());
}
