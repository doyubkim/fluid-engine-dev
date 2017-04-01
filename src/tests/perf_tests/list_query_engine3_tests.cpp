// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <perf_tests.h>

#include <jet/list_query_engine3.h>
#include <jet/timer.h>
#include <jet/triangle_mesh3.h>

#include <gtest/gtest.h>

#include <random>

using namespace jet;

TEST(ListQueryEngine3, Nearest) {
    TriangleMesh3 triMesh;

    std::ifstream file(RESOURCES_DIR "bunny.obj");
    ASSERT_TRUE(file);

    if (file) {
        triMesh.readObj(&file);
        file.close();
    }

    ListQueryEngine3<Triangle3> listQueryEngine;
    for (size_t i = 0; i < triMesh.numberOfTriangles(); ++i) {
        auto tri = triMesh.triangle(i);
        listQueryEngine.add(tri);
    }

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
        results.push_back(listQueryEngine.nearest(makeVec(), distanceFunc));
    }

    JET_PRINT_INFO("ListQueryEngine3::nearest calls took %f sec.\n",
                   timer.durationInSeconds());

    EXPECT_EQ(n, results.size());
}

TEST(ListQueryEngine3, RayIntersects) {
    TriangleMesh3 triMesh;

    std::ifstream file(RESOURCES_DIR "bunny.obj");
    ASSERT_TRUE(file);

    if (file) {
        triMesh.readObj(&file);
        file.close();
    }

    ListQueryEngine3<Triangle3> listQueryEngine;
    for (size_t i = 0; i < triMesh.numberOfTriangles(); ++i) {
        auto tri = triMesh.triangle(i);
        listQueryEngine.add(tri);
    }

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
        results.push_back(listQueryEngine.intersects(
            Ray3D(makeVec(), makeVec().normalized()), testFunc));
    }

    JET_PRINT_INFO("ListQueryEngine3::intersects calls took %f sec.\n",
                   timer.durationInSeconds());

    EXPECT_EQ(n, results.size());
}
