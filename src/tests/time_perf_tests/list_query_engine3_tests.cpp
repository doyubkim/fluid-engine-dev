// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/list_query_engine3.h>
#include <jet/triangle_mesh3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::TriangleMesh3;
using jet::Triangle3;
using jet::BoundingBox3D;
using jet::Vector3D;
using jet::Ray3D;

class ListQueryEngine3 : public ::benchmark::Fixture {
 public:
    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist{0.0, 1.0};
    TriangleMesh3 triMesh;
    jet::ListQueryEngine3<Triangle3> queryEngine;

    void SetUp(const ::benchmark::State&) {
        std::ifstream file(RESOURCES_DIR "/bunny.obj");

        if (file) {
            triMesh.readObj(&file);
            file.close();
        }

        for (size_t i = 0; i < triMesh.numberOfTriangles(); ++i) {
            auto tri = triMesh.triangle(i);
            queryEngine.add(tri);
        }
    }

    Vector3D makeVec() { return Vector3D(dist(rng), dist(rng), dist(rng)); }

    static double distanceFunc(const Triangle3& tri, const Vector3D& pt) {
        return tri.closestDistance(pt);
    }

    static bool intersectsFunc(const Triangle3& tri, const Ray3D& ray) {
        return tri.intersects(ray);
    }
};

BENCHMARK_DEFINE_F(ListQueryEngine3, Nearest)(benchmark::State& state) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(queryEngine.nearest(makeVec(), distanceFunc));
    }
}

BENCHMARK_REGISTER_F(ListQueryEngine3, Nearest);

BENCHMARK_DEFINE_F(ListQueryEngine3, RayIntersects)(benchmark::State& state) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(queryEngine.intersects(
            Ray3D(makeVec(), makeVec().normalized()), intersectsFunc));
    }
}

BENCHMARK_REGISTER_F(ListQueryEngine3, RayIntersects);
