// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/octree.h>
#include <jet/triangle_mesh3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::TriangleMesh3;
using jet::Triangle3;
using jet::BoundingBox3D;
using jet::Vector3D;
using jet::Ray3D;

class Octree : public ::benchmark::Fixture {
 public:
    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist{0.0, 1.0};
    TriangleMesh3 triMesh;
    jet::Octree<Triangle3> queryEngine;

    void SetUp(const ::benchmark::State&) {
        std::ifstream file(RESOURCES_DIR "/bunny.obj");

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

        const auto triBoxTestFunc = [](const Triangle3& tri,
                                       const BoundingBox3D& box) {
            // TODO: Implement actual intersecting test
            return tri.boundingBox().overlaps(box);
        };

        queryEngine.build(triangles, bound, triBoxTestFunc, 6);
    }

    Vector3D makeVec() { return Vector3D(dist(rng), dist(rng), dist(rng)); }

    static double distanceFunc(const Triangle3& tri, const Vector3D& pt) {
        return tri.closestDistance(pt);
    }

    static bool intersectsFunc(const Triangle3& tri, const Ray3D& ray) {
        return tri.intersects(ray);
    }
};

BENCHMARK_DEFINE_F(Octree, Nearest)(benchmark::State& state) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(queryEngine.nearest(makeVec(), distanceFunc));
    }
}

BENCHMARK_REGISTER_F(Octree, Nearest);

BENCHMARK_DEFINE_F(Octree, RayIntersects)(benchmark::State& state) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(queryEngine.intersects(
            Ray3D(makeVec(), makeVec().normalized()), intersectsFunc));
    }
}

BENCHMARK_REGISTER_F(Octree, RayIntersects);
