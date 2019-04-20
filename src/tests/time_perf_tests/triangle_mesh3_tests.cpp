// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/triangle_mesh3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::Vector3D;

class TriangleMesh3 : public ::benchmark::Fixture {
 protected:
    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist{0.0, 1.0};
    jet::TriangleMesh3 triMesh;

    void SetUp(const ::benchmark::State&) {
        std::ifstream file(RESOURCES_DIR "/bunny.obj");

        if (file) {
            triMesh.readObj(&file);
            file.close();
        }
    }

    Vector3D makeVec() { return Vector3D(dist(rng), dist(rng), dist(rng)); }
};

BENCHMARK_DEFINE_F(TriangleMesh3, ClosestPoint)(benchmark::State& state) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(triMesh.closestPoint(makeVec()));
    }
}

BENCHMARK_REGISTER_F(TriangleMesh3, ClosestPoint);

BENCHMARK_DEFINE_F(TriangleMesh3, IsInside)(benchmark::State& state) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(triMesh.isInside(makeVec()));
    }
}

BENCHMARK_REGISTER_F(TriangleMesh3, IsInside);
