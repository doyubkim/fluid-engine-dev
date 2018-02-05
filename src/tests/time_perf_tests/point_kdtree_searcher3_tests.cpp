// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/point_kdtree_searcher3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::Array1;
using jet::Vector3D;

class PointKdTreeSearcher3 : public ::benchmark::Fixture {
 protected:
    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist{0.0, 1.0};
    Array1<Vector3D> points;

    void SetUp(const ::benchmark::State& state) {
        int N = state.range(0);

        points.clear();
        for (int i = 0; i < N; ++i) {
            points.append(makeVec());
        }
    }

    Vector3D makeVec() { return Vector3D(dist(rng), dist(rng), dist(rng)); }
};

BENCHMARK_DEFINE_F(PointKdTreeSearcher3, Build)(benchmark::State& state) {
    while (state.KeepRunning()) {
        jet::PointKdTreeSearcher3 tree;
        tree.build(points);
    }
}

BENCHMARK_REGISTER_F(PointKdTreeSearcher3, Build)
    ->Arg(1 << 5)
    ->Arg(1 << 10)
    ->Arg(1 << 20);

BENCHMARK_DEFINE_F(PointKdTreeSearcher3, ForEachNearbyPoints)
(benchmark::State& state) {
    jet::PointKdTreeSearcher3 tree;
    tree.build(points);

    size_t cnt = 0;
    while (state.KeepRunning()) {
        tree.forEachNearbyPoint(makeVec(), 1.0 / 64.0,
                                [&](size_t, const Vector3D&) { ++cnt; });
    }
}

BENCHMARK_REGISTER_F(PointKdTreeSearcher3, ForEachNearbyPoints)
    ->Arg(1 << 5)
    ->Arg(1 << 10)
    ->Arg(1 << 20);
