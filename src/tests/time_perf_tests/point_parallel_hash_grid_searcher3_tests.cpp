// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/logging.h>
#include <jet/point_parallel_hash_grid_searcher3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::Array1;
using jet::Vector3D;

class PointParallelHashGridSearcher3 : public ::benchmark::Fixture {
 protected:
    Array1<Vector3D> points;

    void SetUp(const ::benchmark::State& state) {
        jet::Logging::mute();

        int N = state.range(0);

        std::mt19937 rng;
        std::uniform_real_distribution<> d(0.0, 1.0);

        points.clear();
        for (int i = 0; i < N; ++i) {
            points.append(Vector3D(d(rng), d(rng), d(rng)));
        }
    }
};

BENCHMARK_DEFINE_F(PointParallelHashGridSearcher3, Build)
(benchmark::State& state) {
    while (state.KeepRunning()) {
        jet::PointParallelHashGridSearcher3 grid(64, 64, 64, 1.0 / 64.0);
        grid.build(points);
    }
}

BENCHMARK_REGISTER_F(PointParallelHashGridSearcher3, Build)
    ->Arg(1 << 5)
    ->Arg(1 << 10)
    ->Arg(1 << 20);
