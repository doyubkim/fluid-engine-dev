// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_linear_system2.h>
#include <jet/fdm_linear_system3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::FdmMatrix2;
using jet::FdmVector2;
using jet::FdmMatrix3;
using jet::FdmVector3;

class FdmBlas2 : public ::benchmark::Fixture {
 public:
    FdmMatrix2 m;
    FdmVector2 a;
    FdmVector2 b;

    void SetUp(const ::benchmark::State& state) {
        const auto dim = static_cast<size_t>(state.range(0));

        m.resize(dim, dim);
        a.resize(dim, dim);
        b.resize(dim, dim);

        std::mt19937 rng;
        std::uniform_real_distribution<> d(0.0, 1.0);

        m.forEachIndex([&](size_t i, size_t j) {
            m(i, j).center = d(rng);
            m(i, j).right = d(rng);
            m(i, j).up = d(rng);
            a(i, j) = d(rng);
        });
    }
};

class FdmBlas3 : public ::benchmark::Fixture {
 public:
    FdmMatrix3 m;
    FdmVector3 a;
    FdmVector3 b;

    void SetUp(const ::benchmark::State& state) {
        const auto dim = static_cast<size_t>(state.range(0));

        m.resize(dim, dim, dim);
        a.resize(dim, dim, dim);
        b.resize(dim, dim, dim);

        std::mt19937 rng;
        std::uniform_real_distribution<> d(0.0, 1.0);

        m.forEachIndex([&](size_t i, size_t j, size_t k) {
            m(i, j, k).center = d(rng);
            m(i, j, k).right = d(rng);
            m(i, j, k).up = d(rng);
            m(i, j, k).front = d(rng);
            a(i, j, k) = d(rng);
        });
    }
};

BENCHMARK_DEFINE_F(FdmBlas2, Mvm)(benchmark::State& state) {
    while (state.KeepRunning()) {
        jet::FdmBlas2::mvm(m, a, &b);
    }
}

BENCHMARK_REGISTER_F(FdmBlas2, Mvm)->Arg(1 << 6)->Arg(1 << 8)->Arg(1 << 10);

BENCHMARK_DEFINE_F(FdmBlas3, Mvm)(benchmark::State& state) {
    while (state.KeepRunning()) {
        jet::FdmBlas3::mvm(m, a, &b);
    }
}

BENCHMARK_REGISTER_F(FdmBlas3, Mvm)->Arg(1 << 4)->Arg(1 << 6)->Arg(1 << 8);
