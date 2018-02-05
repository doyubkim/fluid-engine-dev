// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_linear_system2.h>
#include <jet/fdm_linear_system3.h>

#include <benchmark/benchmark.h>

#include <random>

using jet::Array3;
using jet::FdmMatrix2;
using jet::FdmVector2;
using jet::FdmMatrix3;
using jet::FdmVector3;
using jet::FdmCompressedLinearSystem3;
using jet::Size3;

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

class FdmCompressedBlas3 : public ::benchmark::Fixture {
 public:
    FdmCompressedLinearSystem3 system;

    void SetUp(const ::benchmark::State& state) {
        const auto dim = static_cast<size_t>(state.range(0));

        buildSystem(&system, {dim, dim, dim});
    }

    static void buildSystem(FdmCompressedLinearSystem3* system,
                            const Size3& size) {
        system->clear();

        Array3<size_t> coordToIndex(size);
        const auto acc = coordToIndex.constAccessor();

        coordToIndex.forEachIndex([&](size_t i, size_t j, size_t k) {
            const size_t cIdx = acc.index(i, j, k);
            const size_t lIdx = acc.index(i - 1, j, k);
            const size_t rIdx = acc.index(i + 1, j, k);
            const size_t dIdx = acc.index(i, j - 1, k);
            const size_t uIdx = acc.index(i, j + 1, k);
            const size_t bIdx = acc.index(i, j, k - 1);
            const size_t fIdx = acc.index(i, j, k + 1);

            coordToIndex[cIdx] = system->b.size();
            double bijk = 0.0;

            std::vector<double> row(1, 0.0);
            std::vector<size_t> colIdx(1, cIdx);

            if (i > 0) {
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(lIdx);
            }
            if (i < size.x - 1) {
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(rIdx);
            }

            if (j > 0) {
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(dIdx);
            } else {
                bijk += 1.0;
            }

            if (j < size.y - 1) {
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(uIdx);
            } else {
                bijk -= 1.0;
            }

            if (k > 0) {
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(bIdx);
            } else {
                bijk += 1.0;
            }

            if (k < size.z - 1) {
                row[0] += 1.0;
                row.push_back(-1.0);
                colIdx.push_back(fIdx);
            } else {
                bijk -= 1.0;
            }

            system->A.addRow(row, colIdx);
            system->b.append(bijk);
        });

        system->x.resize(system->b.size(), 0.0);
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

BENCHMARK_DEFINE_F(FdmCompressedBlas3, Mvm)(benchmark::State& state) {
    while (state.KeepRunning()) {
        jet::FdmCompressedBlas3::mvm(system.A, system.b, &system.x);
    }
}

BENCHMARK_REGISTER_F(FdmCompressedBlas3, Mvm)
    ->Arg(1 << 4)
    ->Arg(1 << 6)
    ->Arg(1 << 8);
