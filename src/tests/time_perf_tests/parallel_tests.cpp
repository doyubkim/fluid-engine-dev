// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/parallel.h>

#include <benchmark/benchmark.h>

#include <random>

class Parallel : public ::benchmark::Fixture {
 public:
    std::vector<double> a, b, c;
    size_t n = 0;
    unsigned int numThreads = 1;

    std::mt19937 rng{0};
    std::uniform_real_distribution<> d{0.0, 1.0};

    void SetUp(const ::benchmark::State& state) {
        n = static_cast<size_t>(state.range(0));
        numThreads = static_cast<unsigned int>(state.range(1));

        a.resize(n);
        b.resize(n);
        c.resize(n);
    }
};

BENCHMARK_DEFINE_F(Parallel, ParallelFor)(benchmark::State& state) {
    unsigned int oldNumThreads = jet::maxNumberOfThreads();
    jet::setMaxNumberOfThreads(numThreads);

    while (state.KeepRunning()) {
        jet::parallelFor(jet::kZeroSize, n, [this](size_t i) {
            c[i] = 1.0 / std::sqrt(a[i] / b[i] + 1.0);
        });
    }

    jet::setMaxNumberOfThreads(oldNumThreads);
}

BENCHMARK_REGISTER_F(Parallel, ParallelFor)
    ->UseRealTime()
    ->Args({1 << 8, 1})
    ->Args({1 << 8, 2})
    ->Args({1 << 8, 4})
    ->Args({1 << 8, 8})
    ->Args({1 << 16, 1})
    ->Args({1 << 16, 2})
    ->Args({1 << 16, 4})
    ->Args({1 << 16, 8})
    ->Args({1 << 24, 1})
    ->Args({1 << 24, 2})
    ->Args({1 << 24, 4})
    ->Args({1 << 24, 8});

BENCHMARK_DEFINE_F(Parallel, ParallelRangeFor)(benchmark::State& state) {
    unsigned int oldNumThreads = jet::maxNumberOfThreads();
    jet::setMaxNumberOfThreads(numThreads);

    while (state.KeepRunning()) {
        jet::parallelRangeFor(jet::kZeroSize, n,
                              [this](size_t iBegin, size_t iEnd) {
                                  for (size_t i = iBegin; i < iEnd; ++i) {
                                      c[i] = 1.0 / std::sqrt(a[i] / b[i] + 1.0);
                                  }
                              });
    }

    jet::setMaxNumberOfThreads(oldNumThreads);
}

BENCHMARK_REGISTER_F(Parallel, ParallelRangeFor)
    ->UseRealTime()
    ->Args({1 << 8, 1})
    ->Args({1 << 8, 2})
    ->Args({1 << 8, 4})
    ->Args({1 << 8, 8})
    ->Args({1 << 16, 1})
    ->Args({1 << 16, 2})
    ->Args({1 << 16, 4})
    ->Args({1 << 16, 8})
    ->Args({1 << 24, 1})
    ->Args({1 << 24, 2})
    ->Args({1 << 24, 4})
    ->Args({1 << 24, 8});
