// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/constants.h>
#include <jet/cuda_particle_system_solver3.h>
#include <jet/timer.h>

#include <benchmark/benchmark.h>

#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/transform.h>

namespace {

struct Rng {
    template <typename Index>
    __device__ float4 operator()(Index idx) {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist(0.0f, 1.0f);

        float4 result;
        randEng.discard(3 * idx);
        result.x = uniDist(randEng);
        randEng.discard(3 * idx + 1);
        result.y = uniDist(randEng);
        randEng.discard(3 * idx + 2);
        result.z = uniDist(randEng);
        result.w = 0.0f;

        return result;
    }
};

}  // namespace

class CudaParticleSystemSolver3 : public benchmark::Fixture {
 public:
    jet::CudaParticleSystemSolver3 solver;
    jet::Frame frame{0, 1.0 / 300.0};

    void SetUp(benchmark::State& state) override {
        solver.setDragCoefficient(0.0);
        solver.setRestitutionCoefficient(1.0);

        size_t numParticles = static_cast<size_t>(state.range(0));
        auto particles = solver.particleSystemData();

        jet::CudaArray1<float4> pos(numParticles);
        thrust::transform(thrust::make_counting_iterator(jet::kZeroSize),
            thrust::make_counting_iterator(numParticles),
            thrust::device_ptr<float4>(pos.data()),
            Rng());
        particles->addParticles(jet::CudaArrayView1<float4>(pos));
    }

    void SetUp(const benchmark::State&) override {}

    void TearDown(benchmark::State&) override {
        solver = jet::CudaParticleSystemSolver3();
    }

    void TearDown(const benchmark::State&) override {}

    void update() {
        solver.update(frame);
        frame.advance();
        cudaDeviceSynchronize();
    }
};

BENCHMARK_DEFINE_F(CudaParticleSystemSolver3, Update)
(benchmark::State& state) {
    using namespace std::chrono;

    while (state.KeepRunning()) {
        jet::Timer timer;

        update();

        const double elapsedSeconds = timer.durationInSeconds();

        state.SetIterationTime(elapsedSeconds);
    }
}
BENCHMARK_REGISTER_F(CudaParticleSystemSolver3, Update)
    ->Arg(1 << 18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
