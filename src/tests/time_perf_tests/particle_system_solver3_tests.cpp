// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/constants.h>
#include <jet/particle_system_solver3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>

#include <benchmark/benchmark.h>

#include <thrust/random.h>

#include <random>

class ParticleSystemSolver3 : public benchmark::Fixture {
 public:
    std::mt19937 rng{0};
    std::uniform_real_distribution<> dist{0.0, 1.0};
    jet::Array1<jet::Vector3D> points;
    jet::ParticleSystemSolver3 solver;
    jet::Frame frame{0, 1.0 / 300.0};

    void SetUp(benchmark::State& state) override {
        auto plane = std::make_shared<jet::Plane3>(jet::Vector3D(0, 1, 0),
                                                   jet::Vector3D());
        auto collider = std::make_shared<jet::RigidBodyCollider3>(plane);

        solver.setCollider(collider);
        solver.setDragCoefficient(0.0);
        solver.setRestitutionCoefficient(1.0);

        size_t numParticles = static_cast<size_t>(state.range(0));
        auto& particles = solver.particleSystemData();

        points.clear();
        for (size_t i = 0; i < numParticles; ++i) {
            points.append(makeVec());
        }
        particles->resize(0);
        particles->addParticles(points.constAccessor());
    }

    void SetUp(const benchmark::State&) override {}

    void TearDown(benchmark::State&) override {}

    void TearDown(const benchmark::State&) override {}

    void update() {
        solver.update(frame);
        frame.advance();
    }

    jet::Vector3D makeVec() {
        return jet::Vector3D(dist(rng), dist(rng), dist(rng));
    }
};

BENCHMARK_DEFINE_F(ParticleSystemSolver3, Update)
(benchmark::State& state) {
    using namespace std::chrono;

    while (state.KeepRunning()) {
        auto start = high_resolution_clock::now();
        update();
        auto end = high_resolution_clock::now();

        auto elapsed_seconds = duration_cast<duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK_REGISTER_F(ParticleSystemSolver3, Update)
    ->Arg(1 << 18)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);
