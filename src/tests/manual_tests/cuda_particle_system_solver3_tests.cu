// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/cuda_particle_system_solver3.h>

using namespace jet;
using namespace experimental;

JET_TESTS(CudaParticleSystemSolver3);

JET_BEGIN_TEST_F(CudaParticleSystemSolver3, PerfectBounce) {
    CudaParticleSystemSolver3 solver;
    solver.setDragCoefficient(0.0);
    solver.setRestitutionCoefficient(1.0);

    auto& particles = solver.particleSystemData();
    particles->addParticle({0.0f, 3.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f});

    Array1<float> x(1000);
    Array1<float> y(1000);
    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(x.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(y.constAccessor(), 0, filename);

    Frame frame;
    frame.timeIntervalInSeconds = 1.0 / 300.0;
    for (; frame.index < 1000; frame.advance()) {
        solver.update(frame);

        float4 pos = particles->positions()[0];
        printf("%f, %f\n", pos.x, pos.y);

        x[frame.index] = pos.x;
        y[frame.index] = pos.y;
        snprintf(filename, sizeof(filename), "data.#line2,%04d,x.npy",
                 frame.index);
        saveData(x.constAccessor(), frame.index, filename);
        snprintf(filename, sizeof(filename), "data.#line2,%04d,y.npy",
                 frame.index);
        saveData(y.constAccessor(), frame.index, filename);
    }
}
JET_END_TEST_F
