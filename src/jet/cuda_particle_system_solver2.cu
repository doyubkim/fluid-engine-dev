// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/cuda_particle_system_solver2.h>
#include <jet/cuda_utils.h>

using namespace jet;

namespace {

__global__ void advanceTimeStepKernel(float m, float dt, float2 gravity,
                                      const float2* positions,
                                      const float2* velocities, size_t n,
                                      float2* newPositions,
                                      float2* newVelocities) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Time integration
        float2 v1 = velocities[i] + dt * gravity;
        float2 p1 = positions[i] + dt * v1;

        // Collision handling
        // TODO: Use collider
        if (p1.y < 0.0f) {
            p1.y = 0.0f;
            if (v1.y < 0.0f) {
                v1.y *= -1.0;
            }
        }

        newPositions[i] = p1;
        newVelocities[i] = v1;
    }
}

}  // namespace

void CudaParticleSystemSolver2::onAdvanceTimeStep(double timeStepInSeconds) {
    updateCollider(timeStepInSeconds);
    updateEmitter(timeStepInSeconds);

    auto particles = particleSystemData();
    size_t n = particles->numberOfParticles();
    auto posCurr = particles->positions();
    auto velCurr = particles->velocities();
    auto dt = static_cast<float>(timeStepInSeconds);
    auto g = toFloat2(gravity());

    unsigned int numBlocks, numThreads;
    cudaComputeGridSize((unsigned int)n, 256, numBlocks, numThreads);
    advanceTimeStepKernel<<<numBlocks, numThreads>>>(
        _mass, dt, g, posCurr.data(), velCurr.data(), n, posCurr.data(),
        velCurr.data());
}
