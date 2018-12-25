// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/cuda_particle_system_solver3.h>
#include <jet/cuda_utils.h>

using namespace jet;

namespace {

__global__ void advanceTimeStepKernel(float m, float dt, float4 gravity,
                                      const float4* positions,
                                      const float4* velocities, size_t n,
                                      float4* newPositions,
                                      float4* newVelocities) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Time integration
        float4 v1 = velocities[i] + dt * gravity;
        float4 p1 = positions[i] + dt * v1;

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

void CudaParticleSystemSolver3::onAdvanceTimeStep(double timeStepInSeconds) {
    updateCollider(timeStepInSeconds);
    updateEmitter(timeStepInSeconds);

    auto particles = particleSystemData();
    size_t n = particles->numberOfParticles();
    auto posCurr = particles->positions();
    auto velCurr = particles->velocities();
    auto dt = static_cast<float>(timeStepInSeconds);
    auto g = toFloat4(gravity(), 0.0f);

    unsigned int numBlocks, numThreads;
    cudaComputeGridSize((unsigned int)n, 256, numBlocks, numThreads);
    advanceTimeStepKernel<<<numBlocks, numThreads>>>(
        _mass, dt, g, posCurr.data(), velCurr.data(), n, posCurr.data(),
        velCurr.data());
}
