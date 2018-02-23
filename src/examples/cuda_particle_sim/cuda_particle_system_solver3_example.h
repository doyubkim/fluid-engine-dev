// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_CUDA_PARTICLE_SYSTEM_SOLVER3_EXAMPLE_H_
#define SRC_EXAMPLES_CUDA_PARTICLE_SYSTEM_SOLVER3_EXAMPLE_H_

#include "particle_sim_example.h"

#include <jet/cuda_particle_system_solver3.h>

class CudaParticleSystemSolver3Example final : public ParticleSimExample {
 public:
    CudaParticleSystemSolver3Example(const jet::Frame& frame);

 private:
    jet::experimental::CudaParticleSystemSolver3Ptr _solver;
    jet::viz::PointsRenderable3Ptr _renderable;

    void onSetup(jet::viz::GlfwWindow* window) override;

    void onUpdate(const jet::Frame& frame) override;
};

#endif  // SRC_EXAMPLES_CUDA_PARTICLE_SYSTEM_SOLVER3_EXAMPLE_H_
