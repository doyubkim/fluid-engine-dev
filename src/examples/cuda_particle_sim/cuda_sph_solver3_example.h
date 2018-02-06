// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_CUDA_SPH_SOLVER3_EXAMPLE_H_
#define SRC_EXAMPLES_CUDA_SPH_SOLVER3_EXAMPLE_H_

#include "particle_sim_example.h"

#include <jet/cuda_sph_solver3.h>

class CudaSphSolver3Example final : public ParticleSimExample {
 public:
    CudaSphSolver3Example(const jet::Frame& frame);

 private:
    jet::experimental::CudaSphSolver3Ptr _solver;
    jet::viz::PointsRenderable3Ptr _renderable;

    void onSetup(jet::viz::GlfwWindow* window) override;

    void onUpdate(const jet::Frame& frame) override;
};

#endif  // SRC_EXAMPLES_CUDA_SPH_SOLVER3_EXAMPLE_H_
