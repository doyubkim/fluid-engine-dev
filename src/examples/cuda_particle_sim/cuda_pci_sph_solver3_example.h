// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_CUDA_PCI_SPH_SOLVER3_EXAMPLE_H_
#define SRC_EXAMPLES_CUDA_PCI_SPH_SOLVER3_EXAMPLE_H_

#include "particle_sim_example.h"

#include <jet/cuda_pci_sph_solver3.h>

class CudaPciSphSolver3Example final : public ParticleSimExample {
 public:
    CudaPciSphSolver3Example(const jet::Frame& frame);

 private:
    jet::experimental::CudaPciSphSolver3Ptr _solver;
    jet::viz::PointsRenderable3Ptr _renderable;

    void onSetup(jet::viz::GlfwWindow* window) override;

    void onUpdate(const jet::Frame& frame) override;
};

#endif  // SRC_EXAMPLES_CUDA_PCI_SPH_SOLVER3_EXAMPLE_H_
