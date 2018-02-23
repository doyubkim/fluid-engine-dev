// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_PCI_SPH_SOLVER3_EXAMPLE_H_
#define SRC_EXAMPLES_PCI_SPH_SOLVER3_EXAMPLE_H_

#include "particle_sim_example.h"

#include <jet/pci_sph_solver3.h>

class PciSphSolver3Example final : public ParticleSimExample {
 public:
    PciSphSolver3Example(const jet::Frame& frame);

 private:
    jet::PciSphSolver3Ptr _solver;
    jet::viz::PointsRenderable3Ptr _renderable;

    void onSetup(jet::viz::GlfwWindow* window) override;

    void onUpdate(const jet::Frame& frame) override;
};

#endif  // SRC_EXAMPLES_PCI_SPH_SOLVER3_EXAMPLE_H_
