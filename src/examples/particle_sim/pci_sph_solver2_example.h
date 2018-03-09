// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_PARTICLE_SIM_PCI_SPH_SOLVER2_EXAMPLE_H_
#define SRC_EXAMPLES_PARTICLE_SIM_PCI_SPH_SOLVER2_EXAMPLE_H_

#include <example.h>

#include <jet.viz/points_renderable2.h>
#include <jet/pci_sph_solver2.h>

#include <mutex>

class PciSphSolver2Example final : public Example {
 public:
    PciSphSolver2Example();

    ~PciSphSolver2Example();

 private:
    jet::PciSphSolver2Ptr _solver;
    jet::viz::PointsRenderable2Ptr _renderable;
    jet::Array1<jet::Vector2F> _vertices;

    bool _areVerticesDirty = false;
    std::mutex _verticesMutex;

#ifdef JET_USE_GL
    void onSetup(jet::viz::GlfwWindow* window) override;

    void onGui(jet::viz::GlfwWindow* window) override;
#else
    void onSetup() override;
#endif

    void onAdvanceSim(const jet::Frame& frame) override;

    void onUpdateRenderables() override;

    void setupSim();

    void particlesToVertices();
};

#endif  // SRC_EXAMPLES_PARTICLE_SIM_PCI_SPH_SOLVER2_EXAMPLE_H_
