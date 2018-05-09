// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_PARTICLE_SIM_CUDA_PCI_SPH_SOLVER2_EXAMPLE_H_
#define SRC_EXAMPLES_PARTICLE_SIM_CUDA_PCI_SPH_SOLVER2_EXAMPLE_H_

#ifdef JET_USE_CUDA

#include <example.h>

#include <jet.viz/points_renderable2.h>
#include <jet.viz/vertex.h>
#include <jet/cuda_pci_sph_solver2.h>

#include <thrust/device_vector.h>

#include <mutex>

class CudaPciSphSolver2Example final : public Example {
 public:
    CudaPciSphSolver2Example();

    ~CudaPciSphSolver2Example();

    std::string name() const override;

 private:
    jet::CudaPciSphSolver2Ptr _solver;
    jet::viz::PointsRenderable2Ptr _renderable;
    thrust::device_vector<jet::viz::VertexPosition3Color4> _vertices;

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

#endif  // JET_USE_CUDA

#endif  // SRC_EXAMPLES_PARTICLE_SIM_CUDA_PCI_SPH_SOLVER2_EXAMPLE_H_
