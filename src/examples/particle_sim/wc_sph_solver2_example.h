// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_EXAMPLES_PARTICLE_SIM_WC_SPH_SOLVER2_EXAMPLE_H_
#define SRC_EXAMPLES_PARTICLE_SIM_WC_SPH_SOLVER2_EXAMPLE_H_

#include <example.h>

#include <jet.viz/points_renderable2.h>
#include <jet.viz/vertex.h>
#include <jet/sph_solver2.h>

#include <atomic>
#include <mutex>

class WcSphSolver2Example final : public Example {
 public:
    WcSphSolver2Example();

    ~WcSphSolver2Example();

    std::string name() const override;

 private:
    jet::SphSolver2Ptr _solver;
    jet::viz::PointsRenderable2Ptr _renderable;
    jet::Array1<jet::viz::VertexPosition3Color4> _vertices;

    bool _areVerticesDirty = false;
    std::mutex _verticesMutex;

    std::atomic<double> _viscosityCoefficient;
    std::atomic<double> _pseudoViscosityCoefficient;

    void onRestartSim() override;

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

#endif  // SRC_EXAMPLES_PARTICLE_SIM_WC_SPH_SOLVER2_EXAMPLE_H_
