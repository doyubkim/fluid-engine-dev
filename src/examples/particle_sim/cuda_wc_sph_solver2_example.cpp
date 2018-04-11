// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include "cuda_wc_sph_solver2_example.h"

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/triangle_point_generator.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;
using namespace viz;

CudaWcSphSolver2Example::CudaWcSphSolver2Example()
    : Example(Frame(0, 1.0 / 1000.0)) {}

CudaWcSphSolver2Example::~CudaWcSphSolver2Example() {}

std::string CudaWcSphSolver2Example::name() const {
    return "2-D CUDA WCSPH Example";
}

#ifdef JET_USE_GL
void CudaWcSphSolver2Example::onSetup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<OrthoViewController>(
        std::make_shared<OrthoCamera>(0, 1, 0, 2)));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Setup sim
    setupSim();

    // Setup renderable
    _areVerticesDirty = false;
    _renderable = std::make_shared<PointsRenderable2>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    _renderable->setPositionsAndColors(
        nullptr, _solver->particleSystemData()->positions().size());
    renderer->addRenderable(_renderable);

    advanceSim();
    updateRenderables();
}

void CudaWcSphSolver2Example::onGui(GlfwWindow* window) { (void)window; }
#else
void CudaWcSphSolver2Example::onSetup() {
    // Setup sim
    setupSim();
}
#endif

void CudaWcSphSolver2Example::onAdvanceSim(const Frame& frame) {
    _solver->update(frame);

    particlesToVertices();
}

void CudaWcSphSolver2Example::onUpdateRenderables() {
    std::lock_guard<std::mutex> lock(_verticesMutex);
    if (_areVerticesDirty) {
        _renderable->vertexBuffer()->updateWithCuda(
            (const float*)thrust::raw_pointer_cast(_vertices.data()));
        _areVerticesDirty = false;
    }
}

void CudaWcSphSolver2Example::setupSim() {
    // Setup solver
    _solver = experimental::CudaWcSphSolver2::builder().makeShared();
    _solver->setViscosityCoefficient(0.02f);
    _solver->setIsUsingFixedSubTimeSteps(false);

    const float targetSpacing = 0.03f;
    BoundingBox2F domain(Vector2F(), Vector2F(1, 2));
    domain.expand(-targetSpacing);
    _solver->setContainer(domain);

    auto particles = _solver->sphSystemData();
    particles->setTargetDensity(1000.0f);
    particles->setTargetSpacing(targetSpacing);
    particles->setRelativeKernelRadius(1.8f);

    // Seed particles
    BoundingBox2D vol(Vector2D(), Vector2D(1, 0.5));
    vol.expand(-targetSpacing);
    Array1<Vector2D> pointsD;
    TrianglePointGenerator generator;
    generator.generate(vol, targetSpacing, &pointsD);
    Array1<Vector2F> pointsF(pointsD.size());
    for (size_t i = 0; i < pointsF.size(); ++i) {
        pointsF[i] = pointsD[i].castTo<float>();
    }
    particles->addParticles(pointsF);
}

#endif  // JET_USE_CUDA
