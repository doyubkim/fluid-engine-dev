// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#include "cuda_pci_sph_solver2_example.h"

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/triangle_point_generator.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;
using namespace viz;

CudaPciSphSolver2Example::CudaPciSphSolver2Example()
    : Example(Frame(0, 1.0 / 1000.0)) {}

CudaPciSphSolver2Example::~CudaPciSphSolver2Example() {}

#ifdef JET_USE_GL
void CudaPciSphSolver2Example::onSetup(GlfwWindow* window) {
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

void CudaPciSphSolver2Example::onGui(GlfwWindow* window) { (void)window; }
#else
void CudaPciSphSolver2Example::onSetup() {
    // Setup sim
    setupSim();
}
#endif

void CudaPciSphSolver2Example::onAdvanceSim(const Frame& frame) {
    _solver->update(frame);

    particlesToVertices();
}

void CudaPciSphSolver2Example::onUpdateRenderables() {
    std::lock_guard<std::mutex> lock(_verticesMutex);
    if (_areVerticesDirty) {
        _renderable->vertexBuffer()->updateWithCuda(
            (const float*)thrust::raw_pointer_cast(_vertices.data()));
        _areVerticesDirty = false;
    }
}

void CudaPciSphSolver2Example::setupSim() {
    // Setup solver
    _solver = experimental::CudaPciSphSolver2::builder().makeShared();
    _solver->setViscosityCoefficient(0.002f);
    _solver->setIsUsingFixedSubTimeSteps(true);

    const float targetSpacing = 0.015f;
    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

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
