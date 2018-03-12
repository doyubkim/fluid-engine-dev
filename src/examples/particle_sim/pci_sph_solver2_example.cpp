// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pci_sph_solver2_example.h"

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/volume_particle_emitter2.h>

#include <imgui/imgui.h>

using namespace jet;
using namespace viz;

PciSphSolver2Example::PciSphSolver2Example()
    : Example(Frame(0, 1.0 / 1000.0)) {}

PciSphSolver2Example::~PciSphSolver2Example() {}

std::string PciSphSolver2Example::name() const { return "2-D PCISPH Example"; }

void PciSphSolver2Example::onRestartSim() {
    setupSim();

    // Reload parameters
    _solver->setViscosityCoefficient(_viscosityCoefficient);
    _solver->setPseudoViscosityCoefficient(_pseudoViscosityCoefficient);
    _solver->setMaxDensityErrorRatio(_maxDensityErrorRatio);
    _solver->setMaxNumberOfIterations(_maxNumberOfIterations);

    advanceSim();
    updateRenderables();
}

#ifdef JET_USE_GL
void PciSphSolver2Example::onSetup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<OrthoViewController>(
        std::make_shared<OrthoCamera>(0, 1, 0, 2)));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Setup sim
    setupSim();

    // Reset parameters
    _viscosityCoefficient = _solver->viscosityCoefficient();
    _pseudoViscosityCoefficient = _solver->pseudoViscosityCoefficient();
    _maxDensityErrorRatio = _solver->maxDensityErrorRatio();
    _maxNumberOfIterations = _solver->maxNumberOfIterations();

    // Setup renderable
    _areVerticesDirty = false;
    _renderable = std::make_shared<PointsRenderable2>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    renderer->addRenderable(_renderable);

    advanceSim();
    updateRenderables();
}

void PciSphSolver2Example::onGui(GlfwWindow* window) {
    (void)window;

    ImGui::Begin("Parameters");
    {
        auto viscosity = (float)_viscosityCoefficient;
        auto pseudoViscosity = (float)_pseudoViscosityCoefficient;
        auto maxDensityErrorRatio = (float)_maxDensityErrorRatio;
        auto maxNumberOfIterations = (int)_maxNumberOfIterations;

        ImGui::SliderFloat("Viscosity", &viscosity, 0.0f, 0.05f);
        ImGui::SliderFloat("Pseudo-viscosity", &pseudoViscosity, 0.0f, 10.f);
        ImGui::SliderFloat("Max. density error ratio", &maxDensityErrorRatio,
                           0.0f, 1.f);
        ImGui::SliderInt("Max. Iterations", &maxNumberOfIterations, 1, 10);

        _viscosityCoefficient = viscosity;
        _pseudoViscosityCoefficient = pseudoViscosity;
        _maxDensityErrorRatio = maxDensityErrorRatio;
        _maxNumberOfIterations = (unsigned int)maxNumberOfIterations;
    }
    ImGui::End();
}
#else
void PciSphSolver2Example::onSetup() {
    // Setup sim
    setupSim();
}
#endif

void PciSphSolver2Example::onAdvanceSim(const Frame& frame) {
    _solver->setViscosityCoefficient(_viscosityCoefficient);
    _solver->setPseudoViscosityCoefficient(_pseudoViscosityCoefficient);
    _solver->setMaxDensityErrorRatio(_maxDensityErrorRatio);
    _solver->setMaxNumberOfIterations(_maxNumberOfIterations);

    _solver->update(frame);

    particlesToVertices();
}

void PciSphSolver2Example::onUpdateRenderables() {
    std::lock_guard<std::mutex> lock(_verticesMutex);
    if (_areVerticesDirty) {
        _renderable->setPositions(_vertices.data(), _vertices.size());
        _areVerticesDirty = false;
    }
}

void PciSphSolver2Example::setupSim() {
    // Setup solver
    _solver = PciSphSolver2::builder().makeShared();
    _solver->setViscosityCoefficient(0.002);
    _solver->setIsUsingFixedSubTimeSteps(true);

    const double targetSpacing = 0.015;
    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

    SphSystemData2Ptr particles = _solver->sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet2Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addExplicitSurface(std::make_shared<Plane2>(
        Vector2D(0, 1), Vector2D(0, 0.25 * domain.height())));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(domain.midPoint(), 0.15 * domain.width()));

    BoundingBox2D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter2>(
        surfaceSet, sourceBound, targetSpacing, Vector2D());
    _solver->setEmitter(emitter);

    // Initialize boundary
    Box2Ptr box = std::make_shared<Box2>(domain);
    box->isNormalFlipped = true;
    RigidBodyCollider2Ptr collider = std::make_shared<RigidBodyCollider2>(box);
    _solver->setCollider(collider);
}

void PciSphSolver2Example::particlesToVertices() {
    const auto pos = _solver->sphSystemData()->positions();

    {
        std::lock_guard<std::mutex> lock(_verticesMutex);
        _vertices.resize(pos.size());
        for (size_t i = 0; i < pos.size(); ++i) {
            _vertices[i] = pos[i].castTo<float>();
        }
        _areVerticesDirty = true;
    }
}
