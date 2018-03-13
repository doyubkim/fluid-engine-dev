// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "wc_sph_solver2_example.h"

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/volume_particle_emitter2.h>

#include <imgui/imgui.h>

using namespace jet;
using namespace viz;

WcSphSolver2Example::WcSphSolver2Example() : Example(Frame(0, 1.0 / 1000.0)) {}

WcSphSolver2Example::~WcSphSolver2Example() {}

std::string WcSphSolver2Example::name() const { return "2-D WCISPH Example"; }

void WcSphSolver2Example::onRestartSim() {
    setupSim();

    // Reload parameters
    _solver->setViscosityCoefficient(_viscosityCoefficient);
    _solver->setPseudoViscosityCoefficient(_pseudoViscosityCoefficient);

    advanceSim();
    updateRenderables();
}

#ifdef JET_USE_GL
void WcSphSolver2Example::onSetup(GlfwWindow* window) {
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

    // Setup renderable
    _areVerticesDirty = false;
    _renderable = std::make_shared<PointsRenderable2>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    renderer->addRenderable(_renderable);

    advanceSim();
    updateRenderables();
}

void WcSphSolver2Example::onGui(GlfwWindow* window) {
    (void)window;

    ImGui::Begin("Parameters");
    {
        auto viscosity = (float)_viscosityCoefficient;
        auto pseudoViscosity = (float)_pseudoViscosityCoefficient;

        ImGui::SliderFloat("Viscosity", &viscosity, 0.0f, 0.05f);
        ImGui::SliderFloat("Pseudo-viscosity", &pseudoViscosity, 0.0f, 10.f);

        _viscosityCoefficient = viscosity;
        _pseudoViscosityCoefficient = pseudoViscosity;
    }
    ImGui::End();
}
#else
void WcSphSolver2Example::onSetup() {
    // Setup sim
    setupSim();
}
#endif

void WcSphSolver2Example::onAdvanceSim(const Frame& frame) {
    _solver->setViscosityCoefficient(_viscosityCoefficient);
    _solver->setPseudoViscosityCoefficient(_pseudoViscosityCoefficient);

    _solver->update(frame);

    particlesToVertices();
}

void WcSphSolver2Example::onUpdateRenderables() {
    std::lock_guard<std::mutex> lock(_verticesMutex);
    if (_areVerticesDirty) {
        _renderable->setPositionsAndColors(_vertices.data(), _vertices.size());
        _areVerticesDirty = false;
    }
}

void WcSphSolver2Example::setupSim() {
    // Setup solver
    _solver = SphSolver2::builder().makeShared();
    _solver->setViscosityCoefficient(0.02);
    _solver->setIsUsingFixedSubTimeSteps(false);

    const double targetSpacing = 0.015;
    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

    SphSystemData2Ptr particles = _solver->sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet2Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addExplicitSurface(std::make_shared<Plane2>(
        Vector2D(0, 1), Vector2D(0, 0.25 * domain.height())));
    // surfaceSet->addExplicitSurface(
    //    std::make_shared<Sphere2>(domain.midPoint(), 0.15 * domain.width()));

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

void WcSphSolver2Example::particlesToVertices() {
    const auto pos = _solver->sphSystemData()->positions();
    const auto den = _solver->sphSystemData()->densities();

    {
        std::lock_guard<std::mutex> lock(_verticesMutex);
        _vertices.resize(pos.size());
        for (size_t i = 0; i < pos.size(); ++i) {
            auto p = pos[i].castTo<float>();
            float d = (float)den[i];

            d = std::min(std::max(d / 1000.0f, 0.0f), 1.0f);
            VertexPosition3Color4 vertex;
            vertex.x = p.x;
            vertex.y = p.y;
            vertex.z = 0.0f;
            vertex.r = 1.0f;
            vertex.g = 1.0f - d;
            vertex.b = 1.0f - d;
            vertex.a = 1.0f;

            _vertices[i] = vertex;
        }
        _areVerticesDirty = true;
    }
}
