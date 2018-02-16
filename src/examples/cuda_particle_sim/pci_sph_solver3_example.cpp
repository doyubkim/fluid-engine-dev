// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pci_sph_solver3_example.h"

#include <jet.viz/points_renderable3.h>
#include <jet/box3.h>
#include <jet/grid_point_generator3.h>
#include <jet/rigid_body_collider3.h>

using namespace jet;
using namespace viz;

PciSphSolver3Example::PciSphSolver3Example(const jet::Frame& frame)
    : ParticleSimExample(frame) {}

void PciSphSolver3Example::onSetup(GlfwWindow* window) {
    // Setup desired view controller
    window->setViewController(std::make_shared<PitchYawViewController>(
        std::make_shared<PerspCamera>(Vector3D{0.5, 0.5, 2.0},
                                      Vector3D{0.0, 0.0, -1.0},
                                      Vector3D{0.0, 1.0, 0.0}, 0.01, 10.0),
        Vector3D{0.5, 0.5, 0.5}));

    // Setup desired background
    auto renderer = window->renderer().get();
    renderer->setBackgroundColor(Color{0, 0, 0, 1});

    // Setup solver
    const float targetSpacing = 1.0f / 50.0f;
    _solver = jet::PciSphSolver3::builder().makeShared();
    _solver->setDragCoefficient(0.0f);
    _solver->setRestitutionCoefficient(0.0f);
    _solver->setViscosityCoefficient(0.1f);
    _solver->setPseudoViscosityCoefficient(10.0f);
    _solver->setIsUsingFixedSubTimeSteps(true);
    _solver->setNumberOfFixedSubTimeSteps(1);

    auto particles = _solver->sphSystemData();
    particles->setTargetSpacing(targetSpacing);
    particles->setRelativeKernelRadius(1.8f);

    Box3Ptr box = std::make_shared<Box3>(Vector3D(), Vector3D(1, 1, 1));
    box->isNormalFlipped = true;
    RigidBodyCollider3Ptr collider = std::make_shared<RigidBodyCollider3>(box);
    _solver->setCollider(collider);

    // Seed particles
    BoundingBox3D vol(Vector3D(), Vector3D(0.5, 0.5, 0.5));
    vol.expand(-targetSpacing);
    Array1<Vector3D> rawPoints;
    GridPointGenerator3 generator;
    generator.generate(vol, targetSpacing, &rawPoints);
    particles->addParticles(rawPoints);
    printf("%zu particles generated.\n", rawPoints.size());

    Array1<Vector3F> vertices;
    for (auto p : rawPoints) {
        vertices.append(p.castTo<float>());
    }

    // Setup renderable
    _renderable = std::make_shared<PointsRenderable3>(renderer);
    _renderable->setRadius(5.0f * (float)window->framebufferSize().x /
                           window->windowSize().x);
    _renderable->setPositions(vertices.data(), vertices.size());
    renderer->addRenderable(_renderable);
}

void PciSphSolver3Example::onUpdate(const Frame& frame) {
    _solver->update(frame);

    auto pos = _solver->sphSystemData()->positions();
    auto den = _solver->sphSystemData()->densities();

    Array1<Vector3F> vertices;
    for (auto p : pos) {
        vertices.append(p.castTo<float>());
    }
    _renderable->setPositions(vertices.data(), vertices.size());
}
