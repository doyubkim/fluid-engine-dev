// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/box2.h>
#include <jet/implicit_surface_set2.h>
#include <jet/pci_sph_solver2.h>
#include <jet/plane2.h>
#include <jet/rigid_body_collider2.h>
#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>
#include <jet/volume_particle_emitter2.h>

using namespace jet;

JET_TESTS(PciSphSolver2);

JET_BEGIN_TEST_F(PciSphSolver2, SteadyState) {
    PciSphSolver2 solver;
    solver.setViscosityCoefficient(0.1);
    solver.setPseudoViscosityCoefficient(10.0);

    SphSystemData2Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    const double targetSpacing = particles->targetSpacing();

    BoundingBox2D initialBound(Vector2D(), Vector2D(1, 0.5));
    initialBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter2>(
        std::make_shared<SurfaceToImplicit2>(
            std::make_shared<Sphere2>(Vector2D(), 10.0)),
        initialBound,
        targetSpacing,
        Vector2D());
    emitter->setJitter(0.0);
    solver.setEmitter(emitter);

    Box2Ptr box = std::make_shared<Box2>(Vector2D(), Vector2D(1, 1));
    box->isNormalFlipped = true;
    RigidBodyCollider2Ptr collider = std::make_shared<RigidBodyCollider2>(box);
    solver.setCollider(collider);

    saveParticleDataXy(particles, 0);

    for (Frame frame(0, 1.0 / 60.0) ; frame.index < 100; ++frame) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PciSphSolver2, WaterDrop) {
    const double targetSpacing = 0.02;

    BoundingBox2D domain(Vector2D(), Vector2D(1, 2));

    // Initialize solvers
    PciSphSolver2 solver;
    solver.setPseudoViscosityCoefficient(0.0);

    SphSystemData2Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet2Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet2>();
    surfaceSet->addExplicitSurface(
        std::make_shared<Plane2>(
            Vector2D(0, 1), Vector2D(0, 0.25 * domain.height())));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere2>(
            domain.midPoint(), 0.15 * domain.width()));

    BoundingBox2D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter2>(
        surfaceSet,
        sourceBound,
        targetSpacing,
        Vector2D());
    solver.setEmitter(emitter);

    // Initialize boundary
    Box2Ptr box = std::make_shared<Box2>(domain);
    box->isNormalFlipped = true;
    RigidBodyCollider2Ptr collider = std::make_shared<RigidBodyCollider2>(box);
    solver.setCollider(collider);

    saveParticleDataXy(particles, 0);

    for (Frame frame(0, 1.0 / 60.0) ; frame.index < 120; ++frame) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PciSphSolver2, RotatingTank) {
    const double targetSpacing = 0.02;

    // Build solver
    auto solver = PciSphSolver2::builder()
        .withTargetSpacing(targetSpacing)
        .makeShared();

    solver->setViscosityCoefficient(0.01);

    // Build emitter
    auto box = Box2::builder()
        .withLowerCorner({0.25 + targetSpacing, 0.25 + targetSpacing})
        .withUpperCorner({0.75 - targetSpacing, 0.50})
        .makeShared();

    auto emitter = VolumeParticleEmitter2::builder()
        .withSurface(box)
        .withSpacing(targetSpacing)
        .withIsOneShot(true)
        .makeShared();

    solver->setEmitter(emitter);

    // Build collider
    auto tank = Box2::builder()
        .withLowerCorner({-0.25, -0.25})
        .withUpperCorner({ 0.25,  0.25})
        .withTranslation({0.5, 0.5})
        .withOrientation(0.0)
        .withIsNormalFlipped(true)
        .makeShared();

    auto collider = RigidBodyCollider2::builder()
        .withSurface(tank)
        .withAngularVelocity(2.0)
        .makeShared();

    collider->setOnBeginUpdateCallback([] (Collider2* col, double t, double) {
        if (t < 1.0) {
            col->surface()->transform.setOrientation(2.0 * t);
            static_cast<RigidBodyCollider2*>(col)->angularVelocity = 2.0;
        } else {
            static_cast<RigidBodyCollider2*>(col)->angularVelocity = 0.0;
        }
    });

    solver->setCollider(collider);

    for (Frame frame; frame.index < 120; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
