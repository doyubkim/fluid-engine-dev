// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/box3.h>
#include <jet/implicit_surface_set3.h>
#include <jet/pci_sph_solver3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>
#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

using namespace jet;

JET_TESTS(PciSphSolver3);

JET_BEGIN_TEST_F(PciSphSolver3, SteadyState) {
    PciSphSolver3 solver;
    solver.setViscosityCoefficient(0.1);
    solver.setPseudoViscosityCoefficient(10.0);

    SphSystemData3Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    const double targetSpacing = particles->targetSpacing();

    BoundingBox3D initialBound(Vector3D(), Vector3D(1, 0.5, 1));
    initialBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        std::make_shared<SurfaceToImplicit3>(
            std::make_shared<Sphere3>(Vector3D(), 10.0)),
        initialBound,
        targetSpacing,
        Vector3D());
    emitter->setJitter(0.0);
    solver.setEmitter(emitter);

    Box3Ptr box = std::make_shared<Box3>(Vector3D(), Vector3D(1, 1, 1));
    box->isNormalFlipped = true;
    RigidBodyCollider3Ptr collider = std::make_shared<RigidBodyCollider3>(box);
    solver.setCollider(collider);

    saveParticleDataXy(particles, 0);

    for (Frame frame(0, 1.0 / 60.0) ; frame.index < 100; ++frame) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(PciSphSolver3, WaterDrop) {
    const double targetSpacing = 0.02;

    BoundingBox3D domain(Vector3D(), Vector3D(1, 2, 0.5));

    // Initialize solvers
    PciSphSolver3 solver;
    solver.setPseudoViscosityCoefficient(0.0);

    SphSystemData3Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addExplicitSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet->addExplicitSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        sourceBound,
        targetSpacing,
        Vector3D());
    solver.setEmitter(emitter);

    // Initialize boundary
    Box3Ptr box = std::make_shared<Box3>(domain);
    box->isNormalFlipped = true;
    RigidBodyCollider3Ptr collider = std::make_shared<RigidBodyCollider3>(box);
    solver.setCollider(collider);

    saveParticleDataXy(particles, 0);

    for (Frame frame(0, 1.0 / 60.0) ; frame.index < 100; ++frame) {
        solver.update(frame);

        saveParticleDataXy(particles, frame.index);
    }
}
JET_END_TEST_F
