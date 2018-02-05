// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/rigid_body_collider3.h>
#include <jet/constant_vector_field3.h>
#include <jet/particle_system_solver3.h>
#include <jet/plane3.h>
#include <jet/point_particle_emitter3.h>

using namespace jet;

JET_TESTS(ParticleSystemSolver3);

JET_BEGIN_TEST_F(ParticleSystemSolver3, PerfectBounce) {
    Plane3Ptr plane = std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D());
    RigidBodyCollider3Ptr collider
        = std::make_shared<RigidBodyCollider3>(plane);

    ParticleSystemSolver3 solver;
    solver.setCollider(collider);
    solver.setDragCoefficient(0.0);
    solver.setRestitutionCoefficient(1.0);

    ParticleSystemData3Ptr particles = solver.particleSystemData();
    particles->addParticle({0.0, 3.0, 0.0}, {1.0, 0.0, 0.0});

    Array1<double> x(1000);
    Array1<double> y(1000);
    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(x.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(y.constAccessor(), 0, filename);

    Frame frame;
    frame.timeIntervalInSeconds = 1.0 / 300.0;
    for (; frame.index < 1000; frame.advance()) {
        solver.update(frame);

        x[frame.index] = particles->positions()[0].x;
        y[frame.index] = particles->positions()[0].y;
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,x.npy",
            frame.index);
        saveData(x.constAccessor(), frame.index, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index);
        saveData(y.constAccessor(), frame.index, filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ParticleSystemSolver3, HalfBounce) {
    Plane3Ptr plane = std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D());
    RigidBodyCollider3Ptr collider
        = std::make_shared<RigidBodyCollider3>(plane);

    ParticleSystemSolver3 solver;
    solver.setCollider(collider);
    solver.setDragCoefficient(0.0);
    solver.setRestitutionCoefficient(0.5);

    ParticleSystemData3Ptr particles = solver.particleSystemData();
    particles->addParticle({0.0, 3.0, 0.0}, {1.0, 0.0, 0.0});

    Array1<double> x(1000);
    Array1<double> y(1000);
    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(x.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(y.constAccessor(), 0, filename);

    Frame frame;
    frame.timeIntervalInSeconds = 1.0 / 300.0;
    for (; frame.index < 1000; frame.advance()) {
        solver.update(frame);

        x[frame.index] = particles->positions()[0].x;
        y[frame.index] = particles->positions()[0].y;
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,x.npy",
            frame.index);
        saveData(x.constAccessor(), frame.index, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index);
        saveData(y.constAccessor(), frame.index, filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ParticleSystemSolver3, HalfBounceWithFriction) {
    Plane3Ptr plane = std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D());
    RigidBodyCollider3Ptr collider
        = std::make_shared<RigidBodyCollider3>(plane);
    collider->setFrictionCoefficient(0.04);

    ParticleSystemSolver3 solver;
    solver.setCollider(collider);
    solver.setDragCoefficient(0.0);
    solver.setRestitutionCoefficient(0.5);

    ParticleSystemData3Ptr particles = solver.particleSystemData();
    particles->addParticle({0.0, 3.0, 0.0}, {1.0, 0.0, 0.0});

    Array1<double> x(1000);
    Array1<double> y(1000);
    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(x.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(y.constAccessor(), 0, filename);

    Frame frame;
    frame.timeIntervalInSeconds = 1.0 / 300.0;
    for (; frame.index < 1000; frame.advance()) {
        solver.update(frame);

        x[frame.index] = particles->positions()[0].x;
        y[frame.index] = particles->positions()[0].y;
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,x.npy",
            frame.index);
        saveData(x.constAccessor(), frame.index, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index);
        saveData(y.constAccessor(), frame.index, filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ParticleSystemSolver3, NoBounce) {
    Plane3Ptr plane = std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D());
    RigidBodyCollider3Ptr collider
        = std::make_shared<RigidBodyCollider3>(plane);

    ParticleSystemSolver3 solver;
    solver.setCollider(collider);
    solver.setDragCoefficient(0.0);
    solver.setRestitutionCoefficient(0.0);

    ParticleSystemData3Ptr particles = solver.particleSystemData();
    particles->addParticle({0.0, 3.0, 0.0}, {1.0, 0.0, 0.0});

    Array1<double> x(1000);
    Array1<double> y(1000);
    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(x.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(y.constAccessor(), 0, filename);

    Frame frame;
    frame.timeIntervalInSeconds = 1.0 / 300.0;
    for (; frame.index < 1000; frame.advance()) {
        solver.update(frame);

        x[frame.index] = particles->positions()[0].x;
        y[frame.index] = particles->positions()[0].y;
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,x.npy",
            frame.index);
        saveData(x.constAccessor(), frame.index, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index);
        saveData(y.constAccessor(), frame.index, filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ParticleSystemSolver3, Update) {
    auto plane = Plane3::builder()
        .withNormal({0, 1, 0})
        .withPoint({0, 0, 0})
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(plane)
        .makeShared();

    auto wind = ConstantVectorField3::builder()
        .withValue({1, 0, 0})
        .makeShared();

    auto emitter = PointParticleEmitter3::builder()
        .withOrigin({0, 3, 0})
        .withDirection({0, 1, 0})
        .withSpeed(5)
        .withSpreadAngleInDegrees(45.0)
        .withMaxNumberOfNewParticlesPerSecond(300)
        .makeShared();

    auto solver = ParticleSystemSolver3::builder().makeShared();
    solver->setCollider(collider);
    solver->setEmitter(emitter);
    solver->setWind(wind);
    solver->setDragCoefficient(0.0);
    solver->setRestitutionCoefficient(0.5);

    for (Frame frame(0, 1.0 / 60.0); frame.index < 360; ++frame) {
        solver->update(frame);

        saveParticleDataXy(solver->particleSystemData(), frame.index);
    }
}
JET_END_TEST_F
