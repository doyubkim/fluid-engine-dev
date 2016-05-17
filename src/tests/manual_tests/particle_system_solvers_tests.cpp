// Copyright (c) 2016 Doyub Kim

#include <manual_tests.h>

#include <jet/rigid_body_collider2.h>
#include <jet/rigid_body_collider3.h>
#include <jet/constant_vector_field2.h>
#include <jet/constant_vector_field3.h>
#include <jet/particle_system_solver2.h>
#include <jet/particle_system_solver3.h>
#include <jet/plane2.h>
#include <jet/plane3.h>
#include <jet/point_particle_emitter2.h>
#include <jet/point_particle_emitter3.h>

using namespace jet;

JET_TESTS(ParticleSystemSolver2);

JET_BEGIN_TEST_F(ParticleSystemSolver2, Update) {
    Plane2Ptr plane = std::make_shared<Plane2>(Vector2D(0, 1), Vector2D());
    RigidBodyCollider2Ptr collider
        = std::make_shared<RigidBodyCollider2>(plane);
    ConstantVectorField2Ptr wind
        = std::make_shared<ConstantVectorField2>(Vector2D(1, 0));

    ParticleSystemSolver2 solver;
    solver.setCollider(collider);
    solver.setWind(wind);

    ParticleSystemData2Ptr particles = solver.particleSystemData();
    PointParticleEmitter2Ptr emitter
        = std::make_shared<PointParticleEmitter2>(
            Vector2D(0, 3),
            Vector2D(0, 1), 5.0, 45.0);
    emitter->setMaxNumberOfNewParticlesPerSecond(100);

    saveParticleDataXy(particles, 0);

    for (Frame frame; frame.index < 360; frame.advance()) {
        emitter->emit(frame, particles);
        solver.update(frame);

        saveParticleDataXy(particles, frame.index + 1);
    }
}
JET_END_TEST_F


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
            frame.index + 1);
        saveData(x.constAccessor(), frame.index + 1, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index + 1);
        saveData(y.constAccessor(), frame.index + 1, filename);
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
            frame.index + 1);
        saveData(x.constAccessor(), frame.index + 1, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index + 1);
        saveData(y.constAccessor(), frame.index + 1, filename);
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
            frame.index + 1);
        saveData(x.constAccessor(), frame.index + 1, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index + 1);
        saveData(y.constAccessor(), frame.index + 1, filename);
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
            frame.index + 1);
        saveData(x.constAccessor(), frame.index + 1, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index + 1);
        saveData(y.constAccessor(), frame.index + 1, filename);
    }
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ParticleSystemSolver3, Update) {
    Plane3Ptr plane = std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D());
    RigidBodyCollider3Ptr collider
        = std::make_shared<RigidBodyCollider3>(plane);
    ConstantVectorField3Ptr wind
        = std::make_shared<ConstantVectorField3>(Vector3D(1, 0, 0));

    ParticleSystemSolver3 solver;
    solver.setCollider(collider);
    solver.setWind(wind);
    solver.setDragCoefficient(0.0);
    solver.setRestitutionCoefficient(0.5);

    ParticleSystemData3Ptr particles = solver.particleSystemData();
    PointParticleEmitter3Ptr emitter
        = std::make_shared<PointParticleEmitter3>(
            Vector3D(0, 3, 0), Vector3D(0, 1, 0), 5.0, 45.0);
    emitter->setMaxNumberOfNewParticlesPerSecond(300);

    saveParticleDataXy(particles, 0);

    for (Frame frame; frame.index < 360; frame.advance()) {
        emitter->emit(frame, particles);
        solver.update(frame);

        saveParticleDataXy(particles, frame.index + 1);
    }
}
JET_END_TEST_F
