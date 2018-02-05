// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/particle_system_solver2.h>
#include <jet/particle_system_solver3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(ParticleSystemSolver2, Constructor) {
    ParticleSystemSolver2 solver;

    auto data = solver.particleSystemData();
    EXPECT_EQ(0u, data->numberOfParticles());

    auto wind = solver.wind();
    EXPECT_TRUE(wind != nullptr);

    auto collider = solver.collider();
    EXPECT_EQ(nullptr, collider);
}

TEST(ParticleSystemSolver2, BasicParams) {
    ParticleSystemSolver2 solver;

    solver.setDragCoefficient(6.0);
    EXPECT_DOUBLE_EQ(6.0, solver.dragCoefficient());

    solver.setDragCoefficient(-7.0);
    EXPECT_DOUBLE_EQ(0.0, solver.dragCoefficient());

    solver.setRestitutionCoefficient(0.5);
    EXPECT_DOUBLE_EQ(0.5, solver.restitutionCoefficient());

    solver.setRestitutionCoefficient(8.0);
    EXPECT_DOUBLE_EQ(1.0, solver.restitutionCoefficient());

    solver.setRestitutionCoefficient(-8.0);
    EXPECT_DOUBLE_EQ(0.0, solver.restitutionCoefficient());

    solver.setGravity(Vector2D(2, -10));
    EXPECT_EQ(Vector2D(2, -10), solver.gravity());
}

TEST(ParticleSystemSolver2, Update) {
    ParticleSystemSolver2 solver;
    solver.setGravity(Vector2D(0, -10));

    ParticleSystemData2Ptr data = solver.particleSystemData();
    ParticleSystemData2::VectorData positions(10);
    data->addParticles(positions.accessor());

    Frame frame(0, 1.0 / 60.0);
    solver.update(frame);

    for (size_t i = 0; i < data->numberOfParticles(); ++i) {
        EXPECT_DOUBLE_EQ(0.0, data->positions()[i].x);
        EXPECT_NE(0, data->positions()[i].y);

        EXPECT_DOUBLE_EQ(0.0, data->velocities()[i].x);
        EXPECT_NE(0, data->velocities()[i].y);
    }
}


TEST(ParticleSystemSolver3, Constructor) {
    ParticleSystemSolver3 solver;

    auto data = solver.particleSystemData();
    EXPECT_EQ(0u, data->numberOfParticles());

    auto wind = solver.wind();
    EXPECT_TRUE(wind != nullptr);

    auto collider = solver.collider();
    EXPECT_EQ(nullptr, collider);
}

TEST(ParticleSystemSolver3, BasicParams) {
    ParticleSystemSolver3 solver;

    solver.setDragCoefficient(6.0);
    EXPECT_DOUBLE_EQ(6.0, solver.dragCoefficient());

    solver.setDragCoefficient(-7.0);
    EXPECT_DOUBLE_EQ(0.0, solver.dragCoefficient());

    solver.setRestitutionCoefficient(0.5);
    EXPECT_DOUBLE_EQ(0.5, solver.restitutionCoefficient());

    solver.setRestitutionCoefficient(8.0);
    EXPECT_DOUBLE_EQ(1.0, solver.restitutionCoefficient());

    solver.setRestitutionCoefficient(-8.0);
    EXPECT_DOUBLE_EQ(0.0, solver.restitutionCoefficient());

    solver.setGravity(Vector3D(3, -10, 7));
    EXPECT_EQ(Vector3D(3, -10, 7), solver.gravity());
}

TEST(ParticleSystemSolver3, Update) {
    ParticleSystemSolver3 solver;
    solver.setGravity(Vector3D(0, -10, 0));

    ParticleSystemData3Ptr data = solver.particleSystemData();
    ParticleSystemData3::VectorData positions(10);
    data->addParticles(positions.accessor());

    Frame frame(0, 1.0 / 60.0);
    solver.update(frame);

    for (size_t i = 0; i < data->numberOfParticles(); ++i) {
        EXPECT_DOUBLE_EQ(0.0, data->positions()[i].x);
        EXPECT_NE(0, data->positions()[i].y);
        EXPECT_DOUBLE_EQ(0.0, data->positions()[i].z);

        EXPECT_DOUBLE_EQ(0.0, data->velocities()[i].x);
        EXPECT_NE(0, data->velocities()[i].y);
        EXPECT_DOUBLE_EQ(0.0, data->velocities()[i].z);
    }
}
