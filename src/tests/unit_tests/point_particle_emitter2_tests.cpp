// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/math_utils.h>
#include <jet/point_particle_emitter2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PointParticleEmitter2, Constructors) {
    PointParticleEmitter2 emitter(
        {1.0, 2.0},
        Vector2D(0.5, 1.0).normalized(),
        3.0,
        15.0,
        4,
        18);

    EXPECT_EQ(4u, emitter.maxNumberOfNewParticlesPerSecond());
    EXPECT_EQ(18u, emitter.maxNumberOfParticles());
}

TEST(PointParticleEmitter2, Emit) {
    Vector2D dir = Vector2D(0.5, 1.0).normalized();

    PointParticleEmitter2 emitter(
        {1.0, 2.0},
        dir,
        3.0,
        15.0,
        4,
        18);

    auto particles = std::make_shared<ParticleSystemData2>();
    emitter.setTarget(particles);

    Frame frame(0, 1.0);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_EQ(4u, particles->numberOfParticles());

    frame.advance();
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_EQ(8u, particles->numberOfParticles());

    frame.advance();
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_EQ(12u, particles->numberOfParticles());

    frame.advance();
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_EQ(16u, particles->numberOfParticles());

    frame.advance();
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_EQ(18u, particles->numberOfParticles());

    auto pos = particles->positions();
    auto vel = particles->velocities();

    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        EXPECT_DOUBLE_EQ(1.0, pos[i].x);
        EXPECT_DOUBLE_EQ(2.0, pos[i].y);

        EXPECT_LE(
            std::cos(degreesToRadians(15.0)),
            vel[i].normalized().dot(dir));
        EXPECT_DOUBLE_EQ(3.0, vel[i].length());
    }
}


TEST(PointParticleEmitter2, Builder) {
    PointParticleEmitter2 emitter = PointParticleEmitter2::builder()
        .withOrigin({1.0, 2.0})
        .withDirection(Vector2D(0.5, 1.0).normalized())
        .withSpeed(3.0)
        .withSpreadAngleInDegrees(15.0)
        .withMaxNumberOfNewParticlesPerSecond(4)
        .withMaxNumberOfParticles(18)
        .build();

    EXPECT_EQ(4u, emitter.maxNumberOfNewParticlesPerSecond());
    EXPECT_EQ(18u, emitter.maxNumberOfParticles());
}
