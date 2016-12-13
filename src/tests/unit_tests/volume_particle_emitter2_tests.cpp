// Copyright (c) 2016 Doyub Kim

#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>
#include <jet/volume_particle_emitter2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(VolumeParticleEmitter2, Constructors) {
    auto sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(Vector2D(1.0, 2.0), 3.0));

    VolumeParticleEmitter2 emitter(
        sphere,
        BoundingBox2D({0.0, 0.0}, {3.0, 3.0}),
        0.1,
        {-1.0, 0.5},
        30,
        0.01,
        false,
        true);

    EXPECT_EQ(0.01, emitter.jitter());
    EXPECT_FALSE(emitter.isOneShot());
    EXPECT_TRUE(emitter.allowOverlapping());
    EXPECT_EQ(30u, emitter.maxNumberOfParticles());
    EXPECT_EQ(0.1, emitter.spacing());
    EXPECT_EQ(-1.0, emitter.initialVelocity().x);
    EXPECT_EQ(0.5, emitter.initialVelocity().y);
}

TEST(VolumeParticleEmitter2, Emit) {
    auto sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(Vector2D(1.0, 2.0), 3.0));

    BoundingBox2D box({0.0, 0.0}, {3.0, 3.0});

    VolumeParticleEmitter2 emitter(
        sphere,
        box,
        0.3,
        {-1.0, 0.5},
        30,
        0.0,
        false,
        false);

    auto particles = std::make_shared<ParticleSystemData2>();
    emitter.setTarget(particles);

    Frame frame(1, 1.0);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    auto pos = particles->positions();
    auto vel = particles->velocities();

    EXPECT_EQ(30u, particles->numberOfParticles());
    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        EXPECT_GE(3.0, (pos[i] - Vector2D(1.0, 2.0)).length());
        EXPECT_TRUE(box.contains(pos[i]));

        EXPECT_EQ(-1.0, vel[i].x);
        EXPECT_EQ(0.5, vel[i].y);
    }

    ++frame;
    emitter.setMaxNumberOfParticles(60);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    EXPECT_EQ(51u, particles->numberOfParticles());

    pos = particles->positions();
    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        pos[i] += Vector2D(2.0, 1.5);
    }

    ++frame;
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_LT(51u, particles->numberOfParticles());
}

TEST(VolumeParticleEmitter2, Builder) {
    auto sphere = std::make_shared<Sphere2>(Vector2D(1.0, 2.0), 3.0);

    VolumeParticleEmitter2 emitter = VolumeParticleEmitter2::builder()
        .withSurface(sphere)
        .withMaxRegion(BoundingBox2D({0.0, 0.0}, {3.0, 3.0}))
        .withSpacing(0.1)
        .withInitialVelocity({-1.0, 0.5})
        .withMaxNumberOfParticles(30)
        .withJitter(0.01)
        .withIsOneShot(false)
        .withAllowOverlapping(true)
        .build();

    EXPECT_EQ(0.01, emitter.jitter());
    EXPECT_FALSE(emitter.isOneShot());
    EXPECT_TRUE(emitter.allowOverlapping());
    EXPECT_EQ(30u, emitter.maxNumberOfParticles());
    EXPECT_EQ(0.1, emitter.spacing());
    EXPECT_EQ(-1.0, emitter.initialVelocity().x);
    EXPECT_EQ(0.5, emitter.initialVelocity().y);
}
