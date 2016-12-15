// Copyright (c) 2016 Doyub Kim

#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(VolumeParticleEmitter3, Constructors) {
    auto sphere = std::make_shared<SurfaceToImplicit3>(
        std::make_shared<Sphere3>(Vector3D(1.0, 2.0, 4.0), 3.0));

    VolumeParticleEmitter3 emitter(
        sphere,
        BoundingBox3D({0.0, 0.0, 0.0}, {3.0, 3.0, 3.0}),
        0.1,
        {-1.0, 0.5, 2.5},
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
    EXPECT_EQ(2.5, emitter.initialVelocity().z);
}

TEST(VolumeParticleEmitter3, Emit) {
    auto sphere = std::make_shared<SurfaceToImplicit3>(
        std::make_shared<Sphere3>(Vector3D(1.0, 2.0, 4.0), 3.0));

    BoundingBox3D box({0.0, 0.0, 0.0}, {3.0, 3.0, 3.0});

    VolumeParticleEmitter3 emitter(
        sphere,
        box,
        0.5,
        {-1.0, 0.5, 2.5},
        30,
        0.0,
        false,
        false);

    auto particles = std::make_shared<ParticleSystemData3>();
    emitter.setTarget(particles);

    Frame frame(0, 1.0);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    auto pos = particles->positions();
    auto vel = particles->velocities();

    EXPECT_EQ(30u, particles->numberOfParticles());
    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        EXPECT_GE(3.0, (pos[i] - Vector3D(1.0, 2.0, 4.0)).length());
        EXPECT_TRUE(box.contains(pos[i]));

        EXPECT_EQ(-1.0, vel[i].x);
        EXPECT_EQ(0.5, vel[i].y);
        EXPECT_EQ(2.5, vel[i].z);
    }

    ++frame;
    emitter.setMaxNumberOfParticles(80);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    EXPECT_EQ(69u, particles->numberOfParticles());

    pos = particles->positions();
    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        pos[i] += Vector3D(2.0, 1.5, 5.0);
    }

    ++frame;
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_LT(69u, particles->numberOfParticles());
}

TEST(VolumeParticleEmitter3, Builder) {
    auto sphere = std::make_shared<Sphere3>(Vector3D(1.0, 2.0, 4.0), 3.0);

    VolumeParticleEmitter3 emitter = VolumeParticleEmitter3::builder()
        .withSurface(sphere)
        .withMaxRegion(BoundingBox3D({0.0, 0.0, 0.0}, {3.0, 3.0, 3.0}))
        .withSpacing(0.1)
        .withInitialVelocity({-1.0, 0.5, 2.5})
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
    EXPECT_EQ(2.5, emitter.initialVelocity().z);

    auto emitter2 = VolumeParticleEmitter3::builder()
        .withSurface(sphere)
        .withMaxRegion(BoundingBox3D({0.0, 0.0, 0.0}, {3.0, 3.0, 3.0}))
        .withSpacing(0.1)
        .withInitialVelocity({-1.0, 0.5, 2.5})
        .withMaxNumberOfParticles(30)
        .withJitter(0.01)
        .withIsOneShot(false)
        .withAllowOverlapping(true)
        .makeShared();

    emitter2 = nullptr;
}
