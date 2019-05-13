// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "unit_tests_utils.h"

#include <jet/sphere2.h>
#include <jet/surface_to_implicit2.h>
#include <jet/volume_particle_emitter2.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(VolumeParticleEmitter2, Constructors) {
    auto sphere = std::make_shared<SurfaceToImplicit2>(
        std::make_shared<Sphere2>(Vector2D(1.0, 2.0), 3.0));

    BoundingBox2D region({0.0, 0.0}, {3.0, 3.0});

    VolumeParticleEmitter2 emitter(
        sphere,
        region,
        0.1,
        {-1.0, 0.5},
        {0.0, 0.0},
        0.0,
        30,
        0.01,
        false,
        true);

    EXPECT_BOUNDING_BOX2_EQ(region, emitter.maxRegion());
    EXPECT_EQ(0.01, emitter.jitter());
    EXPECT_FALSE(emitter.isOneShot());
    EXPECT_TRUE(emitter.allowOverlapping());
    EXPECT_EQ(30u, emitter.maxNumberOfParticles());
    EXPECT_EQ(0.1, emitter.spacing());
    EXPECT_EQ(-1.0, emitter.initialVelocity().x);
    EXPECT_EQ(0.5, emitter.initialVelocity().y);
    EXPECT_EQ(Vector2D(), emitter.linearVelocity());
    EXPECT_EQ(0.0, emitter.angularVelocity());
    EXPECT_TRUE(emitter.isEnabled());
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
        {3.0, 4.0},
        5.0,
        30,
        0.0,
        false,
        false);

    auto particles = std::make_shared<ParticleSystemData2>();
    emitter.setTarget(particles);

    Frame frame(0, 1.0);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    auto pos = particles->positions();
    auto vel = particles->velocities();

    EXPECT_EQ(30u, particles->numberOfParticles());
    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        EXPECT_GE(3.0, (pos[i] - Vector2D(1.0, 2.0)).length());
        EXPECT_TRUE(box.contains(pos[i]));

        Vector2D r = pos[i];
        Vector2D w = 5.0 * Vector2D(-r.y, r.x);
        EXPECT_VECTOR2_NEAR(Vector2D(2.0, 4.5) + w, vel[i], 1e-9);
    }

    emitter.setIsEnabled(false);
    ++frame;
    emitter.setMaxNumberOfParticles(60);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    EXPECT_EQ(30u, particles->numberOfParticles());
    emitter.setIsEnabled(true);

    ++frame;
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
        .withLinearVelocity({3.0, 4.0})
        .withAngularVelocity(5.0)
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
    EXPECT_VECTOR2_EQ(Vector2D(3.0, 4.0), emitter.linearVelocity());
    EXPECT_EQ(5.0, emitter.angularVelocity());
}
