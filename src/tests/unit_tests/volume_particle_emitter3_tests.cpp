// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "unit_tests_utils.h"

#include <jet/sphere3.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_particle_emitter3.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(VolumeParticleEmitter3, Constructors) {
    auto sphere = std::make_shared<SurfaceToImplicit3>(
        std::make_shared<Sphere3>(Vector3D(1.0, 2.0, 4.0), 3.0));

    BoundingBox3D region({0.0, 0.0, 0.0}, {3.0, 3.0, 3.0});

    VolumeParticleEmitter3 emitter(
        sphere,
        region,
        0.1,
        {-1.0, 0.5, 2.5},
        {0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        30,
        0.01,
        false,
        true);

    EXPECT_BOUNDING_BOX3_EQ(region, emitter.maxRegion());
    EXPECT_EQ(0.01, emitter.jitter());
    EXPECT_FALSE(emitter.isOneShot());
    EXPECT_TRUE(emitter.allowOverlapping());
    EXPECT_EQ(30u, emitter.maxNumberOfParticles());
    EXPECT_EQ(0.1, emitter.spacing());
    EXPECT_EQ(-1.0, emitter.initialVelocity().x);
    EXPECT_EQ(0.5, emitter.initialVelocity().y);
    EXPECT_EQ(2.5, emitter.initialVelocity().z);
    EXPECT_EQ(Vector3D(), emitter.linearVelocity());
    EXPECT_EQ(Vector3D(), emitter.angularVelocity());
    EXPECT_TRUE(emitter.isEnabled());
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
        {3.0, 4.0, 5.0},
        {0.0, 0.0, 5.0},
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

        Vector3D r = pos[i];
        Vector3D w = 5.0 * Vector3D(-r.y, r.x, 0.0);
        EXPECT_VECTOR3_NEAR(Vector3D(2.0, 4.5, 7.5) + w, vel[i], 1e-9);
    }

    emitter.setIsEnabled(false);
    ++frame;
    emitter.setMaxNumberOfParticles(80);
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    EXPECT_EQ(30u, particles->numberOfParticles());
    emitter.setIsEnabled(true);

    ++frame;
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);

    EXPECT_EQ(79u, particles->numberOfParticles());

    pos = particles->positions();
    for (size_t i = 0; i < particles->numberOfParticles(); ++i) {
        pos[i] += Vector3D(2.0, 1.5, 5.0);
    }

    ++frame;
    emitter.update(frame.timeInSeconds(), frame.timeIntervalInSeconds);
    EXPECT_LT(79u, particles->numberOfParticles());
}

TEST(VolumeParticleEmitter3, Builder) {
    auto sphere = std::make_shared<Sphere3>(Vector3D(1.0, 2.0, 4.0), 3.0);

    VolumeParticleEmitter3 emitter = VolumeParticleEmitter3::builder()
        .withSurface(sphere)
        .withMaxRegion(BoundingBox3D({0.0, 0.0, 0.0}, {3.0, 3.0, 3.0}))
        .withSpacing(0.1)
        .withInitialVelocity({-1.0, 0.5, 2.5})
        .withLinearVelocity({3.0, 4.0, 5.0})
        .withAngularVelocity({0.0, 1.0, 2.0})
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
    EXPECT_VECTOR3_EQ(Vector3D(3.0, 4.0, 5.0), emitter.linearVelocity());
    EXPECT_VECTOR3_EQ(Vector3D(0.0, 1.0, 2.0), emitter.angularVelocity());
}
