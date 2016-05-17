// Copyright (c) 2016 Doyub Kim

#include <jet/rigid_body_collider3.h>
#include <jet/plane3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(RigidBodyCollider3, ResolveCollision) {
    // 1. No penetration
    {
        RigidBodyCollider3 collider(
            std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D(0, 0, 0)));

        Vector3D newPosition(1, 0.1, 0);
        Vector3D newVelocity(1, 0, 0);
        double radius = 0.05;
        double restitutionCoefficient = 0.5;

        collider.resolveCollision(
            radius,
            restitutionCoefficient,
            &newPosition,
            &newVelocity);

        EXPECT_DOUBLE_EQ(1.0, newPosition.x);
        EXPECT_DOUBLE_EQ(0.1, newPosition.y);
        EXPECT_DOUBLE_EQ(0.0, newPosition.z);
        EXPECT_DOUBLE_EQ(1.0, newVelocity.x);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.y);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.z);
    }

    // 2. Penetration within radius
    {
        RigidBodyCollider3 collider(
            std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D(0, 0, 0)));

        Vector3D newPosition(1, 0.1, 0);
        Vector3D newVelocity(1, 0, 0);
        double radius = 0.2;
        double restitutionCoefficient = 0.5;

        collider.resolveCollision(
            radius,
            restitutionCoefficient,
            &newPosition,
            &newVelocity);

        EXPECT_DOUBLE_EQ(1.0, newPosition.x);
        EXPECT_DOUBLE_EQ(0.2, newPosition.y);
        EXPECT_DOUBLE_EQ(0.0, newPosition.z);
    }

    // 3. Sitting
    {
        RigidBodyCollider3 collider(
            std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D(0, 0, 0)));

        Vector3D newPosition(1, 0.1, 0);
        Vector3D newVelocity(1, 0, 0);
        double radius = 0.1;
        double restitutionCoefficient = 0.5;

        collider.resolveCollision(
            radius,
            restitutionCoefficient,
            &newPosition,
            &newVelocity);

        EXPECT_DOUBLE_EQ(1.0, newPosition.x);
        EXPECT_DOUBLE_EQ(0.1, newPosition.y);
        EXPECT_DOUBLE_EQ(0.0, newPosition.z);
        EXPECT_DOUBLE_EQ(1.0, newVelocity.x);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.y);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.z);
    }

    // 4. Bounce-back
    {
        RigidBodyCollider3 collider(
            std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D(0, 0, 0)));

        Vector3D newPosition(1, -1, 0);
        Vector3D newVelocity(1, -1, 0);
        double radius = 0.1;
        double restitutionCoefficient = 0.5;

        collider.resolveCollision(
            radius,
            restitutionCoefficient,
            &newPosition,
            &newVelocity);

        EXPECT_DOUBLE_EQ(1.0, newPosition.x);
        EXPECT_DOUBLE_EQ(0.1, newPosition.y);
        EXPECT_DOUBLE_EQ(0.0, newPosition.z);
        EXPECT_DOUBLE_EQ(1.0, newVelocity.x);
        EXPECT_DOUBLE_EQ(restitutionCoefficient, newVelocity.y);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.z);
    }

    // 4. Friction
    {
        RigidBodyCollider3 collider(
            std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D(0, 0, 0)));

        Vector3D newPosition(1, -1, 0);
        Vector3D newVelocity(1, -1, 0);
        double radius = 0.1;
        double restitutionCoefficient = 0.5;

        collider.setFrictionCoefficient(0.1);

        collider.resolveCollision(
            radius,
            restitutionCoefficient,
            &newPosition,
            &newVelocity);

        EXPECT_DOUBLE_EQ(1.0, newPosition.x);
        EXPECT_DOUBLE_EQ(0.1, newPosition.y);
        EXPECT_DOUBLE_EQ(0.0, newPosition.z);
        EXPECT_GT(1.0, newVelocity.x);
        EXPECT_DOUBLE_EQ(restitutionCoefficient, newVelocity.y);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.z);
    }
}
