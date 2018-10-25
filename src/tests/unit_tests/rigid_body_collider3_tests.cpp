// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/implicit_surface_set3.h>
#include <jet/plane3.h>
#include <jet/rigid_body_collider3.h>

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

        collider.resolveCollision(radius, restitutionCoefficient, &newPosition,
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

        collider.resolveCollision(radius, restitutionCoefficient, &newPosition,
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

        collider.resolveCollision(radius, restitutionCoefficient, &newPosition,
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

        collider.resolveCollision(radius, restitutionCoefficient, &newPosition,
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

        collider.resolveCollision(radius, restitutionCoefficient, &newPosition,
                                  &newVelocity);

        EXPECT_DOUBLE_EQ(1.0, newPosition.x);
        EXPECT_DOUBLE_EQ(0.1, newPosition.y);
        EXPECT_DOUBLE_EQ(0.0, newPosition.z);
        EXPECT_GT(1.0, newVelocity.x);
        EXPECT_DOUBLE_EQ(restitutionCoefficient, newVelocity.y);
        EXPECT_DOUBLE_EQ(0.0, newVelocity.z);
    }
}

TEST(RigidBodyCollider3, VelocityAt) {
    RigidBodyCollider3 collider(
        std::make_shared<Plane3>(Vector3D(0, 1, 0), Vector3D(0, 0, 0)));

    collider.linearVelocity = {1, 3, -2};
    collider.angularVelocity = {0, 0, 4};
    collider.surface()->transform.setTranslation({-1, -2, 2});
    collider.surface()->transform.setOrientation(QuaternionD({1, 0, 0}, 0.1));

    Vector3D result = collider.velocityAt({5, 7, 8});
    EXPECT_DOUBLE_EQ(-35.0, result.x);
    EXPECT_DOUBLE_EQ(27.0, result.y);
    EXPECT_DOUBLE_EQ(-2.0, result.z);
}

TEST(RigidBodyCollider3, Empty) {
    RigidBodyCollider3 collider(ImplicitSurfaceSet3::builder().makeShared());

    Vector3D newPosition(1, 0.1, 0);
    Vector3D newVelocity(1, 0, 0);
    double radius = 0.05;
    double restitutionCoefficient = 0.5;

    collider.resolveCollision(radius, restitutionCoefficient, &newPosition,
                              &newVelocity);

    EXPECT_DOUBLE_EQ(1.0, newPosition.x);
    EXPECT_DOUBLE_EQ(0.1, newPosition.y);
    EXPECT_DOUBLE_EQ(0.0, newPosition.z);
    EXPECT_DOUBLE_EQ(1.0, newVelocity.x);
    EXPECT_DOUBLE_EQ(0.0, newVelocity.y);
    EXPECT_DOUBLE_EQ(0.0, newVelocity.z);
}
