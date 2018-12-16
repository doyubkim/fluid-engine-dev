// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cuda_array_view.h>
#include <jet/cuda_utils.h>

#include <gtest/gtest.h>

using namespace jet;

namespace {

struct CudaColliderQueryResult {
    float distance;
    float2 point;
    float2 normal;
    float2 velocity;
};

__device__ bool isPenetrating(const CudaColliderQueryResult &colliderPoint,
                              float2 position, float radius) {
    // If the new candidate position of the particle is on the other side of
    // the surface OR the new distance to the surface is less than the
    // particle's radius, this particle is in colliding state.
    return dot(position - colliderPoint.point, colliderPoint.normal) < 0.0f ||
           colliderPoint.distance < radius;
}

__device__ void resolveCollision(float radius, float restitutionCoefficient,
                                 float frictionCoeffient,
                                 CudaColliderQueryResult colliderPoint,
                                 float2 &newPosition, float2 &newVelocity) {
    // Check if the new position is penetrating the surface
    if (isPenetrating(colliderPoint, newPosition, radius)) {
        // Target point is the closest non-penetrating position from the
        // new position.
        float2 targetNormal = colliderPoint.normal;
        float2 targetPoint = colliderPoint.point + radius * targetNormal;
        float2 colliderVelAtTargetPoint = colliderPoint.velocity;

        // Get new candidate relative velocity from the target point.
        float2 relativeVel = newVelocity - colliderVelAtTargetPoint;
        float normalDotRelativeVel = dot(targetNormal, relativeVel);
        float2 relativeVelN = normalDotRelativeVel * targetNormal;
        float2 relativeVelT = relativeVel - relativeVelN;

        // Check if the velocity is facing opposite direction of the surface
        // normal
        if (normalDotRelativeVel < 0.0f) {
            // Apply restitution coefficient to the surface normal component of
            // the velocity
            float2 deltaRelativeVelN =
                (-restitutionCoefficient - 1.0f) * relativeVelN;
            relativeVelN *= -restitutionCoefficient;

            // Apply friction to the tangential component of the velocity
            // From Bridson et al., Robust Treatment of Collisions, Contact and
            // Friction for Cloth Animation, 2002
            // http://graphics.stanford.edu/papers/cloth-sig02/cloth.pdf
            if (dot(relativeVelT, relativeVelT) > 0.0f) {
                float frictionScale =
                    fmaxf(1.0f -
                              frictionCoeffient * length(deltaRelativeVelN) /
                                  length(relativeVelT),
                          0.0f);
                relativeVelT *= frictionScale;
            }

            // Reassemble the components
            newVelocity =
                relativeVelN + relativeVelT + colliderVelAtTargetPoint;
        }

        // Geometric fix
        newPosition = targetPoint;
    }
}

__global__ void resolveCollisionKernel(
    float radius, float restitutionCoefficient, float frictionCoefficient,
    CudaArrayView1<float2> closestPointsOnCollider,
    CudaArrayView1<float2> closestNormalsOnCollider,
    CudaArrayView1<float2> closestVelocitiesOnCollider,
    CudaArrayView1<float2> positions, CudaArrayView1<float2> velocities) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < positions.length()) {
        float2 x = positions[i];
        float2 v = velocities[i];

        CudaColliderQueryResult closestPoint;
        closestPoint.point = closestPointsOnCollider[i];
        closestPoint.distance = length(closestPoint.point - x);
        closestPoint.normal = closestNormalsOnCollider[i];
        closestPoint.velocity = closestVelocitiesOnCollider[i];

        resolveCollision(radius, restitutionCoefficient, frictionCoefficient,
                         closestPoint, x, v);
        positions[i] = x;
        velocities[i] = v;
    }
}
}

TEST(CudaCollider2, ResolveCollision) {
    CudaArray1<float2> closestPointsOnCollider(1);
    CudaArray1<float2> closestNormalsOnCollider(1);
    CudaArray1<float2> closestVelocitiesOnCollider(1);

    // Assuming simple plane with normal (0, 1) and on the origin (0, 0).
    closestNormalsOnCollider[0] = make_float2(0.0f, 1.0f);
    closestVelocitiesOnCollider[0] = make_float2(0, 0);

    // 1. No penetration
    {
        CudaArray1<float2> positions(1);
        CudaArray1<float2> velocities(1);
        positions[0] = make_float2(1.f, 0.1f);
        velocities[0] = make_float2(1.f, 0.f);
        closestPointsOnCollider[0] = make_float2(1.f, 0.f);
        float radius = 0.05f;
        float restitutionCoefficient = 0.5f;
        float frictionCoefficient = 0.0f;

        resolveCollisionKernel<<<1, 1>>>(
            radius, restitutionCoefficient, frictionCoefficient,
            closestPointsOnCollider, closestNormalsOnCollider,
            closestVelocitiesOnCollider, positions, velocities);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing resolveCollisionKernel");

        float2 x = positions[0];
        float2 v = velocities[0];
        EXPECT_FLOAT_EQ(1.0f, x.x);
        EXPECT_FLOAT_EQ(0.1f, x.y);
        EXPECT_FLOAT_EQ(1.0f, v.x);
        EXPECT_FLOAT_EQ(0.0f, v.y);
    }

    // 2. Penetration within radius
    {
        CudaArray1<float2> positions(1);
        CudaArray1<float2> velocities(1);
        positions[0] = make_float2(1.f, 0.1f);
        velocities[0] = make_float2(1.f, 0.f);
        closestPointsOnCollider[0] = make_float2(1.f, 0.f);
        float radius = 0.2;
        float restitutionCoefficient = 0.5;
        float frictionCoefficient = 0.0f;

        resolveCollisionKernel<<<1, 1>>>(
            radius, restitutionCoefficient, frictionCoefficient,
            closestPointsOnCollider, closestNormalsOnCollider,
            closestVelocitiesOnCollider, positions, velocities);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing resolveCollisionKernel");

        float2 x = positions[0];
        float2 v = velocities[0];

        EXPECT_FLOAT_EQ(1.0f, x.x);
        EXPECT_FLOAT_EQ(0.2f, x.y);
    }

    // 3. Sitting
    {
        CudaArray1<float2> positions(1);
        CudaArray1<float2> velocities(1);
        positions[0] = make_float2(1.f, 0.1f);
        velocities[0] = make_float2(1.f, 0.f);
        closestPointsOnCollider[0] = make_float2(1.f, 0.f);
        float radius = 0.1f;
        float restitutionCoefficient = 0.5f;
        float frictionCoefficient = 0.0f;

        resolveCollisionKernel<<<1, 1>>>(
            radius, restitutionCoefficient, frictionCoefficient,
            closestPointsOnCollider, closestNormalsOnCollider,
            closestVelocitiesOnCollider, positions, velocities);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing resolveCollisionKernel");

        float2 x = positions[0];
        float2 v = velocities[0];

        EXPECT_FLOAT_EQ(1.0, x.x);
        EXPECT_FLOAT_EQ(0.1, x.y);
        EXPECT_FLOAT_EQ(1.0, v.x);
        EXPECT_FLOAT_EQ(0.0, v.y);
    }

    // 4. Bounce-back
    {
        CudaArray1<float2> positions(1);
        CudaArray1<float2> velocities(1);
        positions[0] = make_float2(1.f, -1.f);
        velocities[0] = make_float2(1.f, -1.f);
        closestPointsOnCollider[0] = make_float2(1.f, 0.f);
        float radius = 0.1f;
        float restitutionCoefficient = 0.5f;
        float frictionCoefficient = 0.0f;

        resolveCollisionKernel<<<1, 1>>>(
            radius, restitutionCoefficient, frictionCoefficient,
            closestPointsOnCollider, closestNormalsOnCollider,
            closestVelocitiesOnCollider, positions, velocities);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing resolveCollisionKernel");

        float2 x = positions[0];
        float2 v = velocities[0];

        EXPECT_FLOAT_EQ(1.0, x.x);
        EXPECT_FLOAT_EQ(0.1, x.y);
        EXPECT_FLOAT_EQ(1.0, v.x);
        EXPECT_FLOAT_EQ(restitutionCoefficient, v.y);
    }

    // 5. Friction
    {
        CudaArray1<float2> positions(1);
        CudaArray1<float2> velocities(1);
        positions[0] = make_float2(1.f, -1.f);
        velocities[0] = make_float2(1.f, -1.f);
        closestPointsOnCollider[0] = make_float2(1.f, 0.f);
        float radius = 0.1f;
        float restitutionCoefficient = 0.5f;
        float frictionCoefficient = 0.1f;

        resolveCollisionKernel<<<1, 1>>>(
            radius, restitutionCoefficient, frictionCoefficient,
            closestPointsOnCollider, closestNormalsOnCollider,
            closestVelocitiesOnCollider, positions, velocities);
        JET_CUDA_CHECK_LAST_ERROR("Failed executing resolveCollisionKernel");

        float2 x = positions[0];
        float2 v = velocities[0];

        EXPECT_FLOAT_EQ(1.0, x.x);
        EXPECT_FLOAT_EQ(0.1, x.y);
        EXPECT_GT(1.0, v.x);
        EXPECT_FLOAT_EQ(restitutionCoefficient, v.y);
    }
}
