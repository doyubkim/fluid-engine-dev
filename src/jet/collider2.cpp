// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/collider2.h>

#include <algorithm>

using namespace jet;

Collider2::Collider2() {}

Collider2::~Collider2() {}

void Collider2::resolveCollision(double radius, double restitutionCoefficient,
                                 Vector2D* newPosition, Vector2D* newVelocity) {
    JET_ASSERT(_surface);

    if (!_surface->isValidGeometry()) {
        return;
    }

    ColliderQueryResult colliderPoint;

    getClosestPoint(_surface, *newPosition, &colliderPoint);

    // Check if the new position is penetrating the surface
    if (isPenetrating(colliderPoint, *newPosition, radius)) {
        // Target point is the closest non-penetrating position from the
        // new position.
        Vector2D targetNormal = colliderPoint.normal;
        Vector2D targetPoint = colliderPoint.point + radius * targetNormal;
        Vector2D colliderVelAtTargetPoint = colliderPoint.velocity;

        // Get new candidate relative velocity from the target point.
        Vector2D relativeVel = *newVelocity - colliderVelAtTargetPoint;
        double normalDotRelativeVel = targetNormal.dot(relativeVel);
        Vector2D relativeVelN = normalDotRelativeVel * targetNormal;
        Vector2D relativeVelT = relativeVel - relativeVelN;

        // Check if the velocity is facing opposite direction of the surface
        // normal
        if (normalDotRelativeVel < 0.0) {
            // Apply restitution coefficient to the surface normal component of
            // the velocity
            Vector2D deltaRelativeVelN =
                (-restitutionCoefficient - 1.0) * relativeVelN;
            relativeVelN *= -restitutionCoefficient;

            // Apply friction to the tangential component of the velocity
            // From Bridson et al., Robust Treatment of Collisions, Contact and
            // Friction for Cloth Animation, 2002
            // http://graphics.stanford.edu/papers/cloth-sig02/cloth.pdf
            if (relativeVelT.lengthSquared() > 0.0) {
                double frictionScale = std::max(
                    1.0 - _frictionCoeffient * deltaRelativeVelN.length() /
                              relativeVelT.length(),
                    0.0);
                relativeVelT *= frictionScale;
            }

            // Reassemble the components
            *newVelocity =
                relativeVelN + relativeVelT + colliderVelAtTargetPoint;
        }

        // Geometric fix
        *newPosition = targetPoint;
    }
}

double Collider2::frictionCoefficient() const { return _frictionCoeffient; }

void Collider2::setFrictionCoefficient(double newFrictionCoeffient) {
    _frictionCoeffient = std::max(newFrictionCoeffient, 0.0);
}

const Surface2Ptr& Collider2::surface() const { return _surface; }

void Collider2::setSurface(const Surface2Ptr& newSurface) {
    _surface = newSurface;
}

void Collider2::getClosestPoint(const Surface2Ptr& surface,
                                const Vector2D& queryPoint,
                                ColliderQueryResult* result) const {
    result->distance = surface->closestDistance(queryPoint);
    result->point = surface->closestPoint(queryPoint);
    result->normal = surface->closestNormal(queryPoint);
    result->velocity = velocityAt(queryPoint);
}

bool Collider2::isPenetrating(const ColliderQueryResult& colliderPoint,
                              const Vector2D& position, double radius) {
    // If the new candidate position of the particle is inside
    // the volume defined by the surface OR the new distance to the surface is
    // less than the particle's radius, this particle is in colliding state.
    return _surface->isInside(position) || colliderPoint.distance < radius;
}

void Collider2::update(double currentTimeInSeconds,
                       double timeIntervalInSeconds) {
    JET_ASSERT(_surface);

    if (!_surface->isValidGeometry()) {
        return;
    }

    _surface->updateQueryEngine();

    if (_onUpdateCallback) {
        _onUpdateCallback(this, currentTimeInSeconds, timeIntervalInSeconds);
    }
}

void Collider2::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback& callback) {
    _onUpdateCallback = callback;
}
