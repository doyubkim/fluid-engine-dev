// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/collider.h>

#include <algorithm>

namespace jet {

template <size_t N>
Collider<N>::Collider() {}

template <size_t N>
Collider<N>::~Collider() {}

template <size_t N>
void Collider<N>::resolveCollision(double radius, double restitutionCoefficient,
                                   Vector<double, N> *newPosition,
                                   Vector<double, N> *newVelocity) {
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
        Vector<double, N> targetNormal = colliderPoint.normal;
        Vector<double, N> targetPoint =
            colliderPoint.point + radius * targetNormal;
        Vector<double, N> colliderVelAtTargetPoint = colliderPoint.velocity;

        // Get new candidate relative velocity from the target point.
        Vector<double, N> relativeVel = *newVelocity - colliderVelAtTargetPoint;
        double normalDotRelativeVel = targetNormal.dot(relativeVel);
        Vector<double, N> relativeVelN = normalDotRelativeVel * targetNormal;
        Vector<double, N> relativeVelT = relativeVel - relativeVelN;

        // Check if the velocity is facing opposite direction of the surface
        // normal
        if (normalDotRelativeVel < 0.0) {
            // Apply restitution coefficient to the surface normal component of
            // the velocity
            Vector<double, N> deltaRelativeVelN =
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

template <size_t N>
double Collider<N>::frictionCoefficient() const {
    return _frictionCoeffient;
}

template <size_t N>
void Collider<N>::setFrictionCoefficient(double newFrictionCoeffient) {
    _frictionCoeffient = std::max(newFrictionCoeffient, 0.0);
}

template <size_t N>
const std::shared_ptr<Surface<N>> &Collider<N>::surface() const {
    return _surface;
}

template <size_t N>
void Collider<N>::setSurface(const std::shared_ptr<Surface<N>> &newSurface) {
    _surface = newSurface;
}

template <size_t N>
void Collider<N>::getClosestPoint(const std::shared_ptr<Surface<N>> &surface,
                                  const Vector<double, N> &queryPoint,
                                  ColliderQueryResult *result) const {
    result->distance = surface->closestDistance(queryPoint);
    result->point = surface->closestPoint(queryPoint);
    result->normal = surface->closestNormal(queryPoint);
    result->velocity = velocityAt(queryPoint);
}

bool Collider3::isPenetrating(const ColliderQueryResult& colliderPoint,
                              const Vector3D& position, double radius) {
    // If the new candidate position of the particle is inside
    // the volume defined by the surface OR the new distance to the surface is
    // less than the particle's radius, this particle is in colliding state.
    return _surface->isInside(position) || colliderPoint.distance < radius;
}

template <size_t N>
void Collider<N>::update(double currentTimeInSeconds,
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

template <size_t N>
void Collider<N>::setOnBeginUpdateCallback(
    const OnBeginUpdateCallback &callback) {
    _onUpdateCallback = callback;
}

template class Collider<2>;

template class Collider<3>;

}  // namespace jet
