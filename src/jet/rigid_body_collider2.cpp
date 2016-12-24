// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider2.h>

using namespace jet;

inline Vector2D worldToLocal(const Vector2D& t, double r, const Vector2D& x) {
    // Convert to the local frame
    double cosAngle = std::cos(r);
    double sinAngle = std::sin(r);
    Vector2D xmt = x - t;
    return Vector2D(
        cosAngle * xmt.x + sinAngle * xmt.y,
       -sinAngle * xmt.x + cosAngle * xmt.y);
}

inline Vector2D localToWorld(const Vector2D& t, double r, const Vector2D& x) {
    // Convert to the local frame
    double cosAngle = std::cos(r);
    double sinAngle = std::sin(r);
    return Vector2D(
        cosAngle * x.x - sinAngle * x.y + t.x,
        sinAngle * x.x + cosAngle * x.y + t.y);
}

RigidBodyCollider2::RigidBodyCollider2(const Surface2Ptr& surface) {
    setSurface(surface);
}

RigidBodyCollider2::RigidBodyCollider2(
    const Surface2Ptr& surface,
    const Vector2D& translation_,
    double rotation_,
    const Vector2D& linearVelocity_,
    double angularVelocity_)
: translation(translation_)
, rotation(rotation_)
, linearVelocity(linearVelocity_)
, angularVelocity(angularVelocity_) {
    setSurface(surface);
}

Vector2D RigidBodyCollider2::velocityAt(const Vector2D& point) const {
    Vector2D r = point - translation;
    return linearVelocity + angularVelocity * Vector2D(-r.y, r.x);
}

void RigidBodyCollider2::getClosestPoint(
    const Surface2Ptr& surface,
    const Vector2D& queryPoint,
    ColliderQueryResult* result) const {
    // Convert to the local frame
    Vector2D localQueryPoint = worldToLocal(translation, rotation, queryPoint);

    result->distance = surface->closestDistance(localQueryPoint);
    Vector2D x = surface->closestPoint(localQueryPoint);
    Vector2D n = surface->closestNormal(localQueryPoint);
    Vector2D v = velocityAt(localQueryPoint);

    // Back to world coord.
    result->point = localToWorld(translation, rotation, x);
    result->normal = localToWorld(Vector2D(), rotation, n);
    result->velocity = localToWorld(Vector2D(), rotation, v);
}

RigidBodyCollider2::Builder RigidBodyCollider2::builder() {
    return Builder();
}

RigidBodyCollider2::Builder&
RigidBodyCollider2::Builder::withSurface(const Surface2Ptr& surface) {
    _surface = surface;
    return *this;
}

RigidBodyCollider2::Builder&
RigidBodyCollider2::Builder::withTranslation(const Vector2D& translation) {
    _translation = translation;
    return *this;
}

RigidBodyCollider2::Builder&
RigidBodyCollider2::Builder::withRotation(double rotation) {
    _rotation = rotation;
    return *this;
}

RigidBodyCollider2::Builder&
RigidBodyCollider2::Builder::withLinearVelocity(
    const Vector2D& linearVelocity) {
    _linearVelocity = linearVelocity;
    return *this;
}

RigidBodyCollider2::Builder&
RigidBodyCollider2::Builder::withAngularVelocity(double angularVelocity) {
    _angularVelocity = angularVelocity;
    return *this;
}

RigidBodyCollider2 RigidBodyCollider2::Builder::build() const {
    return RigidBodyCollider2(
        _surface,
        _translation,
        _rotation,
        _linearVelocity,
        _angularVelocity);
}

RigidBodyCollider2Ptr RigidBodyCollider2::Builder::makeShared() const {
    return std::shared_ptr<RigidBodyCollider2>(
        new RigidBodyCollider2(
            _surface,
            _translation,
            _rotation,
            _linearVelocity,
            _angularVelocity),
        [] (RigidBodyCollider2* obj) {
            delete obj;
    });
}
