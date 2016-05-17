// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider2.h>

using namespace jet;

RigidBodyCollider2::RigidBodyCollider2(const Surface2Ptr& surface) {
    setSurface(surface);
}

Vector2D RigidBodyCollider2::velocityAt(const Vector2D& point) const {
    return _linearVelocity + _angularVelocity.cross(point - _origin);
}

const Vector2D& RigidBodyCollider2::linearVelocity() const {
    return _linearVelocity;
}

void RigidBodyCollider2::setLinearVelocity(const Vector2D& newVelocity) {
    _linearVelocity = newVelocity;
}

const Vector2D& RigidBodyCollider2::angularVelocity() const {
    return _angularVelocity;
}

void RigidBodyCollider2::setAngularVelocity(const Vector2D& newVelocity) {
    _angularVelocity = newVelocity;
}

const Vector2D& RigidBodyCollider2::origin() const {
    return _origin;
}

void RigidBodyCollider2::setOrigin(const Vector2D& newOrigin) {
    _origin = newOrigin;
}
