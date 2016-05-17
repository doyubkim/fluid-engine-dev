// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider3.h>

using namespace jet;

RigidBodyCollider3::RigidBodyCollider3(const Surface3Ptr& surface) {
    setSurface(surface);
}

Vector3D RigidBodyCollider3::velocityAt(const Vector3D& point) const {
    return _linearVelocity + _angularVelocity.cross(point - _origin);
}

const Vector3D& RigidBodyCollider3::linearVelocity() const {
    return _linearVelocity;
}

void RigidBodyCollider3::setLinearVelocity(const Vector3D& newVelocity) {
    _linearVelocity = newVelocity;
}

const Vector3D& RigidBodyCollider3::angularVelocity() const {
    return _angularVelocity;
}

void RigidBodyCollider3::setAngularVelocity(const Vector3D& newVelocity) {
    _angularVelocity = newVelocity;
}

const Vector3D& RigidBodyCollider3::origin() const {
    return _origin;
}

void RigidBodyCollider3::setOrigin(const Vector3D& newOrigin) {
    _origin = newOrigin;
}
