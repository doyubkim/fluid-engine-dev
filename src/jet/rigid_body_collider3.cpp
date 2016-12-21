// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider3.h>

using namespace jet;

RigidBodyCollider3::RigidBodyCollider3(const Surface3Ptr& surface) {
    setSurface(surface);
}

RigidBodyCollider3::RigidBodyCollider3(
    const Surface3Ptr& surface,
    const Vector3D& linearVelocity_,
    const Vector3D& angularVelocity_,
    const Vector3D& rotationOrigin_)
: linearVelocity(linearVelocity_)
, angularVelocity(angularVelocity_)
, rotationOrigin(rotationOrigin_) {
    setSurface(surface);
}

Vector3D RigidBodyCollider3::velocityAt(const Vector3D& point) const {
    return linearVelocity + angularVelocity.cross(point - rotationOrigin);
}

RigidBodyCollider3::Builder RigidBodyCollider3::builder() {
    return Builder();
}

RigidBodyCollider3::Builder&
RigidBodyCollider3::Builder::withSurface(const Surface3Ptr& surface) {
    _surface = surface;
    return *this;
}

RigidBodyCollider3::Builder&
RigidBodyCollider3::Builder::withLinearVelocity(
    const Vector3D& linearVelocity) {
    _linearVelocity = linearVelocity;
    return *this;
}

RigidBodyCollider3::Builder&
RigidBodyCollider3::Builder::withAngularVelocity(
    const Vector3D& angularVelocity) {
    _angularVelocity = angularVelocity;
    return *this;
}

RigidBodyCollider3::Builder&
RigidBodyCollider3::Builder::withRotationOrigin(
    const Vector3D& rotationOrigin) {
    _rotationOrigin = rotationOrigin;
    return *this;
}

RigidBodyCollider3 RigidBodyCollider3::Builder::build() const {
    return RigidBodyCollider3(
        _surface,
        _linearVelocity,
        _angularVelocity,
        _rotationOrigin);
}

RigidBodyCollider3Ptr RigidBodyCollider3::Builder::makeShared() const {
    return std::shared_ptr<RigidBodyCollider3>(
        new RigidBodyCollider3(
            _surface,
            _linearVelocity,
            _angularVelocity,
            _rotationOrigin),
        [] (RigidBodyCollider3* obj) {
            delete obj;
    });
}
