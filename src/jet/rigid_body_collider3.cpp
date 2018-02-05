// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/rigid_body_collider3.h>

using namespace jet;

RigidBodyCollider3::RigidBodyCollider3(const Surface3Ptr& surface) {
    setSurface(surface);
}

RigidBodyCollider3::RigidBodyCollider3(
    const Surface3Ptr& surface,
    const Vector3D& linearVelocity_,
    const Vector3D& angularVelocity_)
: linearVelocity(linearVelocity_)
, angularVelocity(angularVelocity_) {
    setSurface(surface);
}

Vector3D RigidBodyCollider3::velocityAt(const Vector3D& point) const {
    Vector3D r = point - surface()->transform.translation();
    return linearVelocity + angularVelocity.cross(r);
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

RigidBodyCollider3 RigidBodyCollider3::Builder::build() const {
    return RigidBodyCollider3(
        _surface,
        _linearVelocity,
        _angularVelocity);
}

RigidBodyCollider3Ptr RigidBodyCollider3::Builder::makeShared() const {
    return std::shared_ptr<RigidBodyCollider3>(
        new RigidBodyCollider3(
            _surface,
            _linearVelocity,
            _angularVelocity),
        [] (RigidBodyCollider3* obj) {
            delete obj;
    });
}
