// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/rigid_body_collider2.h>

using namespace jet;

RigidBodyCollider2::RigidBodyCollider2(const Surface2Ptr& surface) {
    setSurface(surface);
}

RigidBodyCollider2::RigidBodyCollider2(
    const Surface2Ptr& surface,
    const Vector2D& linearVelocity_,
    double angularVelocity_)
: linearVelocity(linearVelocity_)
, angularVelocity(angularVelocity_) {
    setSurface(surface);
}

Vector2D RigidBodyCollider2::velocityAt(const Vector2D& point) const {
    Vector2D r = point - surface()->transform.translation();
    return linearVelocity + angularVelocity * Vector2D(-r.y, r.x);
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
        _linearVelocity,
        _angularVelocity);
}

RigidBodyCollider2Ptr RigidBodyCollider2::Builder::makeShared() const {
    return std::shared_ptr<RigidBodyCollider2>(
        new RigidBodyCollider2(
            _surface,
            _linearVelocity,
            _angularVelocity),
        [] (RigidBodyCollider2* obj) {
            delete obj;
    });
}
