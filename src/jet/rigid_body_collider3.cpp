// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider3.h>

using namespace jet;

inline Vector3D worldToLocal(
    const Vector3D& t,
    const QuaternionD& r,
    const Vector3D& x) {
    return r.inverse() * (x - t);
}

inline Vector3D localToWorld(
    const Vector3D& t,
    const QuaternionD& r,
    const Vector3D& x) {
    return (r * x) + t;
}

RigidBodyCollider3::RigidBodyCollider3(const Surface3Ptr& surface) {
    setSurface(surface);
}

RigidBodyCollider3::RigidBodyCollider3(
    const Surface3Ptr& surface,
    const Vector3D& translation_,
    const QuaternionD& rotation_,
    const Vector3D& linearVelocity_,
    const Vector3D& angularVelocity_)
: translation(translation_)
, rotation(rotation_)
, linearVelocity(linearVelocity_)
, angularVelocity(angularVelocity_) {
    setSurface(surface);
}

Vector3D RigidBodyCollider3::velocityAt(const Vector3D& point) const {
    return linearVelocity + angularVelocity.cross(point - translation);
}

void RigidBodyCollider3::getClosestPoint(
    const Surface3Ptr& surface,
    const Vector3D& queryPoint,
    ColliderQueryResult* result) const {
    // Convert to the local frame
    Vector3D localQueryPoint = worldToLocal(translation, rotation, queryPoint);

    result->distance = surface->closestDistance(localQueryPoint);
    Vector3D x = surface->closestPoint(localQueryPoint);
    Vector3D n = surface->closestNormal(localQueryPoint);
    Vector3D v = velocityAt(localQueryPoint);

    // Back to world coord.
    result->point = localToWorld(translation, rotation, x);
    result->normal = localToWorld(Vector3D(), rotation, n);
    result->velocity = localToWorld(Vector3D(), rotation, v);
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
RigidBodyCollider3::Builder::withTranslation(const Vector3D& translation) {
    _translation = translation;
    return *this;
}

RigidBodyCollider3::Builder&
RigidBodyCollider3::Builder::withRotation(const QuaternionD& rotation) {
    _rotation = rotation;
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
        _translation,
        _rotation,
        _linearVelocity,
        _angularVelocity);
}

RigidBodyCollider3Ptr RigidBodyCollider3::Builder::makeShared() const {
    return std::shared_ptr<RigidBodyCollider3>(
        new RigidBodyCollider3(
            _surface,
            _translation,
            _rotation,
            _linearVelocity,
            _angularVelocity),
        [] (RigidBodyCollider3* obj) {
            delete obj;
    });
}
