// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider3.h>

using namespace jet;

RigidBodyCollider3::RigidBodyCollider3(const Surface3Ptr& surface) {
    setSurface(surface);
}

Vector3D RigidBodyCollider3::velocityAt(const Vector3D& point) const {
    return linearVelocity + angularVelocity.cross(point - origin);
}
