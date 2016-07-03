// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/rigid_body_collider2.h>

using namespace jet;

RigidBodyCollider2::RigidBodyCollider2(const Surface2Ptr& surface) {
    setSurface(surface);
}

Vector2D RigidBodyCollider2::velocityAt(const Vector2D& point) const {
    Vector2D r = point - origin;
    return linearVelocity + angularVelocity * Vector2D(-r.y, r.x);
}
