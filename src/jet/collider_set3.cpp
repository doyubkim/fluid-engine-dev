// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/collider_set3.h>

using namespace jet;

ColliderSet3::ColliderSet3() {
    setSurface(std::make_shared<SurfaceSet3>());
}

Vector3D ColliderSet3::velocityAt(const Vector3D& point) const {
    size_t closestCollider = kMaxSize;
    double closestDist = kMaxD;
    for (size_t i = 0; i < _colliders.size(); ++i) {
        double dist = _colliders[i]->surface()->closestDistance(point);
        if (dist < closestDist) {
            closestDist = dist;
            closestCollider = i;
        }
    }
    if (closestCollider != kMaxSize) {
        return _colliders[closestCollider]->velocityAt(point);
    } else {
        return Vector3D();
    }
}

void ColliderSet3::addCollider(const Collider3Ptr& collider) {
    auto surfaceSet = std::dynamic_pointer_cast<SurfaceSet3>(surface());
    _colliders.push_back(collider);
    surfaceSet->addSurface(collider->surface());
}
