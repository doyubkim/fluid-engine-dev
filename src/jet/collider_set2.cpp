// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/collider_set2.h>

using namespace jet;

ColliderSet2::ColliderSet2() {
    setSurface(std::make_shared<SurfaceSet2>());
}

Vector2D ColliderSet2::velocityAt(const Vector2D& point) const {
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
        return Vector2D();
    }
}

void ColliderSet2::addCollider(const Collider2Ptr& collider) {
    auto surfaceSet = std::dynamic_pointer_cast<SurfaceSet2>(surface());
    _colliders.push_back(collider);
    surfaceSet->addSurface(collider->surface());
}
