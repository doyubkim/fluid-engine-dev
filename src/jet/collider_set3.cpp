// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/collider_set3.h>
#include <vector>

using namespace jet;

ColliderSet3::ColliderSet3() {
    setSurface(std::make_shared<SurfaceSet3>());
}

ColliderSet3::ColliderSet3(const std::vector<Collider3Ptr>& others) {
    for (const auto& collider : others) {
        addCollider(collider);
    }
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

ColliderSet3::Builder ColliderSet3::builder() {
    return Builder();
}

ColliderSet3::Builder&
ColliderSet3::Builder::withColliders(
    const std::vector<Collider3Ptr>& others) {
    _colliders = others;
    return *this;
}

ColliderSet3 ColliderSet3::Builder::build() const {
    return ColliderSet3(_colliders);
}

ColliderSet3Ptr ColliderSet3::Builder::makeShared() const {
    return std::shared_ptr<ColliderSet3>(
        new ColliderSet3(_colliders),
        [] (ColliderSet3* obj) {
            delete obj;
        });
}
