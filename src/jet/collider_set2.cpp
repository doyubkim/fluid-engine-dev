// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/collider_set2.h>
#include <vector>

using namespace jet;

ColliderSet2::ColliderSet2() : ColliderSet2(std::vector<Collider2Ptr>()) {
}

ColliderSet2::ColliderSet2(const std::vector<Collider2Ptr>& others) {
    setSurface(std::make_shared<SurfaceSet2>());
    for (const auto& collider : others) {
        addCollider(collider);
    }
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

size_t ColliderSet2::numberOfColliders() const {
    return _colliders.size();
}

Collider2Ptr ColliderSet2::collider(size_t i) const {
    return _colliders[i];
}

ColliderSet2::Builder ColliderSet2::builder() {
    return Builder();
}

ColliderSet2::Builder&
ColliderSet2::Builder::withColliders(
    const std::vector<Collider2Ptr>& others) {
    _colliders = others;
    return *this;
}

ColliderSet2 ColliderSet2::Builder::build() const {
    return ColliderSet2(_colliders);
}

ColliderSet2Ptr ColliderSet2::Builder::makeShared() const {
    return std::shared_ptr<ColliderSet2>(
        new ColliderSet2(_colliders),
        [] (ColliderSet2* obj) {
            delete obj;
        });
}
