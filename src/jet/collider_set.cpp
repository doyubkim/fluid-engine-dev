// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/collider_set.h>

namespace jet {

template <size_t N>
ColliderSet<N>::ColliderSet()
    : ColliderSet(Array1<std::shared_ptr<Collider<N>>>()) {}

template <size_t N>
ColliderSet<N>::ColliderSet(
    const ConstArrayView1<std::shared_ptr<Collider<N>>> &others) {
    setSurface(std::make_shared<SurfaceSet<N>>());
    for (const auto &collider : others) {
        addCollider(collider);
    }
}

template <size_t N>
Vector<double, N> ColliderSet<N>::velocityAt(
    const Vector<double, N> &point) const {
    size_t closestCollider = kMaxSize;
    double closestDist = kMaxD;
    for (size_t i = 0; i < _colliders.length(); ++i) {
        double dist = _colliders[i]->surface()->closestDistance(point);
        if (dist < closestDist) {
            closestDist = dist;
            closestCollider = i;
        }
    }
    if (closestCollider != kMaxSize) {
        return _colliders[closestCollider]->velocityAt(point);
    } else {
        return Vector<double, N>();
    }
}

template <size_t N>
void ColliderSet<N>::addCollider(const std::shared_ptr<Collider<N>> &collider) {
    auto surfaceSet = std::dynamic_pointer_cast<SurfaceSet<N>>(surface());
    _colliders.append(collider);
    surfaceSet->addSurface(collider->surface());
}

template <size_t N>
size_t ColliderSet<N>::numberOfColliders() const {
    return _colliders.length();
}

template <size_t N>
std::shared_ptr<Collider<N>> ColliderSet<N>::collider(size_t i) const {
    return _colliders[i];
}

template <size_t N>
typename ColliderSet<N>::Builder ColliderSet<N>::builder() {
    return Builder();
}

template <size_t N>
typename ColliderSet<N>::Builder &ColliderSet<N>::Builder::withColliders(
    const ConstArrayView1<std::shared_ptr<Collider<N>>> &others) {
    _colliders = others;
    return *this;
}

template <size_t N>
ColliderSet<N> ColliderSet<N>::Builder::build() const {
    return ColliderSet(_colliders);
}

template <size_t N>
std::shared_ptr<ColliderSet<N>> ColliderSet<N>::Builder::makeShared() const {
    return std::shared_ptr<ColliderSet>(new ColliderSet(_colliders),
                                        [](ColliderSet *obj) { delete obj; });
}

template class ColliderSet<2>;

template class ColliderSet<3>;

}  // namespace jet
