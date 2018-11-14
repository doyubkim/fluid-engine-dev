// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/rigid_body_collider.h>

namespace jet {

template <size_t N>
RigidBodyCollider<N>::RigidBodyCollider(
    const std::shared_ptr<Surface<N>> &surface) {
    setSurface(surface);
}

template <size_t N>
RigidBodyCollider<N>::RigidBodyCollider(
    const std::shared_ptr<Surface<N>> &surface,
    const Vector<double, N> &linearVelocity_,
    const AngularVelocity<N> &angularVelocity_)
    : linearVelocity(linearVelocity_), angularVelocity(angularVelocity_) {
    setSurface(surface);
}

template <size_t N>
Vector<double, N> RigidBodyCollider<N>::velocityAt(
    const Vector<double, N> &point) const {
    Vector<double, N> r = point - surface()->transform.translation();
    return linearVelocity + angularVelocity.cross(r);
}

template <size_t N>
typename RigidBodyCollider<N>::Builder RigidBodyCollider<N>::builder() {
    return Builder();
}

template <size_t N>
typename RigidBodyCollider<N>::Builder &
RigidBodyCollider<N>::Builder::withSurface(
    const std::shared_ptr<Surface<N>> &surface) {
    _surface = surface;
    return *this;
}

template <size_t N>
typename RigidBodyCollider<N>::Builder &
RigidBodyCollider<N>::Builder::withLinearVelocity(
    const Vector<double, N> &linearVelocity) {
    _linearVelocity = linearVelocity;
    return *this;
}

template <size_t N>
typename RigidBodyCollider<N>::Builder &
RigidBodyCollider<N>::Builder::withAngularVelocity(
    const AngularVelocity<N> &angularVelocity) {
    _angularVelocity = angularVelocity;
    return *this;
}

template <size_t N>
RigidBodyCollider<N> RigidBodyCollider<N>::Builder::build() const {
    return RigidBodyCollider(_surface, _linearVelocity, _angularVelocity);
}

template <size_t N>
std::shared_ptr<RigidBodyCollider<N>>
RigidBodyCollider<N>::Builder::makeShared() const {
    return std::shared_ptr<RigidBodyCollider>(
        new RigidBodyCollider(_surface, _linearVelocity, _angularVelocity),
        [](RigidBodyCollider *obj) { delete obj; });
}

template class RigidBodyCollider<2>;

template class RigidBodyCollider<3>;

}  // namespace jet
