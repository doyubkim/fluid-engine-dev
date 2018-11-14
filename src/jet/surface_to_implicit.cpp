// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/surface_to_implicit.h>

namespace jet {

template <size_t N>
SurfaceToImplicit<N>::SurfaceToImplicit(
    const std::shared_ptr<Surface<N>> &surface, const Transform<N> &transform,
    bool isNormalFlipped)
    : ImplicitSurface<N>(transform, isNormalFlipped), _surface(surface) {}

template <size_t N>
SurfaceToImplicit<N>::SurfaceToImplicit(const SurfaceToImplicit &other)
    : ImplicitSurface<N>(other), _surface(other._surface) {}

template <size_t N>
std::shared_ptr<Surface<N>> SurfaceToImplicit<N>::surface() const {
    return _surface;
}

template <size_t N>
Vector<double, N> SurfaceToImplicit<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    return _surface->closestPoint(otherPoint);
}

template <size_t N>
Vector<double, N> SurfaceToImplicit<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    return _surface->closestNormal(otherPoint);
}

template <size_t N>
double SurfaceToImplicit<N>::closestDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    return _surface->closestDistance(otherPoint);
}

template <size_t N>
bool SurfaceToImplicit<N>::intersectsLocal(const Ray<double, N> &ray) const {
    return _surface->intersects(ray);
}

template <size_t N>
SurfaceRayIntersection<N> SurfaceToImplicit<N>::closestIntersectionLocal(
    const Ray<double, N> &ray) const {
    return _surface->closestIntersection(ray);
}

template <size_t N>
BoundingBox<double, N> SurfaceToImplicit<N>::boundingBoxLocal() const {
    return _surface->boundingBox();
}

template <size_t N>
double SurfaceToImplicit<N>::signedDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    Vector<double, N> x = _surface->closestPoint(otherPoint);
    Vector<double, N> n = _surface->closestNormal(otherPoint);
    n = (isNormalFlipped) ? Vector<double, N>{-n} : n;
    if (n.dot(otherPoint - x) < 0.0) {
        return -x.distanceTo(otherPoint);
    } else {
        return x.distanceTo(otherPoint);
    }
}

template <size_t N>
typename SurfaceToImplicit<N>::Builder &SurfaceToImplicit<N>::Builder::withSurface(
    const std::shared_ptr<Surface<N>> &surface) {
    _surface = surface;
    return *this;
}

template <size_t N>
SurfaceToImplicit<N> SurfaceToImplicit<N>::Builder::build() const {
    return SurfaceToImplicit(_surface, _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<SurfaceToImplicit<N>> SurfaceToImplicit<N>::Builder::makeShared()
    const {
    return std::shared_ptr<SurfaceToImplicit>(
        new SurfaceToImplicit(_surface, _transform, _isNormalFlipped),
        [](SurfaceToImplicit *obj) { delete obj; });
}

template class SurfaceToImplicit<2>;

template class SurfaceToImplicit<3>;

}  // namespace jet
