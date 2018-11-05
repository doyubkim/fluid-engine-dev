// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/surface.h>

#include <algorithm>

namespace jet {

template <size_t N>
Surface<N>::Surface(const Transform<N> &transform_, bool isNormalFlipped_)
    : transform(transform_), isNormalFlipped(isNormalFlipped_) {}

template <size_t N>
Surface<N>::Surface(const Surface &other)
    : transform(other.transform), isNormalFlipped(other.isNormalFlipped) {}

template <size_t N>
Surface<N>::~Surface() {}

template <size_t N>
Vector<double, N> Surface<N>::closestPoint(
    const Vector<double, N> &otherPoint) const {
    return transform.toWorld(closestPointLocal(transform.toLocal(otherPoint)));
}

template <size_t N>
BoundingBox<double, N> Surface<N>::boundingBox() const {
    return transform.toWorld(boundingBoxLocal());
}

template <size_t N>
bool Surface<N>::intersects(const Ray<double, N> &ray) const {
    return intersectsLocal(transform.toLocal(ray));
}

template <size_t N>
double Surface<N>::closestDistance(const Vector<double, N> &otherPoint) const {
    return closestDistanceLocal(transform.toLocal(otherPoint));
}

template <size_t N>
SurfaceRayIntersection<N> Surface<N>::closestIntersection(
    const Ray<double, N> &ray) const {
    auto result = closestIntersectionLocal(transform.toLocal(ray));
    result.point = transform.toWorld(result.point);
    result.normal = transform.toWorldDirection(result.normal);
    result.normal *= (isNormalFlipped) ? -1.0 : 1.0;
    return result;
}

template <size_t N>
Vector<double, N> Surface<N>::closestNormal(
    const Vector<double, N> &otherPoint) const {
    auto result = transform.toWorldDirection(
        closestNormalLocal(transform.toLocal(otherPoint)));
    result *= (isNormalFlipped) ? -1.0 : 1.0;
    return result;
}

template <size_t N>
bool Surface<N>::intersectsLocal(const Ray<double, N> &rayLocal) const {
    auto result = closestIntersectionLocal(rayLocal);
    return result.isIntersecting;
}

template <size_t N>
void Surface<N>::updateQueryEngine() {
    // Do nothing
}

template <size_t N>
bool Surface<N>::isBounded() const {
    return true;
}

template <size_t N>
bool Surface<N>::isValidGeometry() const {
    return true;
}

template <size_t N>
double Surface<N>::closestDistanceLocal(
    const Vector<double, N> &otherPointLocal) const {
    return otherPointLocal.distanceTo(closestPointLocal(otherPointLocal));
}

template class Surface<2>;

template class Surface<3>;

}  // namespace jet
