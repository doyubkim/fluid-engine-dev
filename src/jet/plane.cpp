// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/plane.h>

namespace jet {

template <size_t N>
Plane<N>::Plane(const Transform<N> &transform_, bool isNormalFlipped_)
    : Surface<N>(transform_, isNormalFlipped_) {}

template <size_t N>
Plane<N>::Plane(const Vector<double, N> &normal_,
                const Vector<double, N> &point_, const Transform<N> &transform_,
                bool isNormalFlipped_)
    : Surface<N>(transform_, isNormalFlipped_),
      normal(normal_),
      point(point_) {}

template <size_t N>
Plane<N>::Plane(const Plane &other)
    : Surface<N>(other), normal(other.normal), point(other.point) {}

template <size_t N>
bool Plane<N>::isBounded() const {
    return false;
}

template <size_t N>
Vector<double, N> Plane<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    Vector<double, N> r = otherPoint - point;
    return r - normal.dot(r) * normal + point;
}

template <size_t N>
Vector<double, N> Plane<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    UNUSED_VARIABLE(otherPoint);
    return normal;
}

template <size_t N>
bool Plane<N>::intersectsLocal(const Ray<double, N> &ray) const {
    return std::fabs(ray.direction.dot(normal)) > 0;
}

template <size_t N>
SurfaceRayIntersection<N> Plane<N>::closestIntersectionLocal(
    const Ray<double, N> &ray) const {
    SurfaceRayIntersection<N> intersection;
    double dDotN = ray.direction.dot(normal);

    // Check if not parallel
    if (std::fabs(dDotN) > 0) {
        double t = normal.dot(point - ray.origin) / dDotN;
        if (t >= 0.0) {
            intersection.isIntersecting = true;
            intersection.distance = t;
            intersection.point = ray.pointAt(t);
            intersection.normal = normal;
        }
    }

    return intersection;
}

template <size_t N>
BoundingBox<double, N> Plane<N>::boundingBoxLocal() const {
    Vector<double, N> maxCorner = Vector<double, N>::makeConstant(kMaxD);
    for (size_t i = 0; i < N; ++i) {
        Vector<double, N> n;
        Vector<double, N> corner = maxCorner;
        n[i] = 1.0;
        corner[i] = 0.0;
        // See if the plane is axis-aligned and return flat box if true.
        if (std::fabs(normal.dot(n) - 1.0) < kEpsilonD) {
            return BoundingBox<double, N>(point - corner, point + corner);
        }
    }

    // Otherwise, the plane does not have bbox.
    return BoundingBox<double, N>(-maxCorner, maxCorner);
}

template <size_t N>
typename Plane<N>::Builder Plane<N>::builder() {
    return Builder();
}

template <size_t N>
typename Plane<N>::Builder &Plane<N>::Builder::withNormal(
    const Vector<double, N> &normal) {
    _normal = normal;
    return *this;
}

template <size_t N>
typename Plane<N>::Builder &Plane<N>::Builder::withPoint(
    const Vector<double, N> &point) {
    _point = point;
    return *this;
}

template <size_t N>
Plane<N> Plane<N>::Builder::build() const {
    return Plane<N>(_normal, _point, _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<Plane<N>> Plane<N>::Builder::makeShared() const {
    return std::shared_ptr<Plane<N>>(
        new Plane<N>(_normal, _point, _transform, _isNormalFlipped),
        [](Plane<N> *obj) { delete obj; });
}

template class Plane<2>;

template class Plane<3>;

}  // namespace jet
