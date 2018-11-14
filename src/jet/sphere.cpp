// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/sphere.h>

namespace jet {

template <size_t N>
Sphere<N>::Sphere(const Transform<N> &transform_, bool isNormalFlipped_)
    : Surface<N>(transform_, isNormalFlipped_) {}

template <size_t N>
Sphere<N>::Sphere(const Vector<double, N> &center_, double radius_,
                  const Transform<N> &transform_, bool isNormalFlipped_)
    : Surface<N>(transform_, isNormalFlipped_),
      center(center_),
      radius(radius_) {}

template <size_t N>
Sphere<N>::Sphere(const Sphere &other)
    : Surface<N>(other), center(other.center), radius(other.radius) {}

template <size_t N>
Vector<double, N> Sphere<N>::closestPointLocal(
    const Vector<double, N> &otherPoint) const {
    return radius * closestNormalLocal(otherPoint) + center;
}

template <size_t N>
double Sphere<N>::closestDistanceLocal(
    const Vector<double, N> &otherPoint) const {
    return std::fabs(center.distanceTo(otherPoint) - radius);
}

template <size_t N>
Vector<double, N> Sphere<N>::closestNormalLocal(
    const Vector<double, N> &otherPoint) const {
    if (center.isSimilar(otherPoint)) {
        return Vector<double, N>::makeUnitX();
    } else {
        return (otherPoint - center).normalized();
    }
}

template <size_t N>
bool Sphere<N>::intersectsLocal(const Ray<double, N> &ray) const {
    Vector<double, N> r = ray.origin - center;
    double b = ray.direction.dot(r);
    double c = r.lengthSquared() - square(radius);
    double d = b * b - c;

    if (d > 0.0) {
        d = std::sqrt(d);
        double tMin = -b - d;
        double tMax = -b + d;

        if (tMin < 0.0) {
            tMin = tMax;
        }

        if (tMin >= 0.0) {
            return true;
        }
    }

    return false;
}

template <size_t N>
SurfaceRayIntersection<N> Sphere<N>::closestIntersectionLocal(
    const Ray<double, N> &ray) const {
    SurfaceRayIntersection<N> intersection;
    Vector<double, N> r = ray.origin - center;
    double b = ray.direction.dot(r);
    double c = r.lengthSquared() - square(radius);
    double d = b * b - c;

    if (d > 0.0) {
        d = std::sqrt(d);
        double tMin = -b - d;
        double tMax = -b + d;

        if (tMin < 0.0) {
            tMin = tMax;
        }

        if (tMin < 0.0) {
            intersection.isIntersecting = false;
        } else {
            intersection.isIntersecting = true;
            intersection.distance = tMin;
            intersection.point = ray.origin + tMin * ray.direction;
            intersection.normal = (intersection.point - center).normalized();
        }
    } else {
        intersection.isIntersecting = false;
    }

    return intersection;
}

template <size_t N>
BoundingBox<double, N> Sphere<N>::boundingBoxLocal() const {
    Vector<double, N> r = Vector<double, N>::makeConstant(radius);
    return BoundingBox<double, N>(center - r, center + r);
}

template <size_t N>
typename Sphere<N>::Builder Sphere<N>::builder() {
    return Builder();
}

template <size_t N>
typename Sphere<N>::Builder &Sphere<N>::Builder::withCenter(
    const Vector<double, N> &center) {
    _center = center;
    return *this;
}

template <size_t N>
typename Sphere<N>::Builder &Sphere<N>::Builder::withRadius(double radius) {
    _radius = radius;
    return *this;
}

template <size_t N>
Sphere<N> Sphere<N>::Builder::build() const {
    return Sphere(_center, _radius, _transform, _isNormalFlipped);
}

template <size_t N>
std::shared_ptr<Sphere<N>> Sphere<N>::Builder::makeShared() const {
    return std::shared_ptr<Sphere>(
        new Sphere(_center, _radius, _transform, _isNormalFlipped),
        [](Sphere *obj) { delete obj; });
}

template class Sphere<2>;

template class Sphere<3>;

}  // namespace jet