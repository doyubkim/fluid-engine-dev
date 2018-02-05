// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/sphere2.h>

using namespace jet;

Sphere2::Sphere2(const Transform2& transform_, bool isNormalFlipped_)
    : Surface2(transform_, isNormalFlipped_) {}

Sphere2::Sphere2(const Vector2D& center_, double radius_,
                 const Transform2& transform_, bool isNormalFlipped_)
    : Surface2(transform_, isNormalFlipped_),
      center(center_),
      radius(radius_) {}

Sphere2::Sphere2(const Sphere2& other)
    : Surface2(other), center(other.center), radius(other.radius) {}

Vector2D Sphere2::closestPointLocal(const Vector2D& otherPoint) const {
    return radius * closestNormalLocal(otherPoint) + center;
}

double Sphere2::closestDistanceLocal(const Vector2D& otherPoint) const {
    return std::fabs(center.distanceTo(otherPoint) - radius);
}

Vector2D Sphere2::closestNormalLocal(const Vector2D& otherPoint) const {
    if (center.isSimilar(otherPoint)) {
        return Vector2D(1, 0);
    } else {
        return (otherPoint - center).normalized();
    }
}

bool Sphere2::intersectsLocal(const Ray2D& ray) const {
    Vector2D r = ray.origin - center;
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

SurfaceRayIntersection2 Sphere2::closestIntersectionLocal(
    const Ray2D& ray) const {
    SurfaceRayIntersection2 intersection;
    Vector2D r = ray.origin - center;
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

BoundingBox2D Sphere2::boundingBoxLocal() const {
    Vector2D r(radius, radius);
    return BoundingBox2D(center - r, center + r);
}

Sphere2::Builder Sphere2::builder() { return Builder(); }

Sphere2::Builder& Sphere2::Builder::withCenter(const Vector2D& center) {
    _center = center;
    return *this;
}

Sphere2::Builder& Sphere2::Builder::withRadius(double radius) {
    _radius = radius;
    return *this;
}

Sphere2 Sphere2::Builder::build() const {
    return Sphere2(_center, _radius, _transform, _isNormalFlipped);
}

Sphere2Ptr Sphere2::Builder::makeShared() const {
    return std::shared_ptr<Sphere2>(
        new Sphere2(_center, _radius, _transform, _isNormalFlipped),
        [](Sphere2* obj) { delete obj; });
}
