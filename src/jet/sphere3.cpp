// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/sphere3.h>

using namespace jet;

Sphere3::Sphere3(const Transform3& transform_, bool isNormalFlipped_)
    : Surface3(transform_, isNormalFlipped_) {}

Sphere3::Sphere3(const Vector3D& center_, double radius_,
                 const Transform3& transform_, bool isNormalFlipped_)
    : Surface3(transform_, isNormalFlipped_),
      center(center_),
      radius(radius_) {}

Sphere3::Sphere3(const Sphere3& other)
    : Surface3(other), center(other.center), radius(other.radius) {}

Vector3D Sphere3::closestPointLocal(const Vector3D& otherPoint) const {
    return radius * closestNormalLocal(otherPoint) + center;
}

double Sphere3::closestDistanceLocal(const Vector3D& otherPoint) const {
    return std::fabs(center.distanceTo(otherPoint) - radius);
}

Vector3D Sphere3::closestNormalLocal(const Vector3D& otherPoint) const {
    if (center.isSimilar(otherPoint)) {
        return Vector3D(1, 0, 0);
    } else {
        return (otherPoint - center).normalized();
    }
}

bool Sphere3::intersectsLocal(const Ray3D& ray) const {
    Vector3D r = ray.origin - center;
    double b = ray.direction.dot(r);
    double c = r.lengthSquared() - square(radius);
    double d = b * b - c;

    if (d > 0.) {
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

SurfaceRayIntersection3 Sphere3::closestIntersectionLocal(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection;
    Vector3D r = ray.origin - center;
    double b = ray.direction.dot(r);
    double c = r.lengthSquared() - square(radius);
    double d = b * b - c;

    if (d > 0.) {
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

BoundingBox3D Sphere3::boundingBoxLocal() const {
    Vector3D r(radius, radius, radius);
    return BoundingBox3D(center - r, center + r);
}

Sphere3::Builder Sphere3::builder() { return Builder(); }

Sphere3::Builder& Sphere3::Builder::withCenter(const Vector3D& center) {
    _center = center;
    return *this;
}

Sphere3::Builder& Sphere3::Builder::withRadius(double radius) {
    _radius = radius;
    return *this;
}

Sphere3 Sphere3::Builder::build() const {
    return Sphere3(_center, _radius, _transform, _isNormalFlipped);
}

Sphere3Ptr Sphere3::Builder::makeShared() const {
    return std::shared_ptr<Sphere3>(
        new Sphere3(_center, _radius, _transform, _isNormalFlipped),
        [](Sphere3* obj) { delete obj; });
}
