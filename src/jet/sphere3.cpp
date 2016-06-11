// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/sphere3.h>

#include <limits>

using namespace jet;

Sphere3::Sphere3() {
}

Sphere3::Sphere3(const Vector3D& center_, double radius_) :
    center(center_),
    radius(radius_) {
}

Vector3D Sphere3::closestPoint(const Vector3D& otherPoint) const {
    return radius * closestNormal(otherPoint) + center;
}

double Sphere3::closestDistance(const Vector3D& otherPoint) const {
    return std::fabs(center.distanceTo(otherPoint) - radius);
}

Vector3D Sphere3::actualClosestNormal(const Vector3D& otherPoint) const {
    if (center.isSimilar(otherPoint)) {
        return Vector3D(1, 0, 0);
    } else {
        return (otherPoint - center).normalized();
    }
}

bool Sphere3::intersects(
    const Ray3D& ray) const {
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
            tMax = std::numeric_limits<double>::max();
        }

        if (tMin >= 0.0) {
            return true;
        }
    }

    return false;
}

SurfaceRayIntersection3 Sphere3::closestIntersection(
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
            tMax = std::numeric_limits<double>::max();
        }

        if (tMin < 0.0) {
            intersection.isIntersecting = false;
        } else {
            intersection.isIntersecting = true;
            intersection.t = tMin;
            intersection.point = ray.origin + tMin * ray.direction;
            intersection.normal = (intersection.point - center).normalized();
        }
    } else {
        intersection.isIntersecting = false;
    }

    return intersection;
}

BoundingBox3D Sphere3::boundingBox() const {
    Vector3D r(radius, radius, radius);
    return BoundingBox3D(center - r, center + r);
}
