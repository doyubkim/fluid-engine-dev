// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/plane3.h>

#include <limits>

using namespace jet;

Plane3::Plane3() {
}

Plane3::Plane3(const Vector3D& normal, const Vector3D& point) :
    normal(normal),
    point(point) {
}

Plane3::Plane3(
    const Vector3D& point0,
    const Vector3D& point1,
    const Vector3D& point2) {
    normal = (point1 - point0).cross(point2 - point0).normalized();
    point = point0;
}

Plane3::Plane3(const Plane3& other) :
    Surface3(other),
    normal(other.normal),
    point(other.point) {
}

Vector3D Plane3::closestPoint(const Vector3D& otherPoint) const {
    Vector3D r = otherPoint - point;
    return r - normal.dot(r) * normal + point;
}

double Plane3::closestDistance(const Vector3D& otherPoint) const {
    return (otherPoint - closestPoint(otherPoint)).length();
}

Vector3D Plane3::actualClosestNormal(const Vector3D& otherPoint) const {
    UNUSED_VARIABLE(otherPoint);
    return normal;
}

bool Plane3::intersects(const Ray3D& ray) const {
    return std::fabs(ray.direction.dot(normal)) > 0;
}

SurfaceRayIntersection3 Plane3::actualClosestIntersection(
    const Ray3D& ray) const {
    SurfaceRayIntersection3 intersection;

    double dDotN = ray.direction.dot(normal);

    // Check if not parallel
    if (std::fabs(dDotN) > 0) {
        double t = normal.dot(point - ray.origin) / dDotN;
        if (t >= 0.0) {
            intersection.isIntersecting = true;
            intersection.t = t;
            intersection.point = ray.pointAt(t);
            intersection.normal = normal;
        }
    }

    return intersection;
}

BoundingBox3D Plane3::boundingBox() const {
    static const double eps = std::numeric_limits<double>::epsilon();
    static const double dmax = std::numeric_limits<double>::max();

    if (std::fabs(normal.dot(Vector3D(1, 0, 0)) - 1.0) < eps) {
        return BoundingBox3D(
            point - Vector3D(0, dmax, dmax),
            point + Vector3D(0, dmax, dmax));
    } else if (std::fabs(normal.dot(Vector3D(0, 1, 0)) - 1.0) < eps) {
        return BoundingBox3D(
            point - Vector3D(dmax, 0, dmax),
            point + Vector3D(dmax, 0, dmax));
    } else if (std::fabs(normal.dot(Vector3D(0, 0, 1)) - 1.0) < eps) {
        return BoundingBox3D(
            point - Vector3D(dmax, dmax, 0),
            point + Vector3D(dmax, dmax, 0));
    } else {
        return BoundingBox3D(
            Vector3D(dmax, dmax, dmax),
            Vector3D(dmax, dmax, dmax));
    }
}
