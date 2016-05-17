// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/plane3.h>

#include <limits>

using namespace jet;

Plane3::Plane3() {
}

Plane3::Plane3(const Vector3D& normal, const Vector3D& point) :
    _normal(normal),
    _point(point) {
}

Plane3::Plane3(
    const Vector3D& point0,
    const Vector3D& point1,
    const Vector3D& point2) {
    _normal = (point1 - point0).cross(point2 - point0).normalized();
    _point = point0;
}

Plane3::Plane3(const Plane3& other) :
    _normal(other._normal),
    _point(other._point) {
}

Plane3::~Plane3() {
}

const Vector3D& Plane3::normal() const {
    return _normal;
}

void Plane3::setNormal(const Vector3D& normal) {
    _normal = normal;
}

const Vector3D& Plane3::point() const {
    return _point;
}

void Plane3::setPoint(const Vector3D& point) {
    _point = point;
}

Vector3D Plane3::closestPoint(const Vector3D& otherPoint) const {
    Vector3D r = otherPoint - _point;
    return r - _normal.dot(r) * _normal + _point;
}

double Plane3::closestDistance(const Vector3D& otherPoint) const {
    return (otherPoint - closestPoint(otherPoint)).length();
}

Vector3D Plane3::actualClosestNormal(const Vector3D& otherPoint) const {
    UNUSED_VARIABLE(otherPoint);
    return _normal;
}

bool Plane3::intersects(const Ray3D& ray) const {
    return std::fabs(ray.direction.dot(_normal)) > 0;
}

void Plane3::getClosestIntersection(
    const Ray3D& ray,
    SurfaceRayIntersection3* intersection) const {
    double dDotN = ray.direction.dot(_normal);

    if (std::fabs(dDotN) > 0) {
        double t = _normal.dot(_point - ray.origin) / dDotN;

        intersection->isIntersecting = true;
        intersection->t = t;
        intersection->point = ray.pointAt(t);
        intersection->normal = _normal;
    } else {
        intersection->isIntersecting = false;
    }
}

BoundingBox3D Plane3::boundingBox() const {
    static const double eps = std::numeric_limits<double>::epsilon();
    static const double dmax = std::numeric_limits<double>::max();

    if (std::fabs(_normal.dot(Vector3D(1, 0, 0)) - 1.0) < eps) {
        return BoundingBox3D(
            _point - Vector3D(0, dmax, dmax),
            _point + Vector3D(0, dmax, dmax));
    } else if (std::fabs(_normal.dot(Vector3D(0, 1, 0)) - 1.0) < eps) {
        return BoundingBox3D(
            _point - Vector3D(dmax, 0, dmax),
            _point + Vector3D(dmax, 0, dmax));
    } else if (std::fabs(_normal.dot(Vector3D(0, 0, 1)) - 1.0) < eps) {
        return BoundingBox3D(
            _point - Vector3D(dmax, dmax, 0),
            _point + Vector3D(dmax, dmax, 0));
    } else {
        return BoundingBox3D(
            Vector3D(dmax, dmax, dmax),
            Vector3D(dmax, dmax, dmax));
    }
}
