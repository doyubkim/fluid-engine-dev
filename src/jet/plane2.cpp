// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/plane2.h>

#include <limits>

using namespace jet;

Plane2::Plane2() {
}

Plane2::Plane2(const Vector2D& normal, const Vector2D& point) :
    _normal(normal),
    _point(point) {
}

Plane2::Plane2(const Plane2& other) :
    _normal(other._normal),
    _point(other._point) {
}

Plane2::~Plane2() {
}

const Vector2D& Plane2::normal() const {
    return _normal;
}

void Plane2::setNormal(const Vector2D& normal) {
    _normal = normal;
}

const Vector2D& Plane2::point() const {
    return _point;
}

void Plane2::setPoint(const Vector2D& point) {
    _point = point;
}

Vector2D Plane2::closestPoint(const Vector2D& otherPoint) const {
    Vector2D r = otherPoint - _point;
    return r - _normal.dot(r) * _normal + _point;
}

double Plane2::closestDistance(const Vector2D& otherPoint) const {
    return (otherPoint - closestPoint(otherPoint)).length();
}

Vector2D Plane2::actualClosestNormal(const Vector2D& otherPoint) const {
    UNUSED_VARIABLE(otherPoint);
    return _normal;
}

bool Plane2::intersects(const Ray2D& ray) const {
    return std::fabs(ray.direction.dot(_normal)) > 0;
}

void Plane2::getClosestIntersection(
    const Ray2D& ray,
    SurfaceRayIntersection2* intersection) const {
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

BoundingBox2D Plane2::boundingBox() const {
    static const double eps = std::numeric_limits<double>::epsilon();
    static const double dmax = std::numeric_limits<double>::max();

    if (std::fabs(_normal.dot(Vector2D(1, 0)) - 1.0) < eps) {
        return BoundingBox2D(
            _point - Vector2D(0, dmax),
            _point + Vector2D(0, dmax));
    } else if (std::fabs(_normal.dot(Vector2D(0, 1)) - 1.0) < eps) {
        return BoundingBox2D(
            _point - Vector2D(dmax, 0),
            _point + Vector2D(dmax, 0));
    } else {
        return BoundingBox2D(
            Vector2D(dmax, dmax),
            Vector2D(dmax, dmax));
    }
}
